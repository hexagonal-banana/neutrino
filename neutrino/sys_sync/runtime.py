"""Build the opt-in native interceptor and prepare immutable image catalogs.

No GPU initialization occurs here. JIT recipes are accepted only with matching
unprobed SASS/resources; all compiler children run without tracing or devices.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .build import load_reconstructed_group
from .linkage import link_group
from .probe import instrument_instructions, validate_loop_depth
from .ptx import PTXError, SITE_SELECTION, parse
from .sampling import CONTEXT_ABI, policy_json
from ..utils.reconstruct_cuda import disassembly, sha256

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[1] / 'src'


def compiler_environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith('NEUTRINO_SYS_')}
    env.pop('LD_PRELOAD', None)
    env.pop('LD_AUDIT', None)
    env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def implementation_identity():
    paths = [*Path(__file__).parent.glob('*.py'), *SRC.glob('sys_*')]
    return {str(path.relative_to(SRC.parent)): sha256(path) for path in sorted(paths)}


def compile_native(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    toolchain = Path(os.environ['UNIFIED_TOOLCHAIN']).resolve()
    compiler = os.environ['CXX']
    output = destination / 'sys-intercept.so'
    json_include = ROOT / 'third_party/nccl-unified/contrib/nccl_m2n/third_party'
    command = [compiler, '-std=c++17', '-O2', '-g', '-shared', '-fPIC', '-pthread',
               '-Wall', '-Wextra', '-Werror', '-Wno-deprecated-declarations',
               '-I' + str(toolchain / 'targets/x86_64-linux/include'),
               '-I' + str(toolchain / 'include'), '-I' + str(json_include),
               str(SRC / 'sys_runtime.cpp'), '-L' + str(toolchain / 'lib'),
               '-Wl,-rpath,' + str(toolchain / 'lib'), '-Wl,-z,defs', '-Wl,-Bsymbolic-functions',
               '-ldl', '-lcrypto', '-lcupti', '-o', str(output)]
    identity = implementation_identity()
    if any(value == hashlib.sha256(b'').hexdigest() for name, value in identity.items() if name.startswith('src/')):
        raise PTXError('Native source is empty; refusing an empty shared-library build')
    with (destination / 'build.log').open('x') as log:
        subprocess.run(command, env=compiler_environment(), stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=180)
    if implementation_identity() != identity:
        raise PTXError('Runtime sources changed during compilation')
    symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(output)], env=compiler_environment(), text=True)
    exported = {line.split()[-1] for line in symbols.splitlines() if line.split()}
    required = {'neutrinoSysAbiVersion', 'neutrinoSysCaptureBegin', 'neutrinoSysCaptureEnd',
                'cuLibraryLoadFromFile', 'cuLaunchKernelEx', '__cudaRegisterFatBinary', 'dlsym'}
    if not required <= exported:
        raise PTXError('Missing native interception exports: ' + str(sorted(required - exported)))
    report = {'schema': 1, 'status': 'built', 'command': command, 'library': str(output),
              'sha256': sha256(output), 'implementation': identity,
              'json_header_sha256': sha256(json_include / 'nlohmann/json.hpp'),
              'cuda_header_sha256': sha256(toolchain / 'targets/x86_64-linux/include/cuda.h'),
              'cupti_header_sha256': sha256(toolchain / 'targets/x86_64-linux/include/cupti_activity.h'),
              'cupti_library_sha256': sha256(toolchain / 'lib/libcupti.so')}
    (destination / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def image_manifest(original, replacement, context, label, *, aliases=()):
    if type(context.get('context_abi')) is not int or context['context_abi'] != CONTEXT_ABI:
        raise PTXError('Incompatible sampling context ABI; rebuild transformed images')
    entries = {row['function']: row['original_layout'] for row in context['layouts'] if row['kind'] == 'entry'}
    if len(entries) != sum(row['kind'] == 'entry' for row in context['layouts']):
        raise PTXError('Ambiguous entry ownership')
    return {'id': sha256(original), 'original': str(Path(original).resolve()),
            'original_hashes': sorted({sha256(original), *(sha256(path) for path in aliases)}),
            'replacement': str(Path(replacement).resolve()), 'replacement_sha256': sha256(replacement),
            'label': label, 'entries': entries, 'probe': context, 'context_abi': CONTEXT_ABI,
            'interval_mode': context['interval_mode'], 'loop_depth': context.get('loop_depth', 1),
            'runtime_verified': False}


def catalog_loop_depth(catalog, library):
    """Resolve independent DeepEP JIT depth, with legacy uniform-depth fallback."""
    mode = catalog['interval_mode']
    depth = validate_loop_depth(mode, catalog.get('loop_depth', 1))
    jit_depth = validate_loop_depth(mode, catalog.get('jit_loop_depth', depth))
    return jit_depth if library == 'DeepEP' else depth


def prepare_nccl(build_paths, destination, *, jit_roots=(), jit_loop_depth=None):
    """Package previously verified transformed linked/standalone image owners."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    images, modes, depths = [], set(), set()
    toolchain = Path(os.environ['UNIFIED_TOOLCHAIN'])
    for build_path in build_paths:
        build_path = Path(build_path).resolve()
        report = json.loads(build_path.read_text())
        if (report.get('status') != 'passed' or not report.get('assembly_verified') or
                report.get('interval_mode') not in {'instruction-only', 'loop-boundary'}):
            raise PTXError('Passing system-probe build required')
        modes.add(report['interval_mode'])
        depth = validate_loop_depth(report['interval_mode'], report.get('loop_depth', 1))
        depths.add(depth)
        reconstruction = Path(report['reconstruction'])
        if sha256(reconstruction) != report['reconstruction_sha256']:
            raise PTXError('Reconstruction changed')
        original_report = json.loads(reconstruction.read_text())
        link_index = next(i for i, row in enumerate(original_report['links'])
                          if row['rebuilt'] == report['linked_image'])
        _, target, link, items, _ = load_reconstructed_group(reconstruction, link_index)
        by_unit = {row['unit']: row for row in report['assemblies']}
        for context in report['contexts']:
            if context['interval_mode'] != report['interval_mode'] or context.get('loop_depth', 1) != depth:
                raise PTXError('Build/context loop configuration mismatch')
            owner = context['owner']
            if owner == 'linked':
                original, relative, expected = Path(link['original']), report['linked_image'], report['linked_sha256']
            else:
                item = next(row for row in items if row['rebuilt'] == owner)
                original, relative, expected = Path(item['original']), owner, by_unit[owner]['cubin_sha256']
            cubin = build_path.parent / relative
            if sha256(cubin) != expected:
                raise PTXError('Transformed image changed')
            # NVCC retained the exact registration fatbin alongside each cubin.
            fatbin = original.with_name(original.name.replace('.' + target + '.cubin', '.fatbin'))
            if not fatbin.is_file():
                raise PTXError(f'Missing original registration image: {fatbin}')
            output = destination / (sha256(original) + '.fatbin')
            command = [str(toolchain / 'bin/fatbinary'), '--64', '--create=' + str(output),
                       '--image3=kind=elf,sm=90,file=' + str(cubin)]
            subprocess.run(command, env=compiler_environment(), check=True, capture_output=True, timeout=60)
            image = image_manifest(original, output, context, 'NCCL', aliases=(fatbin,))
            for name, layout in image['entries'].items():
                # NCCL enqueue.cc sends only the live, 16-byte-aligned prefix of
                # its fixed-capacity work blob. Never read past that host buffer.
                if 'ncclDevKernelArgsStorage' in name:
                    parameters = layout['parameters']
                    if len(parameters) != 1 or parameters[0]['offset'] != 0 or '.b8' not in parameters[0]['declaration']:
                        raise PTXError('NCCL packed-prefix storage contract changed')
                    layout['packed_prefix_min'] = 16
            image['target'] = target
            image['build_manifest'] = str(build_path)
            image['build_manifest_sha256'] = sha256(build_path)
            image['pack_command'] = command
            images.append(image)
    if len(modes) != 1:
        raise PTXError('Mixed interval modes in one catalog')
    if len(depths) != 1:
        raise PTXError('Mixed loop depths in one catalog')
    result = {'schema': 1, 'context_abi': CONTEXT_ABI, 'interval_mode': modes.pop(), 'loop_depth': depths.pop(), 'images': images,
              'jit_roots': [str(Path(root).resolve(strict=True)) for root in jit_roots]}
    if jit_loop_depth is not None:
        result['jit_loop_depth'] = validate_loop_depth(result['interval_mode'], jit_loop_depth)
    (destination / 'catalog.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def jit_recipe(original):
    original = Path(original).resolve()
    metadata = json.loads(original.with_name('meta.json').read_text())
    options = metadata['compiler_options']
    ptx = original.with_suffix('.ptx')
    module = parse(ptx.read_text())
    arch = 'sm_' + str(options['arch'])
    if module.target != arch or options.get('optimize_level') != '3':
        raise PTXError('JIT architecture/optimization does not match supported recipe')
    flags = ['-arch=' + arch, '-O3', '--register-usage-level=' + str(options['ptxas_register_usage_level'])]
    if options.get('with_line_info'):
        flags.append('-lineinfo')
    # Additional assembler options cannot be guessed; equivalence remains required.
    return metadata, ptx, flags


def prepare_jit(original, cache, *, interval_mode='instruction-only', loop_depth=1):
    if interval_mode not in {'instruction-only', 'loop-boundary'}:
        raise PTXError('Unsupported JIT interval mode')
    validate_loop_depth(interval_mode, loop_depth)
    original, cache = Path(original).resolve(), Path(cache).resolve()
    metadata, ptx, flags = jit_recipe(original)
    identity = {'original': sha256(original), 'ptx': sha256(ptx), 'metadata': metadata,
                'flags': flags, 'implementation': implementation_identity(), 'interval_mode': interval_mode,
                'loop_depth': loop_depth, 'context_abi': CONTEXT_ABI}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        directory = cache / key
        result_path = directory / 'image.json'
        if result_path.exists():
            image = json.loads(result_path.read_text())
            if sha256(image['replacement']) != image['replacement_sha256']:
                raise PTXError('Cached replacement changed')
            return result_path
        directory.mkdir(exist_ok=True)
        toolchain = Path(os.environ['UNIFIED_TOOLCHAIN'])
        baseline = directory / 'baseline.cubin'
        result = instrument_instructions(link_group({'module': ptx.read_text()},
                                                    external_functions=('vprintf', '__assertfail')),
                                         max_parameter_bytes=32764, interval_mode=interval_mode,
                                         record_storage='pooled-v1', loop_depth=loop_depth)
        rewritten = directory / 'probed.ptx'
        rewritten.write_text(result.sources['module'])
        replacement = directory / 'probed.cubin'
        commands = []
        for source, output in ((ptx, baseline), (rewritten, replacement)):
            command = [str(toolchain / 'bin/ptxas'), *flags, str(source), '-o', str(output)]
            commands.append(command)
            with output.with_suffix('.log').open('w') as log:
                subprocess.run(command, env=compiler_environment(), stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=180)
        for kind in ('--dump-sass', '--dump-resource-usage'):
            if disassembly(toolchain, original, kind) != disassembly(toolchain, baseline, kind):
                raise PTXError('JIT no-probe reconstruction differs: ' + str(original))
        if sha256(original) != identity['original'] or sha256(ptx) != identity['ptx']:
            raise PTXError('JIT inputs changed during compilation')
        image = image_manifest(original, replacement, result.manifest, 'DeepEP')
        image.update(target=parse(ptx.read_text()).target, build_identity=identity, commands=commands,
                     sass_equal=True, resources_equal=True)
        result_path.write_text(json.dumps(image, indent=2) + '\n')
        return result_path


def runtime_environment(build_path, catalog_path, output, session, capture_policy=None, *, capture_mode='full'):
    """Validate the implementation and prepare an opt-in child environment.

    Observation-only modes execute the caller's original images and argument ABI.
    Their catalogs must already contain every selected JIT image.
    """
    if capture_mode not in ('full', 'hooks', 'hooks-cupti'):
        raise PTXError('Unsupported capture mode')
    canonical_policy = policy_json() if capture_policy is None else policy_json(capture_policy)
    build = json.loads(Path(build_path).read_text())
    if (build.get('status') != 'built' or build.get('implementation') != implementation_identity() or
            sha256(build['library']) != build['sha256']):
        raise PTXError('A current, intact native runtime build is required')
    catalog_path = Path(catalog_path).resolve()
    catalog = json.loads(catalog_path.read_text())
    if catalog.get('schema') != 1 or catalog.get('interval_mode') not in {'instruction-only', 'loop-boundary'}:
        raise PTXError('Unsupported runtime catalog')
    catalog_loop_depth(catalog, 'DeepEP')
    if type(catalog.get('context_abi')) is not int or catalog['context_abi'] != CONTEXT_ABI:
        raise PTXError('Incompatible catalog context ABI')
    for image in catalog['images']:
        if type(image.get('probe_enabled', True)) is not bool:
            raise PTXError('Image probe_enabled must be Boolean')
        if capture_mode == 'full' and image['probe'].get('site_selection') != SITE_SELECTION:
            raise PTXError('Incompatible instruction coverage; rebuild transformed images')
        if (type(image.get('context_abi')) is not int or image['context_abi'] != CONTEXT_ABI or
                type(image['probe'].get('context_abi')) is not int or image['probe']['context_abi'] != CONTEXT_ABI or
                image['probe'].get('sampling') != 'launch_policy'):
            raise PTXError('Incompatible image context ABI')
        depth = catalog_loop_depth(catalog, image['label'])
        if (validate_loop_depth(image['interval_mode'], image.get('loop_depth', 1)) != depth or
                validate_loop_depth(image['interval_mode'], image['probe'].get('loop_depth', 1)) != depth):
            raise PTXError('Catalog/image loop depth mismatch')
        if image['interval_mode'] != catalog['interval_mode']:
            raise PTXError('Catalog/image interval mode mismatch')
        if sha256(image['replacement']) != image['replacement_sha256']:
            raise PTXError('Catalog replacement changed')
    return {'LD_PRELOAD': ':'.join(filter(None, (build['library'], os.environ.get('LD_PRELOAD')))),
            'PYTHONPATH': os.pathsep.join(filter(None, (str(ROOT / 'neutrino'), os.environ.get('PYTHONPATH')))),
            'NEUTRINO_SYS_CATALOG': str(catalog_path), 'NEUTRINO_SYS_OUTPUT': str(Path(output).resolve()),
            'NEUTRINO_SYS_CACHE': str(ROOT / '.cache/unified/sys-jit'),
            'NEUTRINO_SYS_PYTHON': sys.executable, 'NEUTRINO_SESSION_ID': session,
            'NEUTRINO_SYS_CAPTURE_POLICY': canonical_policy, 'NEUTRINO_SYS_CAPTURE_MODE': capture_mode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    native = commands.add_parser('build')
    native.add_argument('--output', type=Path, required=True)
    nccl = commands.add_parser('catalog')
    nccl.add_argument('--build', type=Path, action='append', required=True)
    nccl.add_argument('--output', type=Path, required=True)
    nccl.add_argument('--jit-root', type=Path, action='append', default=[])
    nccl.add_argument('--jit-loop-depth', type=int,
                      help='Independent DeepEP JIT depth; defaults to the precompiled image depth')
    jit = commands.add_parser('jit')
    jit.add_argument('--image', type=Path, required=True)
    jit.add_argument('--cache', type=Path, required=True)
    jit.add_argument('--interval-mode', choices=('instruction-only', 'loop-boundary'), default='instruction-only')
    jit.add_argument('--loop-depth', type=int, default=1)
    args = parser.parse_args()
    if args.command == 'build':
        print(json.dumps(compile_native(args.output), indent=2))
    elif args.command == 'catalog':
        result = prepare_nccl(args.build, args.output, jit_roots=args.jit_root, jit_loop_depth=args.jit_loop_depth)
        print(json.dumps({'images': len(result['images']), 'catalog': str(args.output / 'catalog.json')}))
    else:
        print(prepare_jit(args.image, args.cache, interval_mode=args.interval_mode, loop_depth=args.loop_depth))


if __name__ == '__main__':
    main()
