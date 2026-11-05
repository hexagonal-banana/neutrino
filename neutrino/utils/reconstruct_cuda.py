"""CPU-only reconstruction gate for NVCC-recorded linked CUDA artifacts.

Only replay ptxas and nvlink, never shell commands or arbitrary compiler-log text.
Rebuild all retained PTX, then relink using reconstructed relocatable inputs.
This proves binary feasibility, not runtime tracing or collective correctness.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


class ReconstructionError(RuntimeError):
    pass


def expand_options(argv, record_dir, cwd, seen=()):
    """Expand NVCC response files as argument data, never executable shell text."""
    result = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in ('-optf', '--options-file'):
            index += 1
            if index == len(argv):
                raise ReconstructionError('Missing compiler options file')
            value = argv[index]
        elif arg.startswith(('-optf=', '--options-file=')):
            value = arg.split('=', 1)[1]
        else:
            result.append(arg)
            index += 1
            continue
        if record_dir is None:
            raise ReconstructionError('Options files require a compiler record directory')
        for name in value.split(','):
            path = (Path(cwd) / name).resolve()
            if not path.is_relative_to(Path(record_dir).resolve()):
                raise ReconstructionError('Options file escapes the compiler record directory')
            if path in seen or len(seen) >= 8:
                raise ReconstructionError('Recursive compiler options file')
            result.extend(expand_options(shlex.split(path.read_text()), record_dir, cwd, (*seen, path)))
        index += 1
    return result


def compiler_commands(text, record_dir=None, cwd=None):
    commands = []
    for line in text.splitlines():
        if not line.startswith('#$ '):
            continue
        line = line[3:]
        if not re.match(r'(?:\S*/)?(?:ptxas|nvlink)\s', line):
            continue
        argv = expand_options(shlex.split(line), record_dir, cwd or record_dir or Path.cwd())
        if any('$' in arg or '`' in arg or arg in (';', '&&', '||', '|', '>', '<', '&')
               for arg in argv):
            raise ReconstructionError('Unsupported shell expansion in compiler command')
        if '-o' not in argv or argv.count('-o') != 1:
            raise ReconstructionError('Expected one explicit compiler output')
        commands.append(argv)
    return commands


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def architecture(command):
    for arg in command:
        if arg.startswith(('-arch=', '--arch=', '--gpu-name=')):
            return arg.split('=', 1)[1]
    raise ReconstructionError('Compiler command has no explicit architecture')


def disassembly(toolchain, path, kind):
    env = os.environ.copy()
    env.pop('LD_PRELOAD', None)
    output = subprocess.check_output([str(toolchain / 'bin/cuobjdump'), kind, str(path)],
                                     text=True, env=env)
    # Strip only container filename metadata, never code, encodings, or resources.
    return '\n'.join(line for line in output.splitlines()
                     if not re.match(r'^\s*(?:File|file|identifier)\s*:', line))


def verify_host_object(toolchain, path, env):
    """Accept only host ELF objects with independently confirmed absent device code."""
    sections = subprocess.check_output(['readelf', '--wide', '--file-header', '--sections', str(path)],
                                       text=True, env=env)
    if ('.nv' in sections or '.cuda' in sections or
            not re.search(r'Type:\s+REL\b', sections) or
            not re.search(r'Machine:\s+(?:Advanced Micro Devices X86-64|AArch64)\s*$', sections, re.M)):
        raise ReconstructionError(f'Unrecorded device-link input: {path}')
    expected = rf"cuobjdump info\s*: File '{re.escape(str(path))}' does not contain device code"
    for kind in ('--list-elf', '--list-ptx'):
        result = subprocess.run([str(toolchain / 'bin/cuobjdump'), kind, str(path)],
                                text=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        diagnostic = (result.stdout + result.stderr).strip()
        if not ((result.returncode == 0 and not diagnostic) or
                (result.returncode == 255 and re.fullmatch(expected, diagnostic))):
            raise ReconstructionError(f'Cannot verify host-only input {path}: {diagnostic}')
    return {'path': str(path), 'sha256': sha256(path)}


def reconstruct(records, destination, target='sm_90'):
    records, destination = Path(records).resolve(), Path(destination).resolve()
    if destination == records or destination.is_relative_to(records):
        raise ReconstructionError('Reconstruction must not overwrite recorded inputs')
    destination.mkdir(parents=True, exist_ok=False)
    report = {'schema': 1, 'status': 'running', 'target': target, 'assemblies': [],
              'links': [], 'other_architectures': [], 'host_only_inputs': [],
              'runtime_verified': False}
    env = os.environ.copy()
    env.pop('LD_PRELOAD', None)
    objects = {}
    links = []
    manifest = destination / 'reconstruction.json'

    def run(command, info, baseline, output, label):
        toolchain = Path(info['compiler']).resolve().parent.parent
        command[0] = str(toolchain / 'bin' / Path(command[0]).name)
        command[command.index('-o') + 1] = str(output)
        with (output.parent / f'{label}.log').open('w') as log:
            result = subprocess.run(command, cwd=info['cwd'], env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise ReconstructionError(f'{label} failed; see {output.parent / (label + ".log")}')
        original_sass = disassembly(toolchain, baseline, '--dump-sass')
        rebuilt_sass = disassembly(toolchain, output, '--dump-sass')
        original_resources = disassembly(toolchain, baseline, '--dump-resource-usage')
        rebuilt_resources = disassembly(toolchain, output, '--dump-resource-usage')
        sass_equal = original_sass == rebuilt_sass
        resources_equal = original_resources == rebuilt_resources
        item = {'original': str(baseline), 'rebuilt': str(output.relative_to(destination)),
                'original_sha256': sha256(baseline), 'rebuilt_sha256': sha256(output),
                'sass_equal': sass_equal, 'resources_equal': resources_equal, 'command': command}
        if not sass_equal or not resources_equal:
            report['mismatch'] = item
            for name, text in (('original.sass', original_sass), ('rebuilt.sass', rebuilt_sass),
                               ('original.resources', original_resources),
                               ('rebuilt.resources', rebuilt_resources)):
                (output.parent / name).write_text(text)
            raise ReconstructionError(f'No-probe SASS/resource mismatch for {baseline}')
        return item

    try:
        record_paths = sorted(records.glob('*/command.json'))
        if not record_paths:
            raise ReconstructionError('No compiler records')
        for record_path in record_paths:
            info = json.loads(record_path.read_text())
            if info['status'] != 'passed':
                raise ReconstructionError(f'Incomplete compiler record: {record_path}')
            try:
                commands = compiler_commands((record_path.parent / 'compiler.log').read_text(),
                                             record_path.parent, info['cwd'])
            except Exception as exc:
                raise ReconstructionError(f'{record_path}: {exc}') from exc
            per_object = {}
            objects[str((Path(info['cwd']) / info['output']).resolve())] = per_object
            for index, command in enumerate(commands):
                arch = architecture(command)
                if arch != target:
                    report['other_architectures'].append({'record': str(record_path), 'target': arch})
                    continue
                baseline = (Path(info['cwd']) / command[command.index('-o') + 1]).resolve()
                work = destination / record_path.parent.name
                work.mkdir(exist_ok=True)
                output = work / baseline.name
                if Path(command[0]).name == 'nvlink':
                    links.append((command, info, baseline, output))
                    continue
                sources = [(Path(info['cwd']) / arg).resolve()
                           for arg in command[1:] if arg.endswith('.ptx')]
                if len(sources) != 1 or not sources[0].is_file():
                    raise ReconstructionError(f'Missing or ambiguous PTX source: {record_path}')
                item = run(command.copy(), info, baseline, output, f'ptxas-{index}')
                item.update(ptx=str(sources[0]), ptx_sha256=sha256(sources[0]),
                            relocatable='--compile-only' in command or '-c' in command,
                            system_scope_opcode_candidates=len(re.findall(
                                r'(?m)^\s*(?:@!?[%\w]+\s+)?(?:ld|st|atom|red|fence|membar)\.[^;\s]*\bsys\b',
                                sources[0].read_text())))
                report['assemblies'].append(item)
                if arch in per_object:
                    raise ReconstructionError(f'Multiple images for one object/architecture: {record_path}')
                per_object[arch] = output
        for command, info, baseline, output in links:
            rewritten = []
            replaced = 0
            for arg in command:
                if arg.endswith('.o'):
                    key = str((Path(info['cwd']) / arg).resolve())
                    if key not in objects:
                        toolchain = Path(info['compiler']).resolve().parent.parent
                        report['host_only_inputs'].append(verify_host_object(toolchain, key, env))
                    elif target in objects[key]:
                        arg = str(objects[key][target])
                        replaced += 1
                elif arg.startswith('--register-link-binaries='):
                    arg = '--register-link-binaries=' + str(output.with_suffix('.reg.c'))
                rewritten.append(arg)
            if not replaced:
                raise ReconstructionError(f'No reconstructed link inputs for {baseline}')
            item = run(rewritten, info, baseline, output, 'nvlink')
            item['reconstructed_inputs'] = replaced
            report['links'].append(item)
        if not report['assemblies'] or not report['links']:
            raise ReconstructionError('Both PTX assembly and device-link reconstruction are required')
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        manifest.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--target', default='sm_90')
    args = parser.parse_args()
    report = reconstruct(args.records, args.output, args.target)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('assemblies', 'links', 'other_architectures')}, indent=2))
    print(f'Assemblies: {len(report["assemblies"])}; linked images: {len(report["links"])}')
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    sys.exit(main())
