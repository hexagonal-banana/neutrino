"""CPU-only transformation gate for one reconstructed device-link group.

The input is a successful no-probe reconstruction manifest. Outputs are never
loaded into an application by this command. Timestamp probes require explicit
instruction-only or loop-boundary mode; assembly success is not runtime acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import struct

from .context import propagate_context
from .linkage import link_group
from .probe import instrument_instructions, validate_loop_depth
from .ptx import PTXError
from ..utils.reconstruct_cuda import architecture, sha256


def _child(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root) or path == root:
        raise PTXError('Reconstructed artifact escapes its manifest directory')
    return path


def _verify(item, root):
    if not item.get('sass_equal') or not item.get('resources_equal'):
        raise PTXError('No-probe reconstruction equivalence is required')
    for path, expected in ((Path(item['original']), item['original_sha256']),
                           (_child(root, item['rebuilt']), item['rebuilt_sha256'])):
        if sha256(path) != expected:
            raise PTXError(f'Reconstruction input identity changed: {path}')


def _other_architecture(path, link, target, recorded):
    if path not in recorded:
        raise PTXError(f'Unrecorded device link input: {path}')
    env = os.environ.copy()
    env.pop('LD_PRELOAD', None)
    env['CUDA_VISIBLE_DEVICES'] = ''
    tool = Path(link['command'][0]).with_name('cuobjdump')
    result = subprocess.run([str(tool), '-all', '--list-elf', '--list-ptx', path],
                            env=env, capture_output=True, text=True, timeout=60)
    text = (result.stdout + result.stderr).strip()
    lines = text.splitlines()
    images = [re.fullmatch(r'ELF file\s+\d+: .+\.sm_(\d+)[af]?\.cubin', line) for line in lines]
    architectures = [int(match[1]) for match in images if match]
    no_ptx = f"cuobjdump info    : No PTX file found to extract from '{path}'. You may try with -all option."
    if (result.returncode or not architectures or no_ptx not in lines or
            any(match is None and line != no_ptx for line, match in zip(lines, images)) or
            any(arch <= int(re.search(r'\d+', target)[0]) for arch in architectures)):
        raise PTXError(f'Cannot exclude target-compatible device code in {path}')
    return {'path': path, 'sha256': sha256(path), 'higher_architectures': architectures}


def load_reconstructed_group(manifest, link_index):
    manifest = Path(manifest).resolve()
    root = manifest.parent
    report = json.loads(manifest.read_text())
    if report.get('status') != 'passed' or report.get('schema') != 1:
        raise PTXError('A successful reconstruction manifest is required')
    if type(link_index) is not int or not 0 <= link_index < len(report['links']):
        raise PTXError('Invalid reconstructed link index')
    link = report['links'][link_index]
    _verify(link, root)
    all_items = {str(_child(root, item['rebuilt'])): item for item in report['assemblies']}
    if len(all_items) != len(report['assemblies']):
        raise PTXError('Ambiguous reconstructed assembly identity')
    items = [all_items[arg] for arg in link['command'][1:] if arg in all_items]
    if not items or len(items) != link['reconstructed_inputs']:
        raise PTXError('Incomplete reconstructed device-link group')
    if len({item['rebuilt'] for item in items}) != len(items):
        raise PTXError('Duplicate device-link input')
    sources = {}
    for item in items:
        _verify(item, root)
        header = _child(root, item['rebuilt']).read_bytes()[:20]
        if len(header) != 20 or header[:6] != b'\x7fELF\x02\x01':
            raise PTXError('Expected little-endian ELF64 device input')
        kind, machine = struct.unpack_from('<HH', header, 16)
        expected = 1 if item.get('relocatable') else 2
        if kind != expected or machine != 190:
            raise PTXError('Device ELF type disagrees with recorded assembly mode')
        data = Path(item['ptx']).read_bytes()
        if hashlib.sha256(data).hexdigest() != item['ptx_sha256']:
            raise PTXError(f'PTX source identity changed: {item["ptx"]}')
        sources[item['rebuilt']] = data.decode('utf-8')
    host = {item['path']: item['sha256'] for item in report['host_only_inputs']}
    other_outputs = set()
    for other in report.get('other_architectures', []):
        record = json.loads(Path(other['record']).read_text())
        if record.get('status') != 'passed':
            raise PTXError('Incomplete other-architecture compiler record')
        other_outputs.add(str((Path(record['cwd']) / record['output']).resolve()))
    ignored = []
    for arg in link['command'][1:]:
        if arg.endswith('.o') and arg not in all_items:
            if arg in host:
                if sha256(arg) != host[arg]:
                    raise PTXError(f'Host link input identity changed: {arg}')
            else:
                ignored.append(_other_architecture(arg, link, report['target'], other_outputs))
    return root, report['target'], dict(link, other_architecture_inputs=ignored), items, sources


def build_context(manifest, destination, *, link_index=0, max_parameter_bytes,
                  external_functions=(), assemble=False, interval_mode=None, loop_depth=1):
    if interval_mode not in (None, 'instruction-only', 'loop-boundary'):
        raise PTXError('Expected instruction-only or loop-boundary probes')
    validate_loop_depth(interval_mode, loop_depth)
    manifest, destination = Path(manifest).resolve(), Path(destination).resolve()
    if destination == manifest.parent or destination.is_relative_to(manifest.parent):
        raise PTXError('Output must be separate from reconstruction evidence')
    destination.mkdir(parents=True, exist_ok=False)
    report = {'schema': 1, 'status': 'running',
              'kind': 'linked_instruction_probe_gate' if interval_mode else 'linked_context_abi_gate',
              'runtime_verified': False, 'site_instrumentation': interval_mode is not None,
              'interval_mode': interval_mode, 'loop_depth': loop_depth, 'polling_intervals_validated': False,
              'sass_timer_placement_validated': False, 'cooperative_resources_validated': False,
              'assembly_requested': assemble, 'assembly_verified': False,
              'reconstruction': str(manifest), 'assemblies': []}
    implementation = {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob('*.py'))}
    report['implementation_sha256'] = implementation
    try:
        root, target, link, items, sources = load_reconstructed_group(manifest, link_index)
        report['reconstruction_sha256'] = sha256(manifest)
        report['target'] = target
        report['other_architecture_inputs'] = link['other_architecture_inputs']
        # NVCC may list whole-program EXEC images alongside REL inputs in a
        # device-link command. They remain separate loadable images, not members
        # of the linked symbol namespace. Preserve that ownership distinction.
        relocatable = {item['rebuilt']: sources[item['rebuilt']] for item in items if item['relocatable']}
        if not relocatable:
            raise PTXError('No relocatable members in reconstructed link group')
        groups = [('linked', relocatable), *((item['rebuilt'], {item['rebuilt']: sources[item['rebuilt']]})
                                             for item in items if not item['relocatable'])]
        rewritten_sources = {}
        report.update(contexts=[], defined_functions=0, calls=0, system_sites=0)
        for owner, inputs in groups:
            group = link_group(inputs, external_functions=external_functions)
            rewritten = (instrument_instructions(group, max_parameter_bytes=max_parameter_bytes,
                                                  interval_mode=interval_mode, loop_depth=loop_depth) if interval_mode else
                         propagate_context(group, max_parameter_bytes=max_parameter_bytes))
            rewritten_sources.update(rewritten.sources)
            report['contexts'].append({'owner': owner, **rewritten.manifest})
            report['defined_functions'] += len(group.definitions)
            report['calls'] += len(group.calls)
            report['system_sites'] += sum(len(module.sites) for module in group.modules.values())
        env = os.environ.copy()
        env.pop('LD_PRELOAD', None)
        env['CUDA_VISIBLE_DEVICES'] = ''
        replacements = {}

        def run(command, output, tool):
            if (Path(command[0]).name != tool or not Path(command[0]).is_absolute() or
                    command.count('-o') != 1 or architecture(command) != target):
                raise PTXError(f'Invalid recorded {tool} command')
            command = command.copy()
            command[command.index('-o') + 1] = str(output)
            command = ['--register-link-binaries=' + str(output.with_suffix('.reg.c'))
                       if arg.startswith('--register-link-binaries=') else arg for arg in command]
            log = output.with_suffix('.log')
            with log.open('x') as handle:
                result = subprocess.run(command, cwd=destination, env=env, stdout=handle,
                                        stderr=subprocess.STDOUT, timeout=600)
            if result.returncode:
                raise PTXError(f'{tool} failed ({result.returncode}): {log}')
            return command

        for item in items:
            unit = item['rebuilt']
            output = _child(destination, unit)
            output.parent.mkdir(parents=True, exist_ok=True)
            ptx = output.with_suffix('.ptx')
            ptx.write_text(rewritten_sources[unit])
            command = item['command'].copy()
            ptx_arguments = [index for index, arg in enumerate(command) if arg.endswith('.ptx')]
            if len(ptx_arguments) != 1 or command[ptx_arguments[0]] != item['ptx']:
                raise PTXError('Ambiguous recorded assembler source')
            command[ptx_arguments[0]] = str(ptx)
            row = {'unit': unit, 'ptx': str(ptx.relative_to(destination)), 'ptx_sha256': sha256(ptx)}
            if assemble:
                row['command'] = run(command, output, 'ptxas')
                row['cubin_sha256'] = sha256(output)
            report['assemblies'].append(row)
            replacements[str(_child(root, unit))] = str(output)
        if assemble:
            output = _child(destination, link['rebuilt'])
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [replacements.get(arg, arg) for arg in link['command']]
            report['link_command'] = run(command, output, 'nvlink')
            report['linked_image'] = str(output.relative_to(destination))
            report['linked_sha256'] = sha256(output)
            report['assembly_verified'] = True
        # Recheck pinned inputs after compilation; never bless changed source.
        _, _, checked_link, _, _ = load_reconstructed_group(manifest, link_index)
        if (sha256(manifest) != report['reconstruction_sha256'] or
                checked_link['other_architecture_inputs'] != report['other_architecture_inputs'] or
                {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob('*.py'))} != implementation):
            raise PTXError('Build evidence or implementation changed during compilation')
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        (destination / 'context-build.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reconstruction', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--link-index', type=int, default=0)
    parser.add_argument('--max-parameter-bytes', type=int, required=True)
    parser.add_argument('--external-function', action='append', default=[])
    parser.add_argument('--assemble', action='store_true')
    parser.add_argument('--interval-mode', choices=['instruction-only', 'loop-boundary'],
                        help='Explicit diagnostic probes; omitted means context ABI only')
    parser.add_argument('--loop-depth', type=int, default=1,
                        help='Enclosing loop rank from innermost (1); clamp to outermost available')
    args = parser.parse_args()
    report = build_context(args.reconstruction, args.output, link_index=args.link_index,
                           max_parameter_bytes=args.max_parameter_bytes,
                           external_functions=args.external_function, assemble=args.assemble,
                           interval_mode=args.interval_mode, loop_depth=args.loop_depth)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {'contexts', 'assemblies', 'link_command', 'other_architecture_inputs'}}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
