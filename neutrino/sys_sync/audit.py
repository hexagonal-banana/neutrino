"""CPU-only static PTX inventory; this does not establish launched-kernel coverage."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .ptx import PTXError, parse
from .abi import parameter_layout
from .control_flow import analyze


def inventory(paths, target, *, check_abi=False, check_control_flow=False):
    rows = []
    for path in sorted(set(Path(path).resolve() for path in paths)):
        row = {'path': str(path)}
        try:
            data = path.read_bytes()
            row['sha256'] = hashlib.sha256(data).hexdigest()
            module = parse(data.decode('utf-8'))
            row['target'] = module.target
            if module.target != target:
                row['status'] = 'other_architecture'
            else:
                row.update(status='analyzed', functions=[], sites=[asdict(site) for site in module.sites])
                for function in module.functions:
                    info = {'name': function.name, 'kind': function.kind, 'defined': function.defined,
                            'instructions': len(function.instructions)}
                    try:
                        if check_abi:
                            info['parameter_layout'] = asdict(parameter_layout(data.decode('utf-8'), function))
                        if check_control_flow and function.defined:
                            cfg = analyze(function)
                            info['control_flow'] = {'blocks': len(cfg.blocks), 'edges': len(cfg.edges),
                                                    'reachable_blocks': len(cfg.reachable),
                                                    'natural_loops': len(cfg.loops)}
                    except PTXError as exc:
                        info['analysis_error'] = str(exc)
                        row['status'] = 'failed'
                    row['functions'].append(info)
        except (OSError, UnicodeError, PTXError) as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    analyzed = [row for row in rows if row['status'] == 'analyzed']
    return {'schema': 1, 'status': 'passed' if analyzed and all(
                row['status'] != 'failed' for row in rows) else 'failed',
            'kind': 'static_ptx_inventory', 'target': target,
            'check_abi': check_abi, 'check_control_flow': check_control_flow,
            'runtime_verified': False, 'polling_intervals_validated': False,
            'analyzed_modules': len(analyzed), 'sites': sum(len(row['sites']) for row in analyzed),
            'modules': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ptx', type=Path, action='append', default=[])
    parser.add_argument('--ptx-directory', type=Path, action='append', default=[])
    parser.add_argument('--target', default='sm_90')
    parser.add_argument('--check-abi', action='store_true')
    parser.add_argument('--check-control-flow', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.ptx and not args.ptx_directory:
        parser.error('Provide --ptx or --ptx-directory')
    for directory in args.ptx_directory:
        if not directory.is_dir():
            parser.error(f'Not a directory: {directory}')
    paths = [*args.ptx, *(path for directory in args.ptx_directory for path in directory.rglob('*.ptx'))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        report = inventory(paths, args.target, check_abi=args.check_abi,
                           check_control_flow=args.check_control_flow)
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'modules'}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
