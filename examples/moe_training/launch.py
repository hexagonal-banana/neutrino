"""Dry-run by default; GPU execution requires confirmed allocation and live idle checks."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

from config import Workload
from gpu_guard import UnsafeGPUState, check_once, launch_lock, require_idle
from setup_environment import ROOT, verify_sources
from preflight import digest, harness_files, nccl_header
from dataset_helper import evidence as dataset_evidence


def command_for(workload):
    return [str(ROOT / '.venv-unified/bin/python'), '-m', 'torch.distributed.run', '--standalone',
            '--nnodes=1', f'--nproc-per-node={workload.world_size}', '--max-restarts=0',
            str(Path(__file__).with_name('worker.py'))]


def require_clean_baseline(env):
    allowed = {'NEUTRINO_BUILD_DIR', 'NEUTRINO_REAL_DRIVER', 'NEUTRINO_REAL_NVCC',
               'NEUTRINO_COMPILER_RECORDS'}
    active = sorted(key for key, value in env.items()
                    if key.startswith('NEUTRINO_') and key not in allowed and value)
    pinned = (ROOT / '.local/unified/nccl/lib/libnccl.so.2').resolve()
    preloads = env.get('LD_PRELOAD', '').replace(':', ' ').split()
    if (active or env.get('LD_AUDIT') or
            any(Path(path).resolve() != pinned for path in preloads)):
        raise RuntimeError(f'Baseline must not inherit a tracer or external preload: {active}')


def validate_evidence(preflight, reconstruction, directory, sources, workload):
    if (preflight.get('status') != 'passed' or preflight.get('sources') != sources or
            preflight.get('harness_files') != harness_files() or
            not any(row.get('workload') == workload.manifest()
                    for row in preflight.get('config_checks', []))):
        raise RuntimeError('A current passing import/config/library preflight is required')
    helper = dataset_evidence()
    if preflight.get('dataset_helper') != helper:
        raise RuntimeError('Dataset helper changed since preflight')
    libraries = preflight['libraries']
    if libraries['nccl_header'] != nccl_header():
        raise RuntimeError('NCCL header changed since preflight')
    for item in libraries.values():
        if digest(item['path']) != item['sha256']:
            raise RuntimeError(f'Library/header changed since preflight: {item["path"]}')
    extension = Path(preflight['deep_ep_extension']).resolve()
    if (not extension.is_relative_to((ROOT / '.venv-unified').resolve()) or
            digest(extension) != preflight['deep_ep_extension_sha256']):
        raise RuntimeError('DeepEP extension changed since preflight')
    build = json.loads((ROOT / 'artifacts/unified-sys/nccl-build/build.json').read_text())
    if (build['status'] != 'built' or build['source_revision'] != sources['nccl-unified'] or
            libraries['nccl']['sha256'] != build['sha256'] or
            Path(libraries['nccl']['path']).resolve() != Path(build['library']).resolve()):
        raise RuntimeError('Preflight does not match the current isolated NCCL build')
    if (reconstruction.get('status') != 'passed' or reconstruction.get('target') != 'sm_90' or
            not reconstruction.get('assemblies') or not reconstruction.get('links')):
        raise RuntimeError('The no-probe linked-image reconstruction gate must pass')
    records = (ROOT / 'artifacts/unified-sys/nccl-build/compiler').resolve()
    for item in [*reconstruction['assemblies'], *reconstruction['links']]:
        original = Path(item['original']).resolve()
        rebuilt = (directory / item['rebuilt']).resolve()
        if (not original.is_relative_to(records) or not rebuilt.is_relative_to(directory.resolve()) or
                item.get('sass_equal') is not True or item.get('resources_equal') is not True or
                digest(original) != item['original_sha256'] or digest(rebuilt) != item['rebuilt_sha256']):
            raise RuntimeError('Reconstruction artifacts changed or do not match this NCCL build')
        if 'ptx' in item and digest(item['ptx']) != item['ptx_sha256']:
            raise RuntimeError('Recorded PTX changed since reconstruction')
    return {'libraries': libraries, 'deep_ep_extension_sha256': preflight['deep_ep_extension_sha256'],
            'dataset_helper': helper, 'harness_files': preflight['harness_files']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world-size', choices=(2, 8), type=int, default=8)
    parser.add_argument('--tensor-parallel-size', choices=(1, 2), type=int, default=1)
    parser.add_argument('--expert-parallel-size', choices=(2, 4, 8), type=int)
    parser.add_argument('--workload-config', type=Path, help='JSON Workload fields; cannot combine with topology overrides')
    parser.add_argument('--profile-kernels', action='store_true', help='CUPTI/PyTorch activity control without PTX instrumentation')
    parser.add_argument('--mode', choices=('fixture', 'train'), default='train')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--capture-mode', choices=('full', 'hooks', 'hooks-cupti'), default='full',
                        help='Native ablation: full probes, original kernels with hooks, or hooks plus CUPTI')
    parser.add_argument('--runtime-build', type=Path)
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--capture-policy', type=Path, help='Kernel-name GPU sampling policy JSON; default warp')
    parser.add_argument('--trace-capacity', type=int, default=256)
    parser.add_argument('--trace-budget-gib', type=int, default=2)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--execute', action='store_true')
    action.add_argument('--check-evidence', action='store_true', help='Validate prerequisites without querying GPUs')
    parser.add_argument('--allocation-confirmed', action='store_true')
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--reconstruction', type=Path)
    parser.add_argument('--fixture-result', type=Path)
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'neutrino'))
    from neutrino.sys_sync.sampling import validate_policy, policy_hash
    if args.capture_policy and not args.trace:
        parser.error('--capture-policy requires --trace')
    if args.capture_mode != 'full' and not args.trace:
        parser.error('--capture-mode requires --trace')
    try:
        capture_policy = validate_policy(json.loads(args.capture_policy.read_text())) if args.capture_policy else validate_policy()
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.trace and (not args.runtime_build or not args.catalog):
        parser.error('--trace requires --runtime-build and --catalog')
    if not 0 < args.trace_capacity <= 1048576 or not 1 <= args.trace_budget_gib <= 64:
        parser.error('Invalid trace capacity or budget')
    if args.profile_kernels and (args.trace or args.mode != 'train'):
        parser.error('--profile-kernels requires uninstrumented training')
    if args.workload_config:
        if args.world_size != 8 or args.tensor_parallel_size != 1 or args.expert_parallel_size is not None:
            parser.error('--workload-config cannot be combined with topology overrides')
        workload = Workload.from_file(args.workload_config)
        args.world_size = workload.world_size
    else:
        workload = Workload(world_size=args.world_size,
                            expert_parallel_size=args.expert_parallel_size or args.world_size,
                            tensor_parallel_size=args.tensor_parallel_size,
                            global_batch_size=args.world_size // args.tensor_parallel_size).validate()
    command = command_for(workload)
    if not args.execute and not args.check_evidence:
        print(json.dumps({'mode': args.mode, 'trace': args.trace, 'workload': workload.manifest(),
                          'command': command, 'gpu_execution': False}, indent=2))
        print(shlex.join(command))
        return 0
    if args.execute and not args.allocation_confirmed:
        parser.error('--execute requires explicit --allocation-confirmed; idle is not allocation')
    if not args.preflight or not args.reconstruction or (args.execute and not args.run_dir):
        parser.error('Requires --preflight and --reconstruction; execution also requires --run-dir')
    run_dir = args.run_dir.resolve() if args.run_dir else None
    if run_dir is not None and not run_dir.is_relative_to(ROOT / 'artifacts/unified-sys'):
        parser.error('--run-dir must be inside artifacts/unified-sys')
    if Path(sys.prefix).resolve() != (ROOT / '.venv-unified').resolve():
        parser.error('Source env.sh and use the isolated Python')
    if args.mode == 'train' and not args.fixture_result:
        parser.error('Training requires --fixture-result from the matching uninstrumented GPU fixture')
    created_run_dir = False
    try:
        require_clean_baseline(os.environ)
        sources = verify_sources()
        preflight = json.loads(args.preflight.read_text())
        reconstruction = json.loads(args.reconstruction.read_text())
        identity = validate_evidence(preflight, reconstruction, args.reconstruction.resolve().parent,
                                     sources, workload)
        if args.fixture_result:
            fixture = json.loads(args.fixture_result.read_text())
            if (fixture.get('status') != 'passed' or fixture.get('mode') != 'fixture' or
                    fixture.get('sources') != sources or fixture.get('world_size') != args.world_size or
                    fixture.get('workload') != workload.manifest() or
                    fixture.get('environment_identity') != identity or fixture.get('trace') is not False or
                    fixture.get('rank_status') != ['passed'] * args.world_size):
                raise RuntimeError('Fixture result does not match the workload environment')
        session = uuid.uuid4().hex
        trace_env = {}
        if args.trace:
            sys.path.insert(0, str(ROOT / 'neutrino'))
            from neutrino.sys_sync.runtime import runtime_environment
            trace_env = runtime_environment(args.runtime_build, args.catalog,
                                            (run_dir or ROOT / 'artifacts/unified-sys/dry-run') / 'raw', session,
                                            capture_policy=capture_policy, capture_mode=args.capture_mode)
            trace_env.update(NEUTRINO_SYS_CAPACITY=str(args.trace_capacity),
                             NEUTRINO_SYS_BUDGET=str(args.trace_budget_gib * 1024**3))
        if args.check_evidence:
            print(json.dumps({'status': 'passed', 'mode': args.mode, 'world_size': args.world_size,
                              'environment_identity': identity, 'gpu_execution': False}, indent=2))
            return 0
        with launch_lock() as lock:
            devices = require_idle()
            run_dir.mkdir(parents=True, exist_ok=False)
            created_run_dir = True
            selected = devices[:args.world_size]
            env = {**os.environ, **trace_env}
            if workload.tensor_parallel_size > 1:
                env['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
            env.update(CUDA_VISIBLE_DEVICES=','.join(d['uuid'] for d in selected),
                       UNIFIED_GUARDED_LAUNCH='1', UNIFIED_RUN_DIR=str(run_dir),
                       NEUTRINO_SESSION_ID=session, EP_DISABLE_GIN='1', NCCL_LSA_TEAM_SIZE='0')
            payload = {'schema': 1, 'mode': args.mode, 'trace': args.trace,
                       'profile_kernels': args.profile_kernels,
                       'workload': asdict(workload), 'sources': sources, 'devices': selected,
                       'command': command, 'session': session, 'environment_identity': identity,
                       'native_runtime': ({'build': str(args.runtime_build.resolve()),
                                           'catalog': str(args.catalog.resolve()),
                                           'catalog_sha256': digest(args.catalog),
                                           'capacity': args.trace_capacity, 'budget_gib': args.trace_budget_gib,
                                           'capture_mode': args.capture_mode,
                                           'capture_policy': capture_policy,
                                           'capture_policy_sha256': policy_hash(capture_policy)}
                                          if args.trace else None),
                       'preflight': str(args.preflight.resolve()),
                       'reconstruction': str(args.reconstruction.resolve()),
                       'transport': {key: value for key, value in env.items()
                                     if key.startswith(('NCCL_', 'EP_')) or key == 'CUDA_DEVICE_MAX_CONNECTIONS'},
                       'status': 'running'}
            (run_dir / 'launch.json').write_text(json.dumps(payload, indent=2) + '\n')
            final = check_once()
            if [(d['index'], d['uuid']) for d in final] != [(d['index'], d['uuid']) for d in devices]:
                raise UnsafeGPUState('GPU identities changed before launching')
            with (run_dir / 'console.log').open('w') as log:
                process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log,
                                           stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
                code = process.wait()
            ranks = []
            for rank in range(args.world_size):
                path = run_dir / f'rank-{rank}.json'
                ranks.append(json.loads(path.read_text()) if path.exists() else {'status': 'missing'})
            successful = code == 0 and all(
                row.get('status') == 'passed' and row.get('rank') == rank and
                row.get('session') == session and row.get('sources') == sources and
                row.get('libraries') == identity['libraries'] and
                row.get('dataset_helper') == identity['dataset_helper'] and
                row.get('gpu_uuid') == selected[rank]['uuid'] for rank, row in enumerate(ranks))
            validation = {'schema': 1, 'status': 'passed' if successful else 'failed', 'exit_code': code,
                          'mode': args.mode, 'trace': args.trace, 'world_size': args.world_size,
                          'workload': workload.manifest(),
                          'sources': sources, 'environment_identity': identity,
                          'nccl_sha256': preflight['libraries']['nccl']['sha256'],
                          'rank_status': [row.get('status', 'missing') for row in ranks],
                          'capture_mode': args.capture_mode if args.trace else None,
                          'trace_acceptance_validated': False}
            (run_dir / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
            print(json.dumps(validation, indent=2))
            return 0 if successful else 1
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
        if created_run_dir:
            failure = {'schema': 1, 'status': 'failed', 'mode': args.mode, 'trace': args.trace,
                       'world_size': args.world_size, 'error': f'{type(exc).__name__}: {exc}',
                       'trace_acceptance_validated': False}
            (run_dir / 'validation.json').write_text(json.dumps(failure, indent=2) + '\n')
        print(f'LAUNCH REFUSED/FAILED: {exc}', file=sys.stderr)
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
