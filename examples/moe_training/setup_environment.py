"""Build the training overlay from clean pinned sources without changing toolchains."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
PINS = {
    'DeepEP': 'f6aa38fe06cd4a0489e7730365d4e0114970a2f3',
    'DeepEP/third-party/deep_jit': 'cba802df081985141eeee36803685775c183a047',
    'Megatron-LM-unified': 'd7157825675a65c2503a061ffd1fffa7886df9b7',
    'nccl-unified': '12df1a11afad322be5a204a2db890161cbf8131d',
}


def verify_sources():
    result = {}
    for name, expected in PINS.items():
        source = ROOT / 'third_party' / name
        revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        status = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain',
                                          '--untracked-files=no'], text=True).strip()
        if revision != expected or status:
            raise RuntimeError(f'Pinned source changed: {name}: {revision}, {status}')
        result[name] = revision
    return result


def stage_source(source, destination, revision, env, log):
    if destination.exists():
        raise RuntimeError(f'Refusing to reuse a previous build tree: {destination}')
    subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout', str(source), str(destination)],
                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', revision],
                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def configure_nccl_discovery():
    import sysconfig
    overlay = (ROOT / '.venv-unified').resolve()
    if Path(sys.prefix).resolve() != overlay:
        raise RuntimeError('NCCL package discovery can only be configured inside the overlay')
    package = Path(sysconfig.get_path('purelib')) / 'nvidia/nccl'
    target = ROOT / '.local/unified/nccl'
    if not package.parent.resolve().is_relative_to(overlay):
        raise RuntimeError('NVIDIA package directory escapes the overlay')
    package.parent.mkdir(parents=True, exist_ok=True)
    if not (package.is_symlink() and package.resolve() == target.resolve()):
        if package.exists() or package.is_symlink():
            raise RuntimeError(f'Refusing to replace an existing NVIDIA package: {package}')
        package.symlink_to(target, target_is_directory=True)
    runtime = package.parent / 'cuda_runtime/lib/libcudart.so.13'
    runtime_target = (ROOT.parent / 'deepep-monitor/.env/targets/x86_64-linux/lib/libcudart.so.13').resolve()
    if not runtime.parent.resolve().is_relative_to(overlay):
        raise RuntimeError('CUDA runtime discovery escapes the overlay')
    runtime.parent.mkdir(parents=True, exist_ok=True)
    if not (runtime.is_symlink() and runtime.resolve() == runtime_target):
        if runtime.exists() or runtime.is_symlink():
            raise RuntimeError(f'Refusing to replace an existing CUDA runtime: {runtime}')
        runtime.symlink_to(runtime_target)
    return {'nccl': str(package), 'cuda_runtime': str(runtime)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Otherwise only inspect pins and commands')
    parser.add_argument('--attempt', type=int, default=1, help='Fresh build attempt; earlier artifacts are retained')
    parser.add_argument('--configure-nccl', action='store_true', help='Only configure overlay library discovery')
    args = parser.parse_args()
    if args.attempt < 1:
        parser.error('--attempt must be positive')
    sources = verify_sources()
    suffix = '' if args.attempt == 1 else f'-{args.attempt}'
    stage = ROOT / f'.build/unified/training-source{suffix}/DeepEP'
    output = ROOT / f'artifacts/unified-sys/environment{suffix}'
    commands = [
        [sys.executable, '-m', 'pip', 'install', '--index-url', 'https://mirrors.aliyun.com/pypi/simple',
         '--no-deps', '--ignore-installed', 'toml==0.10.2'],
        [sys.executable, '-m', 'pip', 'install', '--no-build-isolation', '--no-deps', '--ignore-installed',
         '--report', str(output / 'pip-deepep.json'), str(stage)],
    ]
    if not args.execute:
        print(json.dumps({'sources': sources, 'commands': commands, 'gpu_execution': False}, indent=2))
        return 0
    if Path(sys.prefix).resolve() != (ROOT / '.venv-unified').resolve():
        raise RuntimeError('Source env.sh and use the isolated Python')
    build = json.loads((ROOT / 'artifacts/unified-sys/nccl-build/build.json').read_text())
    library = Path(build['library'])
    if (build['status'] != 'built' or
            hashlib.sha256(library.read_bytes()).hexdigest() != build['sha256']):
        raise RuntimeError('Verified isolated NCCL build is required')
    discovery = configure_nccl_discovery()
    if args.configure_nccl:
        print(json.dumps({'nccl_discovery': discovery, 'nccl_sha256': build['sha256']}, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.pop('LD_PRELOAD', None)
    env.update(CUDA_VISIBLE_DEVICES='', MAX_JOBS='4', PIP_REQUIRE_VIRTUALENV='true',
               PIP_DISABLE_PIP_VERSION_CHECK='1', PYTHONDONTWRITEBYTECODE='1')
    manifest = {'schema': 1, 'status': 'building', 'sources': sources,
                'nccl': build['sha256'], 'nccl_discovery': discovery,
                'commands': commands, 'gpu_execution': False}
    status = 1
    try:
        with (output / 'build.log').open('w') as log:
            stage_source(ROOT / 'third_party/DeepEP', stage, PINS['DeepEP'], env, log)
            nested = stage / 'third-party/deep_jit'
            # A gitlink checkout creates an empty directory, accepted by git clone.
            subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout',
                            str(ROOT / 'third_party/DeepEP/third-party/deep_jit'), str(nested)],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run(['git', '-C', str(nested), 'checkout', '--detach',
                            PINS['DeepEP/third-party/deep_jit']], env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
            for command in commands:
                subprocess.run(command, env=env, cwd=stage, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        manifest['source_integrity_after'] = verify_sources()
        manifest['status'] = 'installed'
        status = 0
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        (output / 'environment.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
