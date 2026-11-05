"""Build pinned NCCL with retained PTX and full compiler/device-link records.

CPU-only. Run after sourcing env.sh. No dependency source files are patched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

NCCL_REVISION = '12df1a11afad322be5a204a2db890161cbf8131d'


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    root = Path(__file__).resolve().parents[3]
    source = root / 'third_party/nccl-unified'
    build = root / '.build/unified/nccl'
    prefix = root / '.local/unified/nccl'
    records = root / 'artifacts/unified-sys/nccl-build'
    toolchain = (root.parent / 'deepep-monitor/.env').resolve()
    if git(source, 'rev-parse', 'HEAD') != NCCL_REVISION:
        raise RuntimeError('NCCL revision does not match official v2.32.3-1')
    if git(source, 'status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('NCCL tracked source files are modified')
    wrapper = root / 'neutrino/neutrino/utils/capture_nvcc.py'
    command = ['make', '-C', str(source), f'-j{args.jobs}', 'install',
               f'BUILDDIR={build}', f'PREFIX={prefix}', f'CUDA_HOME={toolchain}',
               f'CUDA_LIB={toolchain}/targets/x86_64-linux/lib',
               f'CUDA_INC={toolchain}/targets/x86_64-linux/include',
               f'CXX={toolchain}/bin/x86_64-conda-linux-gnu-c++',
               f'NVCC={wrapper}', 'CUDA_VERSION=13.1', 'KEEP=1', 'VERBOSE=1',
               'NVCC_GENCODE=-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_90,code=compute_90',
               'CUDARTLIB=cudart']
    metadata = {'schema': 1, 'source_revision': NCCL_REVISION, 'source': str(source),
                'prefix': str(prefix), 'build': str(build), 'command': command,
                'cuda_runtime_linkage': 'shared', 'status': 'planned'}
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return 0
    if Path(sys.prefix).resolve() != (root / '.venv-unified').resolve():
        raise RuntimeError('Source examples/moe_training/env.sh first')
    records.mkdir(parents=True, exist_ok=True)
    manifest = records / 'build.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old['command'] != command:
            raise RuntimeError('Build flags changed; use a new build directory')
    wrapper.chmod(wrapper.stat().st_mode | 0o100)
    env = os.environ.copy()
    env.pop('LD_PRELOAD', None)
    env['NEUTRINO_REAL_NVCC'] = str(toolchain / 'bin/nvcc')
    env['NEUTRINO_COMPILER_RECORDS'] = str(records / 'compiler')
    metadata['status'] = 'building'
    manifest.write_text(json.dumps(metadata, indent=2) + '\n')
    with (records / 'build.log').open('a') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    metadata['exit_code'] = result.returncode
    metadata['status'] = 'built' if result.returncode == 0 else 'failed'
    library = prefix / 'lib/libnccl.so.2'
    if result.returncode == 0:
        metadata['library'] = str(library.resolve())
        metadata['sha256'] = hashlib.file_digest(library.open('rb'), 'sha256').hexdigest()
    metadata['source_status_after'] = git(source, 'status', '--porcelain', '--untracked-files=no')
    manifest.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
