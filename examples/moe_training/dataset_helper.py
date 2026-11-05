"""Build/load the pinned, unmodified Megatron dataset helper outside its checkout."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

from setup_environment import ROOT, PINS, verify_sources

NAME = 'megatron.core.datasets.helpers_cpp'
SOURCE = ROOT / 'third_party/Megatron-LM-unified/megatron/core/datasets'
BUILD = ROOT / '.build/unified/megatron-helpers'
MANIFEST = ROOT / 'artifacts/unified-sys/dataset-helper/build.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inputs():
    return {'revision': PINS['Megatron-LM-unified'], 'source_sha256': digest(SOURCE / 'helpers.cpp'),
            'makefile_sha256': digest(SOURCE / 'Makefile'),
            'extension_suffix': sysconfig.get_config_var('EXT_SUFFIX'),
            'python_version': sys.version}


def evidence():
    metadata = json.loads(MANIFEST.read_text())
    extension = (BUILD / ('helpers_cpp' + sysconfig.get_config_var('EXT_SUFFIX'))).resolve()
    if (metadata.get('status') != 'built' or metadata.get('inputs') != inputs() or
            metadata.get('path') != str(extension) or digest(extension) != metadata['sha256']):
        raise RuntimeError('Dataset helper build is missing, changed, or incompatible')
    return {'path': str(extension), 'sha256': metadata['sha256'], 'inputs': metadata['inputs']}


def install():
    """Process-local import/build adapter; no changes to Megatron source files."""
    from megatron.core.datasets import utils
    if Path(utils.__file__).resolve().parent != SOURCE.resolve():
        raise RuntimeError('Unexpected Megatron dataset package')
    metadata = evidence()
    existing = sys.modules.get(NAME)
    if existing is not None:
        if Path(existing.__file__).resolve() != Path(metadata['path']):
            raise RuntimeError('A different dataset helper is already imported')
    else:
        spec = importlib.util.spec_from_file_location(NAME, metadata['path'])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[NAME] = module

    def verify_prebuilt():
        if evidence() != metadata:
            raise RuntimeError('Dataset helper changed during execution')

    utils.compile_helpers = verify_prebuilt
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    verify_sources()
    compiler = os.environ.get('CXX')
    if not compiler or not Path(compiler).is_file():
        raise RuntimeError('Source env.sh to select the read-only C++ toolchain')
    env = os.environ.copy()
    env.pop('LD_PRELOAD', None)
    env['CUDA_VISIBLE_DEVICES'] = ''
    includes = subprocess.check_output([sys.executable, '-m', 'pybind11', '--includes'], text=True, env=env).strip()
    extension = BUILD / ('helpers_cpp' + sysconfig.get_config_var('EXT_SUFFIX'))
    command = ['make', '-C', str(BUILD), '-f', str(SOURCE / 'Makefile'),
               f'CXX={compiler}', f'CPPFLAGS={includes}', f'SRC={SOURCE / "helpers.cpp"}', f'OUT={extension}']
    metadata = {'schema': 1, 'status': 'planned', 'inputs': inputs(), 'command': command,
                'compiler_sha256': digest(compiler), 'gpu_execution': False}
    if not args.execute:
        print(json.dumps(metadata, indent=2))
        return 0
    if Path(sys.prefix).resolve() != (ROOT / '.venv-unified').resolve():
        raise RuntimeError('Use the isolated Python')
    if MANIFEST.exists():
        print(json.dumps(evidence(), indent=2))
        return 0
    BUILD.mkdir(parents=True, exist_ok=False)
    MANIFEST.parent.mkdir(parents=True, exist_ok=False)
    try:
        with (MANIFEST.parent / 'build.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        metadata.update(status='built', path=str(extension.resolve()), sha256=digest(extension),
                        source_integrity_after=verify_sources())
    except Exception as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        MANIFEST.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))
    return int(metadata['status'] != 'built')


if __name__ == '__main__':
    raise SystemExit(main())
