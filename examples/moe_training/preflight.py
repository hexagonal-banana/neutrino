"""CPU-only source, import, argument, and loaded-library preflight."""
import argparse
import contextlib
import ctypes
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch

from config import Workload
from setup_environment import ROOT, PINS, verify_sources

MEGATRON = ROOT / 'third_party/Megatron-LM-unified'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def loaded_library(name, allowed_root, function, expected_version):
    paths = {Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
             if name in line}
    if len(paths) != 1:
        raise RuntimeError(f'Expected exactly one loaded {name}: {paths}')
    path, = paths
    if not path.is_relative_to(allowed_root.resolve()):
        raise RuntimeError(f'Unexpected loaded {name}: {path}')
    library = ctypes.CDLL(str(path), mode=os.RTLD_LOCAL | os.RTLD_NOLOAD)
    version_call = getattr(library, function)
    version_call.argtypes = [ctypes.POINTER(ctypes.c_int)]
    version_call.restype = ctypes.c_int
    version = ctypes.c_int()
    code = version_call(ctypes.byref(version))
    if code != 0 or version.value != expected_version:
        raise RuntimeError(f'{name} runtime version mismatch: status={code}, version={version.value}')
    return {'path': str(path), 'sha256': digest(path), 'runtime_version': version.value}


def harness_files():
    """Content identity prevents stale fixtures from validating a changed adapter/workload."""
    example = Path(__file__).resolve().parent
    paths = [*example.glob('*.py'), *example.glob('*.sh'), ROOT / 'neutrino/neutrino/capture.py']
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def nccl_header():
    path = ROOT / '.local/unified/nccl/include/nccl.h'
    expected = {'MAJOR': 2, 'MINOR': 32, 'PATCH': 3, 'VERSION_CODE': 23203}
    text = path.read_text()
    for name, value in expected.items():
        definitions = re.findall(rf'^#define\s+NCCL_{name}\s+(\d+)\s*$', text, re.M)
        if definitions != [str(value)]:
            raise RuntimeError(f'NCCL header version mismatch: {name}: {definitions}')
    checksum = digest(path)
    if checksum != digest(ROOT / '.build/unified/nccl/include/nccl.h'):
        raise RuntimeError('Installed NCCL header differs from the isolated build')
    return {'path': str(path), 'sha256': checksum, 'version': expected}


def runtime_libraries():
    build = json.loads((ROOT / 'artifacts/unified-sys/nccl-build/build.json').read_text())
    nccl = loaded_library('libnccl.so', ROOT / '.local/unified/nccl', 'ncclGetVersion', 23203)
    if build['status'] != 'built' or nccl['sha256'] != build['sha256']:
        raise RuntimeError('Loaded NCCL does not match the isolated build manifest')
    cuda = loaded_library('libcudart.so', ROOT.parent / 'deepep-monitor/.env',
                          'cudaRuntimeGetVersion', 13010)
    return {'nccl': nccl, 'cuda_runtime': cuda, 'nccl_header': nccl_header()}


def import_checks(extra_workload=None):
    if Path(sys.prefix).resolve() != (ROOT / '.venv-unified').resolve():
        raise RuntimeError('Source env.sh and use the isolated Python')
    if os.environ.get('EP_SUPPRESS_NCCL_CHECK', '0') != '0':
        raise RuntimeError('DeepEP NCCL compatibility checks must remain enabled')
    sys.path.insert(0, str(MEGATRON))
    import torch

    def forbidden(*args, **kwargs):
        raise RuntimeError('CPU preflight attempted CUDA initialization')

    with patch.object(torch.cuda, '_lazy_init', forbidden):
        import deep_ep
        import deep_ep._C
        import transformer_engine.pytorch
        from adapter import install
        api = install(MEGATRON)
        if any(api['counts'].values()):
            raise RuntimeError('CPU preflight executed communication')
        package_root = Path(deep_ep.__file__).resolve().parent
        if not package_root.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(f'DeepEP is outside the overlay: {package_root}')
        for path in (ROOT / 'third_party/DeepEP/deep_ep').rglob('*.py'):
            relative = path.relative_to(ROOT / 'third_party/DeepEP/deep_ep')
            if '__pycache__' not in relative.parts and digest(path) != digest(package_root / relative):
                raise RuntimeError(f'Installed DeepEP Python source mismatch: {relative}')
        deep_ep.check_nccl_so()
        libraries = runtime_libraries()
        from dataset_helper import install as install_dataset_helper
        helper = install_dataset_helper()
        from megatron.core.datasets import helpers, utils as dataset_utils
        import numpy as np
        indices = helpers.build_sample_idx(np.array([8], dtype=np.int32), np.array([0], dtype=np.int32),
                                           2, 1, 8)
        np.testing.assert_array_equal(indices, [[0, 0], [0, 2], [0, 4], [0, 6]])
        dataset_utils.compile_helpers()
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.argument_utils import core_transformer_config_from_args
        config_checks = []
        workloads = [Workload(world_size=w, expert_parallel_size=w, global_batch_size=w) for w in (2, 8)]
        workloads.append(Workload(tensor_parallel_size=2, expert_parallel_size=4, global_batch_size=4))
        if extra_workload is not None and extra_workload not in workloads:
            workloads.append(extra_workload.validate())
        for workload in workloads:
            world = workload.world_size
            text = io.StringIO()
            with patch.object(sys, 'argv', ['pretrain_gpt.py', *workload.megatron_args()]), \
                    patch.dict(os.environ, {'WORLD_SIZE': str(world), 'RANK': '0',
                                            **({'CUDA_DEVICE_MAX_CONNECTIONS': '1'}
                                               if workload.tensor_parallel_size > 1 else {})}), \
                    patch.object(torch.cuda, 'get_device_capability', return_value=(9, 0)), \
                    patch('megatron.training.arguments.get_device_arch_version', return_value=9), \
                    contextlib.redirect_stdout(text):
                args = validate_args(parse_args())
                model = core_transformer_config_from_args(args)
            if (args.data_parallel_size != workload.dense_dp or
                    args.tensor_model_parallel_size != workload.tensor_parallel_size or
                    args.expert_model_parallel_size != workload.expert_parallel_size or
                    args.expert_tensor_parallel_size != workload.expert_tensor_parallel_size or
                    args.sequence_parallel != (workload.tensor_parallel_size > 1) or args.rerun_mode != 'disabled' or
                    args.cuda_graph_impl != 'none' or args.overlap_param_gather or
                    model.moe_flex_dispatcher_backend != 'deepepv2'):
                raise RuntimeError('Parsed Megatron configuration differs from the workload contract')
            config_checks.append({'world_size': world, 'dense_dp': args.data_parallel_size,
                                  'ep': args.expert_model_parallel_size, 'tp': args.tensor_model_parallel_size,
                                  'expert_dp': workload.expert_dp, 'workload': workload.manifest(),
                                  'hardware_capability_mocked': True})
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU preflight initialized CUDA')
        return {'libraries': libraries, 'config_checks': config_checks, 'dataset_helper': helper,
                'deep_ep_extension': str(Path(deep_ep._C.__file__).resolve()),
                'deep_ep_extension_sha256': digest(deep_ep._C.__file__),
                'versions': {name: importlib.metadata.version(name) for name in
                             ('torch', 'deep_ep', 'transformer_engine', 'toml')},
                'cuda_initialized': False, 'gpu_runtime_validated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imports', action='store_true', help='Hide GPUs and validate framework imports/CLI')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workload-config', type=Path)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    report = {'schema': 1, 'status': 'failed', 'workload': Workload().manifest(),
              'gpu_runtime_validated': False}
    try:
        report['sources'] = verify_sources()
        report['harness_files'] = harness_files()
        workload = Workload.from_file(args.workload_config) if args.workload_config else Workload()
        report['workload'] = workload.manifest()
        if args.imports:
            report.update(import_checks(workload))
        report['status'] = 'passed'
    except (Exception, SystemExit) as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    if args.output:
        output = args.output.resolve()
        if not output.is_relative_to(ROOT / 'artifacts/unified-sys'):
            raise RuntimeError('Preflight output must be inside artifacts/unified-sys')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
    print(json.dumps(report, indent=2))
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
