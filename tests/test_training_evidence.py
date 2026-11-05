"""CPU-only provenance and failure-manifest checks using fresh synthetic artifacts."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples/moe_training'))
import launch
import preflight
import dataset_helper


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.addCleanup(patch.stopall)
        patch.object(launch, 'ROOT', self.root).start()
        self.sources = {'nccl-unified': 'pinned'}
        self.harness = {'adapter.py': 'adapter-hash'}
        patch.object(launch, 'harness_files', return_value=self.harness).start()
        patch.object(launch, 'dataset_evidence', return_value={'sha256': 'helper'}).start()
        nccl = self.file('.local/unified/nccl/lib/libnccl.so.2')
        cuda = self.file('toolchain/libcudart.so')
        header = self.file('.local/unified/nccl/include/nccl.h')
        extension = self.file('.venv-unified/deep_ep/_C.so')
        patch.object(launch, 'nccl_header', return_value=header).start()
        self.preflight = {'status': 'passed', 'sources': self.sources, 'harness_files': self.harness,
                          'config_checks': [{'workload': launch.Workload().manifest()}],
                          'libraries': {'nccl': nccl, 'cuda_runtime': cuda, 'nccl_header': header},
                          'dataset_helper': {'sha256': 'helper'}, 'deep_ep_extension': extension['path'],
                          'deep_ep_extension_sha256': extension['sha256']}
        self.file('artifacts/unified-sys/nccl-build/build.json', json.dumps({
            'status': 'built', 'source_revision': 'pinned', 'library': nccl['path'], 'sha256': nccl['sha256']}))
        original = self.file('artifacts/unified-sys/nccl-build/compiler/record/kernel.cubin')
        rebuilt = self.file('artifacts/unified-sys/reconstruction/record/kernel.cubin')
        ptx = self.file('artifacts/unified-sys/nccl-build/compiler/record/kernel.ptx')
        self.directory = self.root / 'artifacts/unified-sys/reconstruction'
        image = {'original': original['path'], 'original_sha256': original['sha256'],
                 'rebuilt': 'record/kernel.cubin', 'rebuilt_sha256': rebuilt['sha256'],
                 'sass_equal': True, 'resources_equal': True}
        self.reconstruction = {'status': 'passed', 'target': 'sm_90', 'links': [image],
                               'assemblies': [{**image, 'ptx': ptx['path'], 'ptx_sha256': ptx['sha256']}]}

    def file(self, relative, text='fixture'):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return {'path': str(path), 'sha256': preflight.digest(path)}

    def validate(self):
        return launch.validate_evidence(self.preflight, self.reconstruction, self.directory,
                                                self.sources, launch.Workload())

    def test_current_evidence(self):
        self.assertEqual(self.validate()['harness_files'], self.harness)

    def test_changed_harness_or_wrong_topology(self):
        self.preflight['harness_files'] = {'adapter.py': 'old'}
        with self.assertRaisesRegex(RuntimeError, 'current passing'):
            self.validate()
        self.preflight['harness_files'] = self.harness
        self.preflight['config_checks'][0]['workload']['tensor_parallel_size'] = 2
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_mixed_evidence_requires_exact_workload(self):
        workload = launch.Workload(tensor_parallel_size=2, expert_parallel_size=4, global_batch_size=4)
        with self.assertRaisesRegex(RuntimeError, 'current passing'):
            launch.validate_evidence(self.preflight, self.reconstruction, self.directory, self.sources, workload)
        self.preflight['config_checks'].append({'workload': workload.manifest()})
        self.assertEqual(launch.validate_evidence(self.preflight, self.reconstruction, self.directory,
                                                 self.sources, workload)['harness_files'], self.harness)
        self.preflight['config_checks'][-1]['workload']['global_batch_size'] = 8
        with self.assertRaisesRegex(RuntimeError, 'current passing'):
            launch.validate_evidence(self.preflight, self.reconstruction, self.directory, self.sources, workload)

    def test_workload_file_dry_run_and_validation(self):
        from dataclasses import asdict
        workload = launch.Workload(tensor_parallel_size=2, expert_parallel_size=4, global_batch_size=4,
                                   hidden_size=4096, num_attention_heads=32,
                                   ffn_hidden_size=14336, seq_length=4096)
        path = self.file('workload.json', json.dumps(asdict(workload)))['path']
        self.assertEqual(launch.Workload.from_file(path), workload)
        argv = ['launch.py', '--workload-config', path]
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()) as output, \
                patch.object(launch, 'require_idle', side_effect=AssertionError('GPU query forbidden')):
            self.assertEqual(launch.main(), 0)
        payload, _ = json.JSONDecoder().raw_decode(output.getvalue())
        self.assertEqual(payload['workload'], workload.manifest())
        for extra in (['--tensor-parallel-size', '2'], ['--profile-kernels', '--mode', 'fixture'],
                      ['--profile-kernels', '--trace', '--runtime-build', 'x', '--catalog', 'x']):
            with patch.object(sys, 'argv', argv + extra), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                launch.main()
        for content, error in (({'hidden_size': 4097}, ValueError), ({'typo': 1}, TypeError)):
            Path(path).write_text(json.dumps(content))
            with self.assertRaises(error):
                launch.Workload.from_file(path)

    def test_larger_model_requires_fresh_config_evidence(self):
        from dataclasses import replace
        workload = replace(launch.Workload(), hidden_size=4096, num_attention_heads=32, seq_length=4096)
        with self.assertRaisesRegex(RuntimeError, 'current passing'):
            launch.validate_evidence(self.preflight, self.reconstruction, self.directory, self.sources, workload)
        self.preflight['config_checks'].append({'workload': workload.manifest()})
        self.assertEqual(launch.validate_evidence(self.preflight, self.reconstruction, self.directory,
                                                 self.sources, workload)['harness_files'], self.harness)

    def test_changed_dataset_helper(self):
        self.preflight['dataset_helper'] = {'sha256': 'old'}
        with self.assertRaisesRegex(RuntimeError, 'Dataset helper'):
            self.validate()

    def test_changed_library_header_and_extension(self):
        paths = [item['path'] for item in self.preflight['libraries'].values()]
        paths.append(self.preflight['deep_ep_extension'])
        for filename in paths:
            with self.subTest(filename=filename):
                path = Path(filename)
                path.write_text('changed')
                with self.assertRaises(RuntimeError):
                    self.validate()
                path.write_text('fixture')

    def test_reconstruction_errors_and_changed_images(self):
        original = copy.deepcopy(self.reconstruction)
        for key, value in (('status', 'failed'), ('target', 'sm_120'), ('links', [])):
            self.reconstruction[key] = value
            with self.assertRaises(RuntimeError):
                self.validate()
            self.reconstruction = copy.deepcopy(original)
        for key, value in (('sass_equal', False), ('resources_equal', False),
                           ('rebuilt_sha256', 'changed'), ('rebuilt', '../../outside.cubin'),
                           ('original', '/another-build/kernel.cubin')):
            self.reconstruction['links'][0][key] = value
            with self.assertRaises(RuntimeError):
                self.validate()
            self.reconstruction = copy.deepcopy(original)
        Path(self.reconstruction['assemblies'][0]['ptx']).write_text('changed')
        with self.assertRaisesRegex(RuntimeError, 'PTX changed'):
            self.validate()

    def test_baseline_rejects_inherited_tracer(self):
        allowed = {'LD_PRELOAD': self.preflight['libraries']['nccl']['path'], 'NEUTRINO_BUILD_DIR': '/build'}
        launch.require_clean_baseline(allowed)
        for key, value in (('LD_PRELOAD', '/external/preload.so'), ('LD_AUDIT', '/audit.so'),
                           ('NEUTRINO_ENABLE', '1'), ('NEUTRINO_SESSION_ID', 'inherited')):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                launch.require_clean_baseline({**allowed, key: value})

    def test_evidence_only_never_queries_gpus(self):
        preflight_path = self.file('preflight.json', json.dumps(self.preflight))['path']
        reconstruction_path = self.file('artifacts/unified-sys/reconstruction/reconstruction.json',
                                        json.dumps(self.reconstruction))['path']
        args = ['launch.py', '--mode', 'fixture', '--check-evidence', '--preflight', preflight_path,
                '--reconstruction', reconstruction_path]
        with patch.object(sys, 'argv', args), patch.object(sys, 'prefix', str(self.root / '.venv-unified')), \
                patch.dict(os.environ, {}, clear=True), patch.object(launch, 'verify_sources', return_value=self.sources), \
                patch.object(launch, 'require_idle', side_effect=AssertionError('GPU query forbidden')), \
                patch.object(launch.subprocess, 'Popen', side_effect=AssertionError('Worker forbidden')), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(launch.main(), 0)
        self.assertFalse(json.loads(output.getvalue())['gpu_execution'])

    def test_prelaunch_failure_writes_manifest_without_workers(self):
        preflight_path = self.file('preflight.json', json.dumps(self.preflight))['path']
        reconstruction_path = self.file('reconstruction.json', json.dumps(self.reconstruction))['path']
        run_dir = self.root / 'artifacts/unified-sys/failed-run'
        devices = [{'index': i, 'uuid': f'GPU-{i}'} for i in range(8)]
        args = ['launch.py', '--mode', 'fixture', '--execute', '--allocation-confirmed',
                '--preflight', preflight_path, '--reconstruction', reconstruction_path,
                '--run-dir', str(run_dir)]
        with patch.object(sys, 'argv', args), patch.object(sys, 'prefix', str(self.root / '.venv-unified')), \
                patch.dict(os.environ, {}, clear=True), patch.object(launch, 'verify_sources', return_value=self.sources), \
                patch.object(launch, 'validate_evidence', return_value={}), \
                patch.object(launch, 'launch_lock', return_value=contextlib.nullcontext()), \
                patch.object(launch, 'require_idle', return_value=devices), \
                patch.object(launch, 'check_once', side_effect=launch.UnsafeGPUState('became busy')), \
                patch.object(launch.subprocess, 'Popen') as popen, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(launch.main(), 3)
        popen.assert_not_called()
        result = json.loads((run_dir / 'validation.json').read_text())
        self.assertEqual(result['status'], 'failed')
        self.assertIn('became busy', result['error'])


class DatasetHelperTests(unittest.TestCase):
    def test_install_reuses_only_verified_helper_without_compilation(self):
        utils = SimpleNamespace(__file__=str(dataset_helper.SOURCE / 'utils.py'))
        package = SimpleNamespace(utils=utils)
        helper = SimpleNamespace(__file__='/isolated/helper.so')
        metadata = {'path': helper.__file__, 'sha256': 'hash'}
        with patch.dict(sys.modules, {'megatron.core.datasets': package, dataset_helper.NAME: helper}), \
                patch.object(dataset_helper, 'evidence', return_value=metadata) as evidence, \
                patch.object(dataset_helper.subprocess, 'run') as run:
            self.assertEqual(dataset_helper.install(), metadata)
            utils.compile_helpers()
            run.assert_not_called()
            evidence.return_value = {'path': '/isolated/helper.so', 'sha256': 'changed'}
            with self.assertRaisesRegex(RuntimeError, 'changed during execution'):
                utils.compile_helpers()
            helper.__file__ = '/unexpected/helper.so'
            with self.assertRaisesRegex(RuntimeError, 'different dataset helper'):
                dataset_helper.install()

    def test_changed_binary_or_input_fails_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suffix = dataset_helper.sysconfig.get_config_var('EXT_SUFFIX')
            binary = root / ('helpers_cpp' + suffix)
            binary.write_bytes(b'original')
            manifest = root / 'build.json'
            metadata = {'status': 'built', 'inputs': {'source': 'hash'}, 'path': str(binary.resolve()),
                        'sha256': preflight.digest(binary)}
            manifest.write_text(json.dumps(metadata))
            with patch.object(dataset_helper, 'BUILD', root), patch.object(dataset_helper, 'MANIFEST', manifest), \
                    patch.object(dataset_helper, 'inputs', return_value={'source': 'hash'}) as inputs:
                self.assertEqual(dataset_helper.evidence()['sha256'], metadata['sha256'])
                inputs.return_value = {'source': 'changed'}
                with self.assertRaises(RuntimeError):
                    dataset_helper.evidence()
                inputs.return_value = {'source': 'hash'}
                binary.write_bytes(b'changed')
                with self.assertRaises(RuntimeError):
                    dataset_helper.evidence()


class HeaderTests(unittest.TestCase):
    def test_header_matches_build_and_expected_version(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(preflight, 'ROOT', Path(temp)):
            text = '#define NCCL_MAJOR 2\n#define NCCL_MINOR 32\n#define NCCL_PATCH 3\n#define NCCL_VERSION_CODE 23203\n'
            installed = Path(temp) / '.local/unified/nccl/include/nccl.h'
            built = Path(temp) / '.build/unified/nccl/include/nccl.h'
            for path in (installed, built):
                path.parent.mkdir(parents=True)
                path.write_text(text)
            self.assertEqual(preflight.nccl_header()['version']['VERSION_CODE'], 23203)
            installed.write_text(text.replace('23203', '22907'))
            with self.assertRaisesRegex(RuntimeError, 'version mismatch'):
                preflight.nccl_header()
            installed.write_text(text + '// changed\n')
            with self.assertRaisesRegex(RuntimeError, 'differs'):
                preflight.nccl_header()

    def test_megatron_argument_exit_is_failure(self):
        with patch.object(sys, 'argv', ['preflight.py', '--imports']), \
                patch.object(preflight, 'verify_sources', return_value={}), \
                patch.object(preflight, 'harness_files', return_value={}), \
                patch.object(preflight, 'import_checks', side_effect=SystemExit(2)), \
                patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(preflight.main(), 1)
        self.assertEqual(json.loads(output.getvalue())['error'], 'SystemExit: 2')


if __name__ == '__main__':
    unittest.main()
