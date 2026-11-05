"""Build gates validate fresh input identities and distinct CUDA image owners."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neutrino.sys_sync.build import build_context, load_reconstructed_group
from neutrino.sys_sync.ptx import PTXError

HEADER = '.version 9.1\n.target sm_90\n.address_size 64\n'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fixture(root):
    evidence = root / 'evidence'
    evidence.mkdir()
    items = []
    for name, relocatable in [('reloc', True), ('whole', False)]:
        data = b'\x7fELF\x02\x01' + bytes(10) + struct.pack('<HH', 1 if relocatable else 2, 190)
        original = root / f'{name}.original.cubin'
        original.write_bytes(data)
        (evidence / f'{name}.cubin').write_bytes(data)
        ptx = root / f'{name}.ptx'
        ptx.write_text(HEADER + '.visible .entry same_entry() { membar.sys; ret; }')
        items.append({'original': str(original), 'original_sha256': digest(data),
                      'rebuilt': f'{name}.cubin', 'rebuilt_sha256': digest(data),
                      'sass_equal': True, 'resources_equal': True, 'relocatable': relocatable,
                      'ptx': str(ptx), 'ptx_sha256': digest(ptx.read_bytes()),
                      'command': ['/toolchain/bin/ptxas', '-arch=sm_90', *(['-c'] if relocatable else []),
                                  str(ptx), '-o', str(evidence / f'{name}.cubin')]})
    link = dict(items[0], rebuilt='linked.cubin', command=[
        '/toolchain/bin/nvlink', '-arch=sm_90', *(str(evidence / item['rebuilt']) for item in items),
        '-o', str(evidence / 'linked.cubin')], reconstructed_inputs=2)
    (evidence / 'linked.cubin').write_bytes((root / 'reloc.original.cubin').read_bytes())
    report = {'schema': 1, 'status': 'passed', 'target': 'sm_90', 'assemblies': items,
              'links': [link], 'host_only_inputs': [], 'other_architectures': []}
    path = evidence / 'reconstruction.json'
    path.write_text(json.dumps(report))
    return path, report


class BuildTests(unittest.TestCase):
    def test_executable_and_relocatable_images_keep_separate_namespaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = fixture(root)
            with patch('neutrino.sys_sync.build.subprocess.run') as run:
                result = build_context(manifest, root / 'output', max_parameter_bytes=32764)
                run.assert_not_called()
            self.assertEqual(result['status'], 'passed', result.get('error'))
            self.assertEqual(result['defined_functions'], 2)
            self.assertEqual([item['owner'] for item in result['contexts']], ['linked', 'whole.cubin'])
            self.assertFalse(result['assembly_verified'])
            self.assertFalse(result['runtime_verified'])
            self.assertFalse(result['site_instrumentation'])

    def test_instruction_probes_require_explicit_mode_and_keep_owner_local_sites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = fixture(root)
            result = build_context(manifest, root / 'output', max_parameter_bytes=32764,
                                   interval_mode='instruction-only')
            self.assertEqual(result['status'], 'passed', result.get('error'))
            self.assertTrue(result['site_instrumentation'])
            self.assertFalse(result['runtime_verified'])
            self.assertFalse(result['polling_intervals_validated'])
            self.assertFalse(result['cooperative_resources_validated'])
            self.assertEqual(result['kind'], 'linked_instruction_probe_gate')
            self.assertEqual([len(c['sites']) for c in result['contexts']], [1, 1])
            self.assertEqual([c['sites'][0]['ordinal'] for c in result['contexts']], [1, 1])
            for row in result['assemblies']:
                self.assertIn('%globaltimer', (root / 'output' / row['ptx']).read_text())
            with self.assertRaisesRegex(PTXError, 'instruction-only'):
                build_context(manifest, root / 'unsupported', max_parameter_bytes=32764,
                              interval_mode='polling')
            self.assertFalse((root / 'unsupported').exists())

    def test_source_or_binary_changes_invalidate_previous_gate(self):
        for target in ('reloc.ptx', 'reloc.original.cubin', 'evidence/reloc.cubin'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, _ = fixture(root)
                (root / target).write_bytes(b'changed')
                with self.assertRaisesRegex(PTXError, 'identity changed'):
                    load_reconstructed_group(manifest, 0)

    def test_missing_duplicate_outside_or_nonmatching_elf_inputs_fail(self):
        mutations = [lambda r: r['links'][0].update(reconstructed_inputs=1),
                     lambda r: r['assemblies'][0].update(rebuilt='../escape.cubin'),
                     lambda r: r['assemblies'][0].update(relocatable=False),
                     lambda r: r['assemblies'][0].update(resources_equal=False),
                     lambda r: r['assemblies'].append(deepcopy(r['assemblies'][0])),
                     lambda r: r.update(status='failed')]
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as temporary:
                manifest, report = fixture(Path(temporary))
                mutate(report)
                manifest.write_text(json.dumps(report))
                with self.assertRaises(PTXError):
                    load_reconstructed_group(manifest, 0)

    def test_failures_recorded_and_existing_outputs_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, report = fixture(root)
            report['status'] = 'failed'
            manifest.write_text(json.dumps(report))
            result = build_context(manifest, root / 'output', max_parameter_bytes=32764)
            self.assertEqual(result['status'], 'failed')
            self.assertTrue((root / 'output/context-build.json').is_file())
            with self.assertRaises(FileExistsError):
                build_context(manifest, root / 'output', max_parameter_bytes=32764)
            with self.assertRaisesRegex(PTXError, 'separate'):
                build_context(manifest, manifest.parent / 'overwrite', max_parameter_bytes=32764)


if __name__ == '__main__':
    unittest.main()
