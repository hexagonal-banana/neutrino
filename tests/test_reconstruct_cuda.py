"""Fresh synthetic compiler records; no dependence on historical GPU artifacts."""
import json
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neutrino.utils import reconstruct_cuda as gate


class CommandTests(unittest.TestCase):
    def test_only_compiler_commands(self):
        text = '\n'.join(('#$ gcc a.cpp', '#$ : ignored', 'ptxas info : diagnostic',
                          '#$ ptxas -arch=sm_90 --compile-only "a b.ptx" -o "c d.cubin"',
                          '#$ nvlink --arch=sm_90 a.o -o linked.cubin'))
        commands = gate.compiler_commands(text)
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][-3:], ['a b.ptx', '-o', 'c d.cubin'])
        self.assertEqual(gate.architecture(commands[1]), 'sm_90')

    def test_response_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'link.optf').write_text('--arch=sm_90 "a b.o" -o "linked image.cubin"')
            command, = gate.compiler_commands('#$ nvlink -optf link.optf', root, root)
            self.assertEqual(command, ['nvlink', '--arch=sm_90', 'a b.o', '-o', 'linked image.cubin'])
            (root / 'link.optf').write_text('-optf link.optf')
            with self.assertRaisesRegex(gate.ReconstructionError, 'Recursive'):
                gate.compiler_commands('#$ nvlink -optf link.optf', root, root)
            with self.assertRaisesRegex(gate.ReconstructionError, 'escapes'):
                gate.compiler_commands('#$ nvlink -optf ../outside.optf', root, root)

    def test_unsafe_and_incomplete_commands(self):
        for text in ('#$ ptxas -arch=sm_90 $INPUT -o output',
                     '#$ ptxas -arch=sm_90 x.ptx -o output ; bad',
                     '#$ ptxas -arch=sm_90 x.ptx',
                     '#$ ptxas -arch=sm_90 x.ptx -o first -o second',
                     '#$ nvlink -optf absent.optf'):
            with self.subTest(text=text), self.assertRaises(gate.ReconstructionError):
                gate.compiler_commands(text)
        with self.assertRaises(gate.ReconstructionError):
            gate.architecture(['ptxas', 'kernel.ptx', '-o', 'kernel.cubin'])


class HostObjectTests(unittest.TestCase):
    def test_verified_no_device_diagnostic(self):
        header = 'Type: REL (Relocatable file)\nMachine: Advanced Micro Devices X86-64\n'
        for status, stdout, stderr in ((0, '', ''), (255, '',
                "cuobjdump info    : File '/host.o' does not contain device code\n")):
            with self.subTest(status=status), \
                    patch.object(gate.subprocess, 'check_output', return_value=header), \
                    patch.object(gate.subprocess, 'run', return_value=SimpleNamespace(
                        returncode=status, stdout=stdout, stderr=stderr)) as run, \
                    patch.object(gate, 'sha256', return_value='hash'):
                self.assertEqual(gate.verify_host_object(Path('/toolchain'), '/host.o', {}),
                                 {'path': '/host.o', 'sha256': 'hash'})
                self.assertEqual(run.call_count, 2)

    def test_unknown_errors_and_device_objects_fail_closed(self):
        header = 'Type: REL\nMachine: Advanced Micro Devices X86-64\n'
        for code, text in ((255, 'corrupt ELF'), (1, ''), (0, 'ELF file 1: device.sm_90.cubin')):
            with self.subTest(code=code, text=text), \
                    patch.object(gate.subprocess, 'check_output', return_value=header), \
                    patch.object(gate.subprocess, 'run', return_value=SimpleNamespace(
                        returncode=code, stdout=text, stderr='')):
                with self.assertRaises(gate.ReconstructionError):
                    gate.verify_host_object(Path('/toolchain'), '/host.o', {})
        for text in (header + '.nv_fatbin', header + '.cuda', 'Type: REL\nMachine: NVIDIA CUDA\n', ''):
            with patch.object(gate.subprocess, 'check_output', return_value=text), \
                    patch.object(gate.subprocess, 'run') as run:
                with self.assertRaises(gate.ReconstructionError):
                    gate.verify_host_object(Path('/toolchain'), '/host.o', {})
                run.assert_not_called()


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = self.root / 'compiler'
        self.records.mkdir()
        self.output = self.root / 'rebuilt'
        self.commands = []

    def record(self, name, command, *, status='passed'):
        directory = self.records / name
        directory.mkdir()
        (directory / 'command.json').write_text(json.dumps({
            'cwd': str(self.root), 'compiler': '/toolchain/bin/nvcc',
            'output': str(self.root / (name + '.o')), 'status': status}))
        (directory / 'compiler.log').write_text('#$ ' + shlex.join(command))

    def fixture(self):
        (self.root / 'kernel.ptx').write_text('.version 9.0\n.target sm_90\nfence.acq_rel.sys;\n')
        (self.root / 'kernel.cubin').write_bytes(b'original')
        (self.root / 'linked.cubin').write_bytes(b'original')
        self.record('a', ['ptxas', '-arch=sm_90', '--compile-only', 'kernel.ptx', '-o', 'kernel.cubin'])
        self.record('b', ['nvlink', '--arch=sm_90', 'a.o',
                          '--register-link-binaries=original.reg.c', '-o', 'linked.cubin'])

    def run_compiler(self, command, **kwargs):
        self.commands.append(command)
        self.assertNotIn('LD_PRELOAD', kwargs['env'])
        Path(command[command.index('-o') + 1]).write_bytes(b'reconstructed')
        return SimpleNamespace(returncode=0)

    def test_success_rewrites_link_inputs_and_registration(self):
        self.fixture()
        with patch.object(gate.subprocess, 'run', side_effect=self.run_compiler), \
                patch.object(gate, 'disassembly', return_value='same'):
            report = gate.reconstruct(self.records, self.output)
        self.assertEqual(report['status'], 'passed')
        self.assertFalse(report['runtime_verified'])
        self.assertEqual(report['assemblies'][0]['system_scope_opcode_candidates'], 1)
        self.assertEqual(report['links'][0]['reconstructed_inputs'], 1)
        self.assertIn(str(self.output / 'a/kernel.cubin'), self.commands[1])
        self.assertIn('--register-link-binaries=' + str(self.output / 'b/linked.reg.c'), self.commands[1])
        self.assertEqual((self.root / 'kernel.cubin').read_bytes(), b'original')

    def test_mismatch_preserves_evidence_and_blocks_link(self):
        self.fixture()
        def dump(toolchain, path, kind):
            return 'changed' if Path(path).is_relative_to(self.output) else 'original'
        with patch.object(gate.subprocess, 'run', side_effect=self.run_compiler), \
                patch.object(gate, 'disassembly', side_effect=dump):
            report = gate.reconstruct(self.records, self.output)
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(report['mismatch']['sass_equal'])
        self.assertTrue((self.output / 'a/changed.sass').exists() is False)
        self.assertEqual((self.output / 'a/rebuilt.sass').read_text(), 'changed')
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(json.loads((self.output / 'reconstruction.json').read_text())['status'], 'failed')

    def test_empty_and_incomplete_records_fail_closed(self):
        self.assertEqual(gate.reconstruct(self.records, self.output)['status'], 'failed')
        self.record('a', [], status='running')
        report = gate.reconstruct(self.records, self.root / 'second')
        self.assertIn('Incomplete compiler record', report['error'])

    def test_no_overwriting_original_or_existing_results(self):
        with self.assertRaises(gate.ReconstructionError):
            gate.reconstruct(self.records, self.records / 'out')
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            gate.reconstruct(self.records, self.output)


if __name__ == '__main__':
    unittest.main()
