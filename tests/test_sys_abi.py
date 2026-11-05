"""Original parameter layouts remain unchanged when reserving a launch context."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neutrino.sys_sync.abi import append_context_parameter, parameter_layout
from neutrino.sys_sync.ptx import PTXError, parse


def module(parameters):
    source = '.version 9.1\n.target sm_90\n.address_size 64\n.visible .entry k(' + parameters + ') { ret; }\n'
    return source, parse(source).functions[0]


class ABITests(unittest.TestCase):
    def test_padding_aggregates_and_scalars(self):
        source, function = module('.param .b8 flag, .param .align 16 .b8 blob[4096], .param .u32 n')
        layout = parameter_layout(source, function)
        self.assertEqual([(p.offset, p.size, p.alignment) for p in layout.parameters],
                         [(0, 1, 1), (16, 4096, 16), (4112, 4, 4)])
        self.assertEqual(layout.size, 4116)
        self.assertEqual(layout.context_offset(), 4120)

    def test_pointer_pointee_alignment_does_not_move_parameter_slot(self):
        source, function = module('.param .u32 n, .param .u64 .ptr .global .align 128 ptr')
        layout = parameter_layout(source, function)
        self.assertEqual([(p.offset, p.size, p.alignment) for p in layout.parameters], [(0, 4, 4), (8, 8, 8)])

    def test_exact_insertion_preserves_original_bytes_and_offsets(self):
        source, function = module('.param .b8 a, /* keep this */ .param .u64 b')
        offset, insertion, before = append_context_parameter(source, function, '__context', max_parameter_bytes=32764)
        changed = source[:offset] + insertion + source[offset:]
        after = parameter_layout(changed, parse(changed).functions[0])
        self.assertEqual(before.parameters, after.parameters[:-1])
        self.assertEqual(after.parameters[-1].offset, before.context_offset())
        self.assertEqual(changed[:offset] + changed[offset + len(insertion):], source)

    def test_zero_argument_entry_and_collision(self):
        source, function = module('')
        offset, insertion, layout = append_context_parameter(source, function, '__context', max_parameter_bytes=8)
        self.assertEqual(layout.size, 0)
        self.assertNotIn(',', insertion)
        changed = source[:offset] + insertion + source[offset:]
        with self.assertRaisesRegex(PTXError, 'collides'):
            append_context_parameter(changed, parse(changed).functions[0], '__context', max_parameter_bytes=16)

    def test_unsafe_or_unknown_abi_fails_closed(self):
        for parameters in ('.param .v2 .u32 v', '.param .b8 bytes[]', '.param .b8 bytes[0]',
                           '.param .align 3 .b8 bytes[4]', '.param .u64 p,',
                           '.param .u64 p, .param .u32 p', '.reg .b64 r',
                           '.param .u64 .ptr .align 3 p'):
            with self.subTest(parameters=parameters), self.assertRaises(PTXError):
                source, function = module(parameters)
                parameter_layout(source, function)
        source, function = module('.param .align 16 .b8 storage[4096]')
        with self.assertRaisesRegex(PTXError, 'exceeds'):
            append_context_parameter(source, function, '__context', max_parameter_bytes=4096)

    @unittest.skipUnless(os.environ.get('UNIFIED_TOOLCHAIN'), 'Selected CUDA toolchain required; no GPU needed')
    def test_hopper_assembly_accepts_large_preserved_parameter_blob(self):
        toolchain = Path(os.environ['UNIFIED_TOOLCHAIN'])
        source, function = module('.param .align 16 .b8 storage[4096]')
        offset, insertion, _ = append_context_parameter(source, function, '__context', max_parameter_bytes=32764)
        changed = source[:offset] + insertion + source[offset:]
        env = os.environ.copy()
        env.pop('LD_PRELOAD', None)
        env['CUDA_VISIBLE_DEVICES'] = ''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'kernel.ptx').write_text(changed)
            subprocess.run([str(toolchain / 'bin/ptxas'), '-arch=sm_90', str(root / 'kernel.ptx'),
                            '-o', str(root / 'kernel.cubin')], env=env, check=True, capture_output=True)
            result = subprocess.check_output([str(toolchain / 'bin/cuobjdump'), '--dump-elf',
                                              str(root / 'kernel.cubin')], text=True, env=env)
            self.assertIn('EIATTR_KPARAM_INFO', result)
            self.assertRegex(result, r'Ordinal\s*:\s*0x0\s+Offset\s*:\s*0x0')
            self.assertRegex(result, r'Ordinal\s*:\s*0x1\s+Offset\s*:\s*0x1000')


if __name__ == '__main__':
    unittest.main()
