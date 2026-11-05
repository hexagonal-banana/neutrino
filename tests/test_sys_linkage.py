"""Fresh multi-image fixtures for device-call closure and hidden-context ABI."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neutrino.sys_sync.abi import parameter_layout
from neutrino.sys_sync.context import PARAMETER, propagate_context
from neutrino.sys_sync.linkage import link_group
from neutrino.sys_sync.ptx import PTXError, parse

HEADER = '.version 9.1\n.target sm_90\n.address_size 64\n'
CALLEES = HEADER + '''
.visible .global .align 4 .u32 state;
.visible .func first() { membar.sys; ret; }
.visible .func second() { fence.sc.sys; ret; }
'''
TABLE = HEADER + '''
.extern .func first();
.extern .func second();
.visible .global .align 8 .u64 callbacks[3] = {first, second, 0};
'''
CALLER = HEADER + '''
.extern .global .align 8 .b8 callbacks[];
.visible .entry dispatch(.param .align 16 .b8 blob[4096]) {
 .reg .b64 %rd<4>;
 .reg .b32 %r;
 .reg .pred %p;
 mov.u32 %r, %tid.x;
 mul.wide.u32 %rd0, %r, 8;
 mov.u64 %rd1, callbacks;
 add.u64 %rd2, %rd1, %rd0;
 ld.global.u64 %rd3, [%rd2];
 { proto: .callprototype () _ ();
   call %rd3, (), proto;
 }
 ret;
}
.visible .entry other() { ret; }
'''


def rewrite(sources, **kwargs):
    return propagate_context(link_group(sources, **kwargs), max_parameter_bytes=32764)


class LinkageTests(unittest.TestCase):
    def test_cross_unit_direct_call_and_original_layouts(self):
        sources = {'callee': HEADER + '.visible .func f(.param .u64 x) { membar.sys; ret; }',
                   'caller': HEADER + '.extern .func f(.param .u64 other); .entry k(.param .u32 n) {'
                   '.param .u64 arg; call.uni f, (arg); ret; }'}
        group = link_group(sources)
        self.assertEqual(group.calls[0].targets, (('callee', 'f'),))
        result = propagate_context(group, max_parameter_bytes=32764)
        again = link_group(result.sources)
        self.assertEqual(again.calls[0].argument_count, 2)
        for unit, original in group.modules.items():
            for before, after in zip(original.functions, again.modules[unit].functions):
                a = parameter_layout(sources[unit], before)
                b = parameter_layout(result.sources[unit], after)
                self.assertEqual(a.parameters, b.parameters[:-1])
                self.assertEqual(b.parameters[-1].offset, a.context_offset())
        self.assertFalse(result.manifest['runtime_verified'])
        self.assertFalse(result.manifest['site_instrumentation'])

    def test_same_local_symbol_in_distinct_units_is_not_merged(self):
        source = HEADER + '.func f() { ret; } .entry k() { call f; ret; }'
        group = link_group({'a': source, 'b': source})
        self.assertEqual([call.targets for call in group.calls], [(('a', 'f'),), (('b', 'f'),)])
        result = rewrite({'a': source, 'b': source})
        self.assertEqual(len(link_group(result.sources).calls), 2)

    def test_return_parameter_and_predicate_preserved(self):
        source = HEADER + '''
.func (.param .b32 result) f(.param .u32 x) { ret; }
.entry k() { .reg .pred %p; .param .u32 x; .param .b32 result;
 @!%p call.uni (result), f, (x); ret; }
'''
        result = rewrite({'x': source})
        call = next(i for i in parse(result.sources['x']).functions[1].instructions if i.opcode == 'call.uni')
        self.assertEqual(call.predicate, '!%p')
        self.assertIn('(result), f, (x, ', result.sources['x'])
        self.assertEqual(link_group(result.sources).calls[0].argument_count, 2)

    def test_table_provenance_across_complete_group_and_null_guard(self):
        group = link_group({'a': CALLER, 'b': TABLE, 'c': CALLEES})
        call = group.calls[0]
        self.assertEqual(call.targets, (('c', 'first'), ('c', 'second')))
        self.assertEqual(call.table, ('b', 'callbacks'))
        self.assertTrue(group.tables[0].has_null)
        result = propagate_context(group, max_parameter_bytes=32764)
        self.assertEqual(result.manifest['indirect_guards'][0]['target_count'], 2)
        self.assertIn('@%__neutrino_sys_test trap;', result.sources['a'])
        self.assertIn('.extern .func first(.param .u64', result.sources['a'])
        self.assertIn(f'.callprototype () _ (.param .u64 {PARAMETER})', result.sources['a'])
        self.assertIn('.u64 callbacks[3] = {first, second, 0}', result.sources['b'])
        self.assertIn('.global .align 4 .u32 state;', result.sources['c'])
        self.assertEqual([row['context_offset'] for row in result.manifest['layouts']
                          if row['function'] == 'dispatch'], [4096])

    def test_scoped_call_directives_and_predicated_membership_guard(self):
        source = HEADER + '''
.func f() { ret; }
.entry k() { .reg .b64 %rd; .reg .pred %p;
 { proto: .calltargets f; @!%p call %rd, (), proto; }
 { proto: .calltargets f; call %rd, (), proto; } ret; }
'''
        result = rewrite({'x': source})
        self.assertEqual(len(result.manifest['indirect_guards']), 2)
        self.assertIn('and.pred %__neutrino_sys_test, %__neutrino_sys_test, !%p;', result.sources['x'])
        self.assertEqual(len(link_group(result.sources).calls), 2)

    def test_unknown_external_requires_explicit_unchanged_leaf_contract(self):
        source = HEADER + '.extern .func diagnostic(); .entry k() { call diagnostic; ret; }'
        with self.assertRaisesRegex(PTXError, 'Unresolved device'):
            link_group({'x': source})
        result = rewrite({'x': source}, external_functions=['diagnostic'])
        self.assertIn('.extern .func diagnostic();', result.sources['x'])
        self.assertIn('call diagnostic;', result.sources['x'])
        self.assertEqual(result.manifest['external_leaf_contracts'], ['diagnostic'])

    def test_recursive_direct_calls_propagate_same_context(self):
        source = HEADER + '.func f() { call f; ret; } .entry k() { call f; ret; }'
        self.assertEqual(len(link_group(rewrite({'x': source}).sources).calls), 2)

    def test_missing_group_inputs_and_architecture_mismatch_fail(self):
        for sources in ({'a': CALLER, 'b': TABLE}, {'a': CALLER, 'b': CALLEES},
                        {'a': CALLER, 'b': TABLE, 'c': CALLEES.replace('sm_90', 'sm_90a')}, {}):
            with self.subTest(keys=tuple(sources)), self.assertRaises(PTXError):
                link_group(sources)

    def test_exported_duplicates_and_incompatible_signatures_fail(self):
        cases = [
            {'a': HEADER + '.visible .func f() { ret; }', 'b': HEADER + '.visible .func f() { ret; }'},
            {'a': HEADER + '.visible .func f(.param .u32 x) { ret; }',
             'b': HEADER + '.extern .func f(.param .f32 y);'},
            {'a': HEADER + '.func f(.param .u32 x) { ret; } .entry k() { call f; ret; }'},
            {'a': CALLER, 'b': TABLE, 'c': CALLEES.replace('second()', 'second(.param .u64 x)')},
        ]
        for sources in cases:
            with self.subTest(sources=sources), self.assertRaises(PTXError):
                link_group(sources)

    def test_unresolved_predicated_or_merged_pointer_fails(self):
        variants = [CALLER.replace('ld.global.u64 %rd3, [%rd2];', 'ld.param.u64 %rd3, [blob];'),
                    CALLER.replace('ld.global.u64 %rd3, [%rd2];', '@%p ld.global.u64 %rd3, [%rd2];'),
                    CALLER.replace(' { proto:', ' @%p bra JOIN; JOIN: { proto:'),
                    CALLER.replace('add.u64 %rd2, %rd1, %rd0;', 'xor.b64 %rd2, %rd1, %rd0;')]
        for source in variants:
            with self.subTest(source=source), self.assertRaises(PTXError):
                link_group({'a': source, 'b': TABLE, 'c': CALLEES})

    def test_function_linkage_does_not_leak_into_local_table(self):
        source = HEADER + '.visible .func f() { ret; } .global .u64 table[1] = {f};'
        group = link_group({'a': source, 'b': source.replace('f()', 'g()').replace('{f}', '{g}')})
        self.assertEqual([table.linkage for table in group.tables], ['local', 'local'])

    def test_indirect_guard_cannot_resolve_a_shadowed_cross_unit_symbol(self):
        caller = CALLER + '\n.func first() { ret; }\n'
        with self.assertRaisesRegex(PTXError, 'shadowed'):
            rewrite({'a': caller, 'b': TABLE, 'c': CALLEES})

    def test_private_table_target_cannot_escape_its_unit(self):
        table = TABLE.replace('.extern .func first();', '.func first() { ret; }')
        with self.assertRaisesRegex(PTXError, 'module-private'):
            rewrite({'a': CALLER, 'b': table, 'c': CALLEES})

    def test_reserved_symbols_and_parameter_limit_fail(self):
        with self.assertRaisesRegex(PTXError, 'collides'):
            rewrite({'x': HEADER + '.entry __neutrino_sys_original() { ret; }'})
        with self.assertRaisesRegex(PTXError, 'exceeds'):
            propagate_context(link_group({'x': CALLER, 'b': TABLE, 'c': CALLEES}), max_parameter_bytes=4096)

    @unittest.skipUnless(os.environ.get('UNIFIED_TOOLCHAIN'), 'Selected CUDA toolchain required; no GPU needed')
    def test_hopper_assembles_and_links_guarded_table_and_large_entry(self):
        result = rewrite({'caller': CALLER, 'table': TABLE, 'callees': CALLEES})
        toolchain = Path(os.environ['UNIFIED_TOOLCHAIN'])
        env = os.environ.copy()
        env.pop('LD_PRELOAD', None)
        env['CUDA_VISIBLE_DEVICES'] = ''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            objects = []
            for unit, source in result.sources.items():
                path, obj = root / f'{unit}.ptx', root / f'{unit}.o'
                path.write_text(source)
                command = [str(toolchain / 'bin/ptxas'), '-arch=sm_90', '-c', str(path), '-o', str(obj)]
                completed = subprocess.run(command, env=env, text=True, capture_output=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                objects.append(str(obj))
            completed = subprocess.run([str(toolchain / 'bin/nvlink'), '-arch=sm_90', *objects,
                                        '-o', str(root / 'linked.cubin')], env=env, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            elf = subprocess.check_output([str(toolchain / 'bin/cuobjdump'), '--dump-elf',
                                           str(root / 'linked.cubin')], env=env, text=True)
            self.assertRegex(elf, r'Ordinal\s*:\s*0x1\s+Offset\s*:\s*0x1000')


if __name__ == '__main__':
    unittest.main()
