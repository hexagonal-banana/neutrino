"""Fresh CPU fixtures for lossless, library-independent PTX site discovery."""
from pathlib import Path
import hashlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neutrino.sys_sync.ptx import PTXError, parse, system_scope, tokenize
from neutrino.sys_sync.audit import inventory

HEADER = '.version 9.1\n.target sm_90\n.address_size 64\n'


class PTXTests(unittest.TestCase):
    def test_explicit_scope_boundaries(self):
        for opcode in ('ld.acquire.gpu.global.u32', 'ld.acquire.cluster.u32',
                       'st.volatile.global.u32', 'ld.relaxed.cluster.global.u32',
                       'ld.global.u32', 'mov.u32', 'nanosleep.u32', 'foo.system.u32',
                       'ld.relaxed.sys.u32', 'st.release.sys.global.u32',
                       'atom.release.sys.global.add.u32', 'atom.sys.global.add.u32',
                       'red.release.sys.global.add.u32',
                       'multimem.ld_reduce.relaxed.sys.global.add.acc::f16.v4.f16x2',
                       'multimem.red.relaxed.sys.add.u32', 'multimem.st.relaxed.sys.global.v4.f32',
                       'cp.async.mbarrier.arrive', 'barrierish.sync', 'fencepost.sysless',
                       'fence.acq_rel.gpu', 'fence.sc.cta', 'fence.sc.cluster',
                       'fence.proxy.async.shared::cta', 'fence.mbarrier_init.release.cluster',
                       'fence.proxy.alias', 'membar.gl', 'membar.cta',
                       'bar.sync', 'bar.warp.sync', 'bar.arrive', 'bar.red.popc.u32',
                       'barrier.sync.aligned', 'barrier.cluster.arrive.relaxed',
                       'barrier.cluster.wait.acquire', 'mbarrier', 'mbarrier.try_wait_extra.shared.b64',
                       'mbarrier.init.shared.b64', 'mbarrier.inval.shared.b64',
                       'mbarrier.arrive.release.cta.shared::cta.b64',
                       'mbarrier.arrive_drop.shared.b64', 'mbarrier.arrive.expect_tx.shared.b64',
                       'mbarrier.expect_tx.relaxed.cta.shared::cta.b64',
                       'mbarrier.complete_tx.relaxed.cluster.shared::cluster.b64',
                       'mbarrier.test_wait.parity.shared.b64', 'mbarrier.pending_count.b64',
                       'cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes'):
            self.assertIsNone(system_scope(opcode), opcode)
        for opcode in ('ld.acquire.sys.u32', 'atom.acquire.sys.global.add.u32',
                       'atom.acq_rel.sys.global.cas.b32',
                       'multimem.ld_reduce.acquire.sys.global.add.acc::f16.v4.f16x2'):
            self.assertEqual(system_scope(opcode)[2], 'ordered_memory_instruction', opcode)
        for opcode in ('fence.sc.sys', 'fence.acq_rel.sys', 'fence.proxy.alias.sys', 'membar.sys'):
            self.assertEqual(system_scope(opcode)[2], 'fence_instruction', opcode)
        for opcode in ('mbarrier.try_wait.shared.b64', 'mbarrier.try_wait.parity.shared.b64',
                       'mbarrier.try_wait.acquire.cta.shared::cta.b64',
                       'mbarrier.try_wait.parity.acquire.cta.shared::cta.b64'):

            self.assertEqual(system_scope(opcode)[2], 'barrier_instruction', opcode)
        with self.assertRaisesRegex(PTXError, 'Unsupported'):
            system_scope('future_operation.sys.u32')
        with self.assertRaisesRegex(PTXError, 'Conflicting'):
            system_scope('ld.acquire.relaxed.sys.u32')

    def test_coordinates_predicates_and_opaque_text(self):
        source = HEADER + '''
// .entry fake() { ld.acquire.sys.u32 %r, [%ptr]; }
.file 1 "source { with // comment markers }.cu"
.visible .entry real(.param .u64 ptr) {
 .reg .b32 %r<4>;
 .reg .pred %p;
 .pragma "ld.acquire.sys.u32; bar.sync; mbarrier.init; fence.sc.cta; fake }";
 .loc 1 17 0
 LOOP: @!%p ld.acquire.sys.global.u32 %r1, [ptr];
 { @%p mbarrier.try_wait.parity.shared.b64 %p, [%ptr], 0; }
 st.release.sys.global.u32 [ptr], %r1;
 mov.u64 %rd1, symbol.sys;
 ret;
}
'''
        module = parse(source)
        self.assertEqual([f.name for f in module.functions], ['real'])
        self.assertEqual(len(module.sites), 2)
        first = module.sites[0]
        self.assertEqual(first.predicate, '!%p')
        self.assertEqual(source[first.start:first.end], '@!%p ld.acquire.sys.global.u32 %r1, [ptr];')
        self.assertEqual(first.line, source[:first.start].count('\n') + 1)
        function = module.functions[0]
        self.assertEqual(source[function.parameter_start:function.parameter_end], '(.param .u64 ptr)')
        self.assertEqual(function.instructions[0].labels, ('LOOP',))
        self.assertEqual(len(function.instructions), 5)
        self.assertEqual(module.sites[1].opcode, 'mbarrier.try_wait.parity.shared.b64')
        self.assertEqual(module.sites[1].predicate, '%p')

    def test_multiple_entries_callees_and_declarations(self):
        source = HEADER + '''
.extern .func (.param .b32 result) external(.param .b64 x);
.visible .func callee(.param .u64 x) { membar.sys; ret; }
.visible .entry alpha() { call.uni callee, (); ret; }
.visible .entry beta() { ld.acquire.sys.b32 %r, [%ptr]; ret; }
'''
        module = parse(source)
        self.assertEqual([f.name for f in module.functions], ['external', 'callee', 'alpha', 'beta'])
        self.assertEqual([f.defined for f in module.functions], [False, True, True, True])
        self.assertEqual([s.function for s in module.sites], ['callee', 'beta'])
        self.assertEqual(module.functions[2].instructions[0].opcode, 'call.uni')

    def test_vector_operands_and_indirect_call_prototype(self):
        source = HEADER + '''
.global .u64 table[2] = {first, second};
.entry indirect() {
 .reg .b32 %r<4>;
 prototype: .callprototype () _ (.param .b64 arg);
 call %rd1, (), prototype;
 multimem.ld_reduce.acquire.sys.global.add.acc::f16.v4.f16x2 {%r0,%r1,%r2,%r3}, [%rd0];
 ret;
}
'''
        module = parse(source)
        instructions = module.functions[0].instructions
        self.assertEqual(len(instructions), 3)
        self.assertEqual(instructions[0].labels, ())
        self.assertEqual(module.sites[0].instruction, 1)
        self.assertEqual(module.sites[0].operation, 'multimem.ld_reduce')

    def test_exact_symbols_no_fuzzy_match(self):
        module = parse(HEADER + '.entry a() { ret; } .entry ab() { membar.sys; ret; }')
        self.assertEqual([site.function for site in module.sites], ['ab'])
        with self.assertRaisesRegex(PTXError, 'Duplicate'):
            parse(HEADER + '.entry a() { ret; } .entry a() { ret; }')

    def test_malformed_input_fails_closed(self):
        for body in ('.entry k() { ret; ', '.entry k() { ld.acquire.sys.u32 %r, [%p] }',
                     '.entry k() { "unterminated }', '.entry k() { /* unterminated }',
                     '.entry k() { future_operation.sys.u32; ret; }',
                     '.entry k() { @ ret; }', '.entry k() { ret; END: }'):
            with self.subTest(body=body), self.assertRaises(PTXError):
                parse(HEADER + body)
        for source in ('', HEADER + HEADER, HEADER.replace('64', '32')):
            with self.subTest(source=source), self.assertRaises(PTXError):
                parse(source)
        with self.assertRaises(PTXError):
            tokenize(HEADER + '\x00')


class InventoryTests(unittest.TestCase):
    def test_source_hash_and_explicit_analysis_stages(self):
        source = HEADER + '.entry k(.param .u64 p) { L: ld.acquire.sys.u32 %r,[%ptr]; @%p bra L; ret; }'
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'fresh.ptx'
            path.write_text(source)
            report = inventory([path, path], 'sm_90', check_abi=True, check_control_flow=True)
        self.assertEqual(report['status'], 'passed')
        self.assertFalse(report['runtime_verified'])
        self.assertFalse(report['polling_intervals_validated'])
        self.assertEqual(report['analyzed_modules'], 1)
        self.assertEqual(report['modules'][0]['sha256'], hashlib.sha256(source.encode()).hexdigest())
        function = report['modules'][0]['functions'][0]
        self.assertEqual(function['parameter_layout']['size'], 8)
        self.assertEqual(function['control_flow']['natural_loops'], 1)

    def test_partial_or_missing_evidence_never_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'fresh.ptx'
            path.write_text(HEADER + '.entry k() { bra unknown; }')
            self.assertEqual(inventory([path], 'sm_90')['status'], 'passed')
            report = inventory([path], 'sm_90', check_control_flow=True)
            self.assertEqual(report['status'], 'failed')
            self.assertIn('Unresolved', report['modules'][0]['functions'][0]['analysis_error'])
            self.assertEqual(inventory([path.with_name('missing.ptx')], 'sm_90')['status'], 'failed')
            self.assertEqual(inventory([path], 'sm_90a')['modules'][0]['status'], 'other_architecture')
        self.assertEqual(inventory([], 'sm_90')['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
