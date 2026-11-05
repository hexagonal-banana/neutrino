"""Static edge recovery, including conditional exits and ambiguous loops."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neutrino.sys_sync.control_flow import analyze
from neutrino.sys_sync.ptx import PTXError, parse


def flow(body):
    return analyze(parse('.version 9.1\n.target sm_90\n.address_size 64\n.entry k() {' + body + '}').functions[0])


class ControlFlowTests(unittest.TestCase):
    def test_single_block_polling_shape(self):
        cfg = flow('L: ld.acquire.sys.u32 %r, [%ptr]; setp.eq.u32 %p, %r, 0; @%p bra L; ret;')
        self.assertEqual(len(cfg.loops), 1)
        loop = cfg.loops[0]
        self.assertEqual(loop.blocks, frozenset({0}))
        self.assertEqual(loop.latches, frozenset({0}))
        self.assertEqual([(edge.source, edge.target) for edge in loop.entries], [(-1, 0)])
        self.assertEqual([(edge.kind, edge.predicate, edge.taken) for edge in loop.exits],
                         [('fallthrough', '%p', False)])

    def test_every_exit_including_predicated_return(self):
        cfg = flow('mov.u32 %r, 0; L: ld.acquire.sys.u32 %r, [%ptr]; '
                   '@!%p ret; @%q bra DONE; bra L; DONE: ret;')
        loop = cfg.loops[0]
        self.assertEqual(len(loop.entries), 1)
        self.assertEqual(len(loop.exits), 2)
        self.assertEqual({(edge.kind, edge.predicate, edge.taken) for edge in loop.exits},
                         {('ret', '!%p', True), ('branch', '%q', True)})

    def test_multiple_latches_share_one_header(self):
        cfg = flow('L: ld.relaxed.sys.u32 %r, [%ptr]; @%p bra A; '
                   '@%q bra L; ret; A: @%r bra L; ret;')
        self.assertEqual(len(cfg.loops), 1)
        self.assertEqual(len(cfg.loops[0].latches), 2)
        self.assertEqual(len(cfg.loops[0].exits), 2)

    def test_nested_natural_loops(self):
        cfg = flow('OUTER: mov.u32 %r, 0; INNER: ld.acquire.sys.u32 %r, [%ptr]; '
                   '@%p bra INNER; @%q bra OUTER; ret;')
        self.assertEqual(len(cfg.loops), 2)
        outer, inner = cfg.loops
        self.assertLess(inner.blocks, outer.blocks)
        self.assertEqual(len(inner.exits), 1)
        self.assertEqual(len(outer.exits), 1)

    def test_unreachable_code_and_acyclic_branch(self):
        cfg = flow('bra END; DEAD: membar.sys; bra DEAD; END: @%p ret; ret;')
        self.assertEqual(cfg.loops, ())
        self.assertNotIn(1, cfg.reachable)
        self.assertEqual(len(cfg.blocks), 4)

    def test_trap_is_exceptional_not_fallthrough_or_normal_return(self):
        cfg = flow('@%timeout trap; mov.u32 %r, 0; ret;')
        trap = [edge for edge in cfg.edges if edge.kind == 'trap']
        self.assertEqual(len(trap), 1)
        self.assertEqual((trap[0].target, trap[0].predicate, trap[0].taken),
                         (len(cfg.blocks), '%timeout', True))
        cfg = flow('trap; membar.sys; ret;')
        self.assertEqual(cfg.reachable, frozenset({0}))

    def test_sibling_labels_resolve_to_distinct_loops(self):
        cfg = flow('{ LAB_WAIT: ld.acquire.sys.u32 %r, [%ptr]; @%p bra LAB_WAIT; }'
                   '{ LAB_WAIT: ld.acquire.sys.u32 %r, [%ptr]; @%q bra LAB_WAIT; } ret;')
        self.assertEqual([loop.header for loop in cfg.loops], [0, 1])
        self.assertEqual([loop.blocks for loop in cfg.loops], [frozenset({0}), frozenset({1})])

    def test_local_label_shadows_ancestor(self):
        cfg = flow('L: mov.u32 %r, 0; { L: @%p bra L; } @%q bra L; ret;')
        self.assertEqual([(edge.source, edge.target) for edge in cfg.edges
                          if edge.kind == 'branch'], [(1, 1), (2, 0)])
        self.assertEqual(len(cfg.loops), 2)

    def test_nested_branch_to_ancestor_and_label_before_scope(self):
        cfg = flow('L: { mov.u32 %r, 0; { @%p bra L; } } ret;')
        self.assertEqual(len(cfg.loops), 1)
        self.assertEqual(cfg.loops[0].blocks, frozenset({0}))

    def test_vector_operands_do_not_open_lexical_scopes(self):
        cfg = flow('L: ld.global.v2.u32 {%r0, %r1}, [%ptr]; @%p bra L; ret;')
        self.assertEqual(cfg.loops[0].blocks, frozenset({0}))

    def test_fail_closed_on_irreducible_or_unresolved_control(self):
        cases = ('@%p bra B; A: bra B; B: @%q bra A; ret;',
                 'brx.idx %r, table;', 'bra missing;', 'brkpt;',
                 'L: mov.u32 %r, 0; L: ret;',
                 '{ L: mov.u32 %r, 0; } { bra L; }',
                 'bra L; { L: ret; }')
        for body in cases:
            with self.subTest(body=body), self.assertRaises(PTXError):
                flow(body)


if __name__ == '__main__':
    unittest.main()
