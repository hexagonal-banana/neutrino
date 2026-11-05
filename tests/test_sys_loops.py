"""Loop-edge emission, executed PTX paths, nested decoding, and timeline semantics."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neutrino.sys_sync.linkage import link_group
from neutrino.sys_sync.probe import instrument_instructions
from neutrino.sys_sync.ptx import PTXError, parse
from neutrino.sys_sync.records import BufferLayout, RECORD, STATE, RecordError, decode_native
from neutrino.sys_sync.export import export_trace, NativeProcess, TraceError
from test_sys_records import launch, raw
from test_sys_export import fixture, add_activity

HEADER = '.version 9.1\n.target sm_90\n.address_size 64\n'


def rewrite(body, storage='per-thread-v1', loop_depth=1):
    return instrument_instructions(link_group({'module': HEADER + '.visible .entry k() {\n' + body + '\n}'}),
                                   max_parameter_bytes=32764, interval_mode='loop-boundary', record_storage=storage,
                                   loop_depth=loop_depth)


def execute(result, *, enabled=True, capacity=64, initial=None, sampling=0):
    """Tiny fail-closed scalar PTX interpreter, only for these bounded fixtures.

    Executes the actual inserted predicates/branches and memory operations. It
    verifies timer/record counts without assuming that assembly implies behavior.
    GPU fixtures separately check the compiler and hardware execution paths.
    """
    function = parse(result.sources['module']).functions[0]
    regs = {'%tid.x': 0, '%tid.y': 0, '%tid.z': 0, '%ntid.x': 1, '%ntid.y': 1, '%ntid.z': 1,
            '%ctaid.x': 0, '%ctaid.y': 0, '%ctaid.z': 0, '%nctaid.x': 1, '%nctaid.y': 1, '%nctaid.z': 1}
    regs.update(initial or {})
    pooled = result.manifest['record_storage'] == 'pooled-v1'
    layout = BufferLayout.from_dimensions(tuple(regs['%nctaid.' + a] for a in 'xyz'),
                                          tuple(regs['%ntid.' + a] for a in 'xyz'), capacity,
                                          {0: 'all', 1: 'warp', 2: 'block'}.get(sampling, 'all'))
    memory = {100: 10000, 108: 1000, 124: layout.threads, 132: capacity, 136: 4 if pooled else 3, 140: 2000,
              148: sampling,
              1000: 0, 1004: 0, 1008: 0, 1012: 0, 2000: 0}
    params = {'__neutrino_sys_context': 100 if enabled else 0}
    labels = {(scope, label): inst.index for inst in function.instructions
              for scope, label in zip(inst.label_scopes, inst.labels)}
    pc, timers, steps = 0, [], 0

    def value(text):
        if text.startswith('!'):
            return not value(text[1:])
        if text == '%globaltimer':
            timers.append(steps)
            return steps
        if text in regs:
            return regs[text]
        return int(text, 0)

    def address(text):
        term = text.strip('[]')
        if term in params:
            return term
        parts = term.split('+')
        return sum(value(part) for part in parts)

    while pc < len(function.instructions):
        steps += 1
        if steps > 100000:
            raise AssertionError('Unbounded test PTX execution')
        inst = function.instructions[pc]
        pc += 1
        if inst.predicate and not value(inst.predicate):
            continue
        op, *qualifiers = inst.opcode.split('.')
        args = ''.join(token.text for token in inst.operands).split(',')
        if op in {'ret', 'exit'}:
            break
        if op == 'trap':
            raise AssertionError('Executed trap')
        if op == 'bra':
            pc = next(labels[(inst.scope[:depth], args[0])]
                      for depth in range(len(inst.scope), -1, -1)
                      if (inst.scope[:depth], args[0]) in labels)
        elif op in {'membar', 'fence'}:
            pass
        elif op in {'mov', 'cvt'}:
            regs[args[0]] = value(args[1])
        elif op == 'ld':
            addr = address(args[1])
            regs[args[0]] = params[addr] if 'param' in qualifiers else memory.get(addr, 0)
        elif op == 'st':
            memory[address(args[0])] = value(args[1])
        elif op == 'atom' and 'cas' in qualifiers:
            addr = address(args[1])
            old = memory.get(addr, 0)
            if old == value(args[2]):
                memory[addr] = value(args[3])
            regs[args[0]] = old
        elif op == 'setp':
            a, b = value(args[1]), value(args[2])
            regs[args[0]] = {'eq': a == b, 'ne': a != b, 'lt': a < b, 'ge': a >= b, 'gt': a > b}[qualifiers[0]]
        elif op in {'add', 'mul', 'mad', 'and', 'not', 'or', 'shr'}:
            values = [value(arg) for arg in args[1:]]
            if op == 'add':
                output = values[0] + values[1]
            elif op in {'mul', 'mad'}:
                output = values[0] * values[1] + (values[2] if op == 'mad' else 0)
            elif op == 'and':
                output = values[0] & values[1]
            elif op == 'or':
                output = values[0] | values[1]
            elif op == 'shr':
                output = values[0] >> values[1]
            else:
                output = not values[0]
            regs[args[0]] = output
        else:
            raise AssertionError('Unsupported fixture opcode: ' + inst.opcode)
    data = STATE.pack(*(memory.get(1000 + offset, 0) for offset in (0, 4, 8, 12)))
    stride = 40 if pooled else 32
    for index in range(memory[1000]):
        slot = 10000 + stride * index
        data += RECORD.pack(*(memory.get(slot + offset, 0) for offset in (0, 8, 16, 24, 28)))
    regs['__memory'] = memory
    return data, timers, regs


class LoopTests(unittest.TestCase):
    def decoded(self, result, **kwargs):
        data, timers, regs = execute(result, **kwargs)
        sites = result.manifest['sites']
        events = decode_native(data, launch=launch(), layout=BufferLayout(1, kwargs.get('capacity', 64)),
                               sites=[site['ordinal'] for site in sites], raw_format='states-packed-v1',
                               loop_sites=[site['ordinal'] for site in sites if site['interval_kind'] == 'loop'])
        self.assertEqual(len(timers), 2 * len(events))
        return events, regs

    def test_many_iterations_multiple_sites_one_interval_both_storage_modes(self):
        body = '.reg .u32 %i; .reg .pred %p; mov.u32 %i, 0;\nL: membar.sys; fence.acq_rel.sys;\n' \
               'add.u32 %i, %i, 1; setp.lt.u32 %p, %i, 99; @%p bra L; ret;'
        for storage in ('per-thread-v1', 'pooled-v1'):
            result = rewrite(body, storage)
            events, regs = self.decoded(result)
            self.assertEqual(regs['%i'], 99)
            self.assertEqual(len(events), 1)
            self.assertEqual(len(result.manifest['loops'][0]['contained_sites']), 2)
            self.assertEqual(len(parse(result.sources['module']).sites), 2)
            self.assertFalse(result.manifest['polling_intervals_validated'])

    def test_nested_loops_reenter_inner_and_preserve_outer_state(self):
        body = '''.reg .u32 %i, %j; .reg .pred %p;
mov.u32 %i, 0;
OUTER: membar.sys; mov.u32 %j, 0;
INNER: fence.acq_rel.sys; add.u32 %j, %j, 1; setp.lt.u32 %p, %j, 4; @%p bra INNER;
add.u32 %i, %i, 1; setp.lt.u32 %p, %i, 3; @%p bra OUTER; ret;'''
        for storage in ('per-thread-v1', 'pooled-v1'):
            result = rewrite(body, storage)
            events, _ = self.decoded(result)
            self.assertEqual(len(events), 4)
            self.assertTrue(all(events[0].begin <= event.begin <= event.end <= events[0].end for event in events))
            self.assertEqual([event.occurrence for event in events], [0, 0, 1, 2])

    def test_depth_selection_merge_clamp_and_nested_state(self):
        body = '''.reg .u32 %i, %j, %k; .reg .pred %p;
mov.u32 %i, 0;
O: membar.sys; mov.u32 %j, 0;
M: fence.acq_rel.sys; mov.u32 %k, 0;
I: membar.sys; add.u32 %k, %k, 1; setp.lt.u32 %p, %k, 4; @%p bra I;
add.u32 %j, %j, 1; setp.lt.u32 %p, %j, 3; @%p bra M;
add.u32 %i, %i, 1; setp.lt.u32 %p, %i, 2; @%p bra O;
fence.acq_rel.sys; ret;'''
        for storage in ('per-thread-v1', 'pooled-v1'):
            for depth, count, loops in ((1, 10, 3), (2, 4, 2), (3, 2, 1), (4, 2, 1), (99, 2, 1)):
                with self.subTest(storage=storage, depth=depth):
                    result = rewrite(body, storage, depth)
                    events, regs = self.decoded(result)
                    self.assertEqual((regs['%i'], regs['%j'], regs['%k']), (2, 3, 4))
                    self.assertEqual(len(events), count)
                    self.assertEqual(len(result.manifest['loops']), loops)
                    self.assertEqual(result.manifest['loop_depth'], depth)
                    self.assertEqual(sum(len(l['contained_sites']) for l in result.manifest['loops']), 3)
                    for loop in result.manifest['loops']:
                        for site in loop['contained_sites']:
                            self.assertEqual(site['selected_loop_depth'], min(depth, site['available_loop_depth']))
                    self.assertTrue(all(events[0].begin <= e.begin <= e.end <= events[0].end for e in events[:-1]))
                    self.assertLess(events[0].end, events[-1].begin)
                    self.assertEqual(execute(result, enabled=False)[1], [])
        self.assertEqual(rewrite(body).sources, rewrite(body, loop_depth=1).sources)
        self.assertNotEqual(rewrite(body).manifest['loops'][0]['id'], rewrite(body, loop_depth=2).manifest['loops'][0]['id'])

    def test_all_selected_families_share_depth_three_loop(self):
        body = '''.reg .u32 %i, %j, %k, %v; .reg .pred %p;
mov.u32 %i, 0;
O: mov.u32 %j, 0;
M: mov.u32 %k, 0;
I: ld.acquire.sys.global.u32 %v, [%ptr];
fence.sc.sys; membar.sys;
fence.sc.gpu; membar.cta; bar.sync 0; barrier.sync 0;
mbarrier.test_wait.shared.b64 %p, [%ptr], %state;
mbarrier.try_wait.shared.b64 %p, [%ptr], %state;
mbarrier.arrive.shared.b64 %state, [%ptr];
st.release.sys.global.u32 [%ptr], %v;
ld.relaxed.sys.global.u32 %v, [%ptr];
add.u32 %k, %k, 1; setp.lt.u32 %p, %k, 4; @%p bra I;
add.u32 %j, %j, 1; setp.lt.u32 %p, %j, 3; @%p bra M;
add.u32 %i, %i, 1; setp.lt.u32 %p, %i, 2; @%p bra O;
membar.sys; ret;'''
        for storage in ('per-thread-v1', 'pooled-v1'):
            result = rewrite(body, storage, 3)
            self.assertEqual(len(result.manifest['loops']), 1)
            loop = result.manifest['loops'][0]
            self.assertEqual(len(loop['contained_sites']), 4)
            self.assertEqual({s['selected_loop_depth'] for s in loop['contained_sites']}, {3})
            self.assertEqual({s['operation'] for s in loop['contained_sites']},
                             {'ld', 'fence', 'membar', 'mbarrier'})
            self.assertEqual(len(result.manifest['sites']), 2)
            self.assertEqual(result.manifest['sites'][0]['opcode'], 'membar.sys')
            self.assertEqual(result.manifest['sampling'], 'launch_policy')
            source = HEADER + '.visible .entry k() {\n' + body + '\n}'
            before = parse(source).functions[0].instructions
            after = parse(result.sources['module']).functions[0].instructions
            signature = lambda ins: (ins.opcode, ins.predicate, tuple(t.text for t in ins.operands))
            originals = [signature(ins) for ins in before]
            retained = [signature(ins) for ins in after if signature(ins) in originals]
            self.assertEqual(retained, originals)

    def test_depth_validation(self):
        for depth in (0, -1, True, False, 1.5, '2', None, 2147483648):
            with self.assertRaisesRegex(PTXError, 'Loop depth'):
                rewrite('ret;', loop_depth=depth)
        with self.assertRaisesRegex(PTXError, 'requires loop-boundary'):
            instrument_instructions(link_group({'module': HEADER + '.entry k() { ret; }'}),
                                    max_parameter_bytes=32764, interval_mode='instruction-only', loop_depth=2)

    def test_depth_two_skipped_loop_and_early_exit(self):
        body = '''.reg .u32 %i, %j; .reg .pred %p, %skip, %early;
@%skip bra END; mov.u32 %i, 0;
O: mov.u32 %j, 0;
I: membar.sys; @%early bra END;
add.u32 %j, %j, 1; setp.lt.u32 %p, %j, 4; @%p bra I;
add.u32 %i, %i, 1; setp.lt.u32 %p, %i, 3; @%p bra O;
END: ret;'''
        for storage in ('per-thread-v1', 'pooled-v1'):
            result = rewrite(body, storage, 2)
            for skip, early in ((0, 0), (0, 1), (1, 0)):
                events, _ = self.decoded(result, initial={'%skip': skip, '%early': early})
                self.assertEqual(len(events), 0 if skip else 1)

    def test_function_entry_header_and_predicated_early_return(self):
        result = rewrite('''.reg .u32 %i; .reg .pred %p;
L: membar.sys; add.u32 %i, %i, 1; setp.eq.u32 %p, %i, 3; @%p ret; bra L;''')
        events, regs = self.decoded(result, initial={'%i': 0})
        self.assertEqual(regs['%i'], 3)
        self.assertEqual(len(events), 1)
        self.assertEqual(result.manifest['sites'][0]['exit_edge']['kind'], 'ret')

    def test_zero_entry_and_negated_branch_multiple_entries_and_exits(self):
        body = '''.reg .u32 %i; .reg .pred %p, %skip, %short, %choice;
@%skip bra END;
@%choice bra L;
mov.u32 %i, 0;
L: membar.sys; @%short bra END;
add.u32 %i, %i, 1; setp.ge.u32 %p, %i, 5; @!%p bra L;
END: ret;'''
        for storage in ('per-thread-v1', 'pooled-v1'):
            result = rewrite(body, storage)
            for skip, short, choice in ((0, 0, 0), (0, 1, 1), (1, 0, 0), (0, 0, 1)):
                events, _ = self.decoded(result, initial={'%skip': skip, '%short': short, '%choice': choice, '%i': 0})
                self.assertEqual(len(events), 0 if skip else 1)
                if events:
                    site = next(site for site in result.manifest['sites'] if site['ordinal'] == events[0].site)
                    self.assertEqual(site['exit_edge']['kind'], 'branch' if short else 'fallthrough')
            self.assertEqual(len(result.manifest['loops'][0]['entries']), 2)

    def test_false_sys_predicate_still_measures_loop_not_instruction(self):
        result = rewrite('''.reg .u32 %i; .reg .pred %p, %false;
mov.u32 %i, 0; mov.pred %false, 0;
L: @%false membar.sys; add.u32 %i, %i, 1; setp.lt.u32 %p, %i, 7; @%p bra L; ret;''')
        events, _ = self.decoded(result)
        self.assertEqual(len(events), 1)
        self.assertFalse(result.manifest['dynamic_sys_predicate_required'])

    def test_outside_instruction_adjacent_to_loop_entry(self):
        result = rewrite('''.reg .u32 %i; .reg .pred %p; mov.u32 %i, 0;
membar.sys;
L: membar.sys; add.u32 %i, %i, 1; setp.lt.u32 %p, %i, 8; @%p bra L;
fence.acq_rel.sys; ret;''')
        events, _ = self.decoded(result)
        self.assertEqual(len(events), 3)
        self.assertLess(events[0].end, events[1].begin)
        self.assertLess(events[1].end, events[2].begin)

    def test_null_context_and_overflow_remain_strict(self):
        result = rewrite('membar.sys; .reg .pred %p; L: membar.sys; @%p bra L; ret;')
        data, timers, _ = execute(result, enabled=False, initial={'%p': False})
        self.assertEqual(timers, [])
        self.assertEqual(data, bytes(16))
        with self.assertRaisesRegex(RecordError, 'overflow'):
            self.decoded(result, capacity=1, initial={'%p': False})

    def test_exceptional_exit_metadata_and_unsupported_cycles_fail_closed(self):
        result = rewrite('.reg .pred %p; L: membar.sys; @%p trap; bra L;')
        self.assertEqual(result.manifest['sites'][0]['exit_outcome'], 'exceptional')
        for body in ('L: membar.sys; bra L;',
                     '.reg .pred %p; @%p bra B; A: membar.sys; bra B; B: @%p bra A; ret;',
                     'membar.sys; brx.idx %r, targets; ret;'):
            with self.assertRaises(PTXError):
                rewrite(body)

    def test_nested_decoder_rejects_crossing_and_nonloop_containment(self):
        identity, layout = launch(), BufferLayout(1, 4)
        cases = [[(10, 30, 1), (20, 40, 2)], [(10, 30, 2), (20, 25, 2)],
                 [(10, 30, 1), (9, 20, 2)]]
        for rows in cases:
            data = raw(identity, layout, [rows])[64:]
            with self.assertRaises(RecordError):
                decode_native(data, launch=identity, layout=layout, sites=[1, 2],
                              raw_format='states-fixed-v1', loop_sites=[1])

    @unittest.skipUnless(os.environ.get('UNIFIED_TOOLCHAIN'), 'Selected CUDA assembler required')
    def test_nested_predicated_edge_assembly_both_storages(self):
        body = '''.reg .u32 %i, %j; .reg .pred %p; mov.u32 %i, 0;
O: membar.sys; mov.u32 %j, 0;
I: fence.acq_rel.sys; add.u32 %j, %j, 1; setp.lt.u32 %p, %j, 9; @%p bra I;
add.u32 %i, %i, 1; setp.lt.u32 %p, %i, 3; @%p bra O; @!%p ret; ret;'''
        with tempfile.TemporaryDirectory() as temporary:
            for storage, depth in ((s, d) for s in ('per-thread-v1', 'pooled-v1') for d in (1, 2)):
                ptx = Path(temporary) / (storage + str(depth) + '.ptx')
                ptx.write_text(rewrite(body, storage, loop_depth=depth).sources['module'])
                subprocess.run([os.environ['UNIFIED_TOOLCHAIN'] + '/bin/ptxas', '-arch=sm_90', str(ptx),
                                '-o', str(ptx.with_suffix('.cubin'))], check=True, capture_output=True,
                               env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'LD_PRELOAD': ''})

    def test_export_per_library_depths_and_legacy_depth_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for rank, depth in enumerate((1, 2)):
                journal, rows = fixture(root, rank)
                result = rewrite('.reg .pred %p; L: membar.sys; @%p bra L; ret;', loop_depth=depth)
                image_path = next(journal.parent.glob('*.image.json'))
                image = json.loads(image_path.read_text())
                legacy_probe = dict(result.manifest, sampling='all_threads')
                legacy_probe.pop('context_abi')
                image.update(interval_mode='loop-boundary', loop_depth=depth, probe=legacy_probe, entries={'k': {}})
                if depth == 1:
                    image.pop('loop_depth')
                    image['probe'].pop('loop_depth')
                    image['probe'].pop('depth_fallback')
                    for loop in image['probe']['loops']:
                        loop.pop('loop_depth')
                        loop.pop('depth_fallback')
                image_path.write_text(json.dumps(image))
                ordinal = result.manifest['sites'][0]['ordinal']
                begin = 2**60 + rank * 999999999 + 400000
                data = STATE.pack(1, 0, 1, 0) + STATE.pack(0, 0, 1, 0) + RECORD.pack(begin, begin + 100, ordinal, 0, 2)
                (journal.parent / 'launch-1.bin').write_bytes(data)
                next(row for row in rows if row['kind'] == 'launch')['kernel'] = 'k'
                add_activity(journal, rows)
                process = NativeProcess(journal)
                self.assertEqual(len(process.decode_launch(process.launches[0])[3]), 1)
            report = export_trace(root, root / 'mixed', require_kernel_activity=True,
                                  instruction_granularity='captured')
            self.assertIsNone(report['loop_depth'])
            self.assertEqual(report['loop_depths'], {'NCCL': 1, 'DeepEP': 2})
            self.assertEqual(report['instruction_export']['loop_selection'], 'per_library')
            checked = validate_loop_export(root / 'mixed')
            self.assertEqual(checked['loop_depths'], report['loop_depths'])
            image['label'] = 'NCCL'
            image_path.write_text(json.dumps(image))
            with self.assertRaisesRegex(TraceError, 'Mixed loop depths within library'):
                export_trace(root, root / 'conflicting', require_kernel_activity=True)

    def test_loop_export_has_distinct_categories_original_sites_and_kernel(self):
        result = rewrite('.reg .pred %p; L: membar.sys; @%p bra L; ret;', loop_depth=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal, rows = fixture(root, 0)
            image_path = next(journal.parent.glob('*.image.json'))
            image = json.loads(image_path.read_text())
            legacy_probe = dict(result.manifest, sampling='all_threads')
            legacy_probe.pop('context_abi')
            image.update(interval_mode='loop-boundary', loop_depth=2, probe=legacy_probe, entries={'k': {}})
            image_path.write_text(json.dumps(image))
            ordinal = result.manifest['sites'][0]['ordinal']
            data = STATE.pack(1, 0, 1, 0) + STATE.pack(0, 0, 1, 0)
            data += RECORD.pack(2**60 + 400000, 2**60 + 400100, ordinal, 0, 2)
            (journal.parent / 'launch-1.bin').write_bytes(data)
            next(row for row in rows if row['kind'] == 'launch')['kernel'] = 'k'
            add_activity(journal, rows)
            report = export_trace(root, root / 'export', require_kernel_activity=True, instruction_granularity='warp')
            self.assertEqual(report['interval_mode'], 'loop-boundary')
            self.assertEqual(report['loop_depth'], 2)
            self.assertEqual(report['counts']['NCCL_loop_intervals'], 1)
            trace = json.loads((root / 'export/timeline.json').read_text())
            event = next(event for event in trace['traceEvents'] if event.get('cat') == 'NCCL.sys_loop')
            self.assertEqual(event['dur'], .1)
            self.assertEqual(event['args']['contained_sys_sites'][0]['opcode'], 'membar.sys')
            self.assertEqual(event['args']['synchronization_success'], 'unclassified')
            self.assertTrue(trace['metadata']['loop_inventory'])
            overview = json.loads((root / 'export/communication.json').read_text())
            self.assertFalse(any(event.get('cat', '').endswith('.sys_loop') for event in overview['traceEvents']))
            checked = validate_loop_export(root / 'export')
            self.assertEqual(checked['exported_intervals'], 1)
            self.assertEqual(checked['categories'], {'NCCL.sys_loop': 1})
            export_trace(root, root / 'comparison', require_kernel_activity=True, instruction_granularity='warp')
            comparison = validate_loop_export(root / 'comparison', root / 'export/loop-export-validation.json')
            self.assertEqual(comparison['exported_record_reduction_percent'], 0)
            self.assertEqual(list(stream_timeline(root / 'export/timeline.json', chunk_size=7))[-1][0], 'metadata')
            broken = root / 'broken.json'
            broken.write_text('{"traceEvents":[{},],"metadata":{}}')
            with self.assertRaises(json.JSONDecodeError):
                list(stream_timeline(broken, chunk_size=3))
            for field, value in (('loop_depth', 1), ('loop_depth', True), ('loop_depth', 0)):
                invalid = dict(image, **{field: value})
                image_path.write_text(json.dumps(invalid))
                process = NativeProcess(journal)
                with self.assertRaises(TraceError):
                    process.decode_launch(process.launches[0])
            image['probe']['sites'][0]['loop_id'] = 'missing'
            image_path.write_text(json.dumps(image))
            process = NativeProcess(journal)
            with self.assertRaisesRegex(TraceError, 'loop exit'):
                process.decode_launch(process.launches[0])


def stream_timeline(path, chunk_size=1024 * 1024):
    """Parse the exporter's JSON incrementally, including its complete footer."""
    decoder = json.JSONDecoder()
    with Path(path).open() as stream:
        prefix = '{"traceEvents":['
        if stream.read(len(prefix)) != prefix:
            raise ValueError('Unexpected timeline prefix')
        buffer, position, eof = '', 0, False

        def refill():
            nonlocal buffer, position, eof
            chunk = stream.read(chunk_size)
            buffer, position, eof = buffer[position:] + chunk, 0, not chunk

        def peek():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    return buffer[position]
                if eof:
                    raise ValueError('Truncated timeline')
                refill()

        if peek() == ']':
            position += 1
        else:
            while True:
                peek()
                try:
                    event, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()
                    continue
                if not isinstance(event, dict):
                    raise ValueError('Expected event object')
                yield 'event', event
                position = end
                delimiter = peek()
                position += 1
                if delimiter == ']':
                    break
                if delimiter != ',':
                    raise ValueError('Invalid event separator')
        footer = json.loads('{"traceEvents":[]' + buffer[position:] + stream.read())
        yield 'metadata', footer['metadata']


def validate_loop_export(directory, reference=None):
    """Check the delivered JSON against CSV, activity evidence, and gzip bytes."""
    import csv
    import gzip
    import hashlib
    from collections import Counter
    from neutrino.sys_sync.export import kernel_track_label, warp_coordinates

    directory = Path(directory)
    validation = json.loads((directory / 'validation.json').read_text())
    overview = json.loads((directory / 'communication.json').read_text())
    assert validation['full_trace_acceptance'] and validation['interval_mode'] == 'loop-boundary'
    assert validation['instruction_export']['instruction_granularity'] in {'warp', 'captured'}
    assert validation.get('lane_layout') in {None, 'kernel-qualified-v1', 'kernel-ctx-stream-v1', 'kernel-rank-ctx-stream-v1'}
    expected_kernels = {event['args']['launch_id']: event for event in overview['traceEvents']
                        if event.get('cat', '').endswith('.kernel')}
    coverage = json.loads((directory / 'coverage.json').read_text())
    launches = {row['launch_id']: row for row in coverage if row.get('launch_id')}
    origin = overview['metadata']['origin_host_ns']
    categories, libraries, observed_exits, kernel_ids = Counter(), Counter(), {}, set()
    metadata, tracks = None, {}
    with (directory / 'events.csv').open() as csv_stream:
        rows = csv.DictReader(csv_stream)
        for kind, event in stream_timeline(directory / 'timeline.json'):
            if kind == 'metadata':
                metadata = event
                continue
            assert type(event['pid']) is int and type(event['tid']) is int
            if event.get('name') == 'thread_name':
                tracks[event['pid'], event['tid']] = event['args']['name']
            category = event.get('cat', '')
            if category.endswith('.kernel'):
                key = event['args']['launch_id']
                assert key not in kernel_ids
                kernel_ids.add(key)
                expected = expected_kernels[key]
                assert all(event[field] == expected[field] for field in ('pid', 'ts', 'dur', 'args'))
                if validation.get('lane_layout') == 'kernel-qualified-v1':
                    args = event['args']
                    label = kernel_track_label(args['library'], args.get('original_kernel', args['kernel']), args['image'])
                    assert tracks[event['pid'], event['tid']].startswith(label + ' | ')
                elif validation.get('lane_layout') in {'kernel-ctx-stream-v1', 'kernel-rank-ctx-stream-v1'}:
                    row = launches[key]
                    label = kernel_track_label(row['library'], row['kernel'], row['image'])
                    prefix = (f'rank {row["rank"]}' if validation['lane_layout'] == 'kernel-rank-ctx-stream-v1'
                              else f'{row["library"]}/{label}')
                    expected_track = f'{prefix}/ctx{row["context"]}/stream{row["stream"]}'
                    assert tracks[event['pid'], event['tid']] == expected_track
                continue
            if not category.endswith(('.sys_loop', '.sys_instruction')):
                continue
            row = next(rows)
            args = event['args']
            assert all(str(args[key]) == value for key, value in row.items())
            assert event['ts'] == (args['begin_host_ns'] - origin) / 1000
            assert event['dur'] == (args['end_host_ns'] - args['begin_host_ns']) / 1000
            assert event['dur'] >= 0 and int(args['begin_tick']) <= int(args['end_tick'])
            launch_row = launches[args['launch_id']]
            block, warp, lane = warp_coordinates(args['thread'], launch_row['block'])
            granularity = launch_row['instruction_export']['granularity']
            prefix = args['library']
            if validation.get('lane_layout') == 'kernel-rank-ctx-stream-v1':
                prefix = f'rank {args["rank"]}'
            elif validation.get('lane_layout') is not None:
                prefix += '/' + kernel_track_label(args['library'], args['kernel'], args['image'])
            track = f'{prefix}/ctx{args["context"]}/stream{args["stream"]}'
            if validation.get('lane_layout') not in {'kernel-ctx-stream-v1', 'kernel-rank-ctx-stream-v1'}:
                if granularity in {'warp', 'block'}:
                    track += f'/block{block}'
                    if granularity == 'warp':
                        track += f'/warp{warp}'
                else:
                    track += f'/thread{args["thread"]}'
            assert tracks[event['pid'], event['tid']] == track
            if 'capture_sampling' in launch_row:
                assert args['capture_sampling'] == launch_row['capture_sampling']
                assert args['display_granularity'] == granularity
                if args['capture_sampling'] == 'warp':
                    assert lane == 0
                elif args['capture_sampling'] == 'block':
                    assert warp == lane == 0
            categories[category] += 1
            libraries[args['library']] += 1
            if category.endswith('.sys_loop'):
                assert args['opcode'] == 'sys.loop' and args['interval_kind'] == 'loop'
                assert args['synchronization_success'] == 'unclassified'
                assert args.get('loop_depth', 1) == validation.get('loop_depths', {}).get(
                    args['library'], validation.get('loop_depth', 1))
                assert args['contained_sys_sites']
                key = (args['loop_id'], args['exit_index'])
                value = (args['exit_edge'], args['contained_sys_sites'])
                assert observed_exits.setdefault(key, value) == value
        assert next(rows, None) is None
    assert metadata['session'] == overview['metadata']['session']
    assert metadata.get('lane_layout') == validation.get('lane_layout')
    assert metadata['interval_mode'] == 'loop-boundary' and kernel_ids == set(expected_kernels)
    assert metadata.get('loop_depth', 1) == validation.get('loop_depth', 1)
    assert metadata.get('loop_depths', {}) == validation.get('loop_depths', {})
    for (loop_id, index), (edge, contained) in observed_exits.items():
        loop = metadata['loop_inventory'][loop_id]
        assert loop['exits'][index] == edge
        assert contained == [{key: site[key] for key in ('opcode', 'line', 'predicate')}
                             for site in loop['contained_sites']]
    assert metadata['instruction_export'] == validation['instruction_export']
    if validation['instruction_export']['instruction_granularity'] == 'captured':
        assert not metadata['instruction_export']['second_offline_sampling']
        assert sum(libraries.values()) == sum(row['intervals'] for row in launches.values())
        assert all(row['instruction_export']['input_records'] == row['instruction_export']['exported_records']
                   for row in launches.values())
    for library, count in libraries.items():
        assert count == validation['counts'][library + '_intervals']
        for kind in ('loop', 'instruction'):
            assert categories[library + '.sys_' + kind] == validation['counts'].get(library + '_' + kind + '_intervals', 0)
    with (directory / 'timeline.json').open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    with gzip.open(directory / 'timeline.json.gz', 'rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == checksum
    result = {'status': 'passed', 'interval_mode': 'loop-boundary',
              'loop_depth': validation.get('loop_depth', 1), 'loop_depths': validation.get('loop_depths', {}),
              'exported_intervals': sum(libraries.values()), 'categories': dict(categories),
              'raw_interval_records': sum(row['intervals'] for row in launches.values()),
              'kernel_spans': len(kernel_ids), 'json_csv_timestamps_match': True,
              'gzip_roundtrip_sha256': checksum,
              'files': {name: (directory / name).stat().st_size
                        for name in ('timeline.json', 'timeline.json.gz', 'events.csv')},
              'semantics': 'Configured-depth structural loop entry/exit; includes selected threads with inactive sys predicates. '
                           'Outside-loop instructions retain brackets. Observations are not whole-warp/block completion.',
              'instruction_export': validation['instruction_export']}
    if reference is not None:
        previous = json.loads(Path(reference).read_text())
        result['reference'] = str(reference)
        result['fresh_execution_comparison'] = True
        previous_files = {name: value['warp_bytes'] if isinstance(value, dict) else value
                          for name, value in previous['files'].items()}
        result['file_reduction_percent'] = {
            name: (1 - size / previous_files[name]) * 100 for name, size in result['files'].items()}
        previous_count = (previous['exported_intervals'] if 'exported_intervals' in previous else
                          previous['instruction_records']['warp'])
        result['exported_record_reduction_percent'] = (1 - result['exported_intervals'] / previous_count) * 100
    with (directory / 'loop-export-validation.json').open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    return result


def compare_kernel_lane_export(reference, directory):
    """Verify a presentation-only re-export preserves every event and raw evidence."""
    from collections import Counter
    import hashlib
    from itertools import zip_longest
    from neutrino.sys_sync.export import kernel_track_label

    reference, directory = Path(reference), Path(directory)
    read = lambda root, name: json.loads((root / name).read_text())
    before, after = read(reference, 'validation.json'), read(directory, 'validation.json')
    layout, previous_layout = after.pop('lane_layout'), before.pop('lane_layout', None)
    layouts = {'kernel-qualified-v1', 'kernel-ctx-stream-v1', 'kernel-rank-ctx-stream-v1'}
    assert layout in layouts and previous_layout in layouts | {None}
    assert before == after
    digests = {}
    for name in ('events.csv', 'coverage.json', 'clocks.json'):
        with (reference / name).open('rb') as stream:
            expected = hashlib.file_digest(stream, 'sha256').hexdigest()
        with (directory / name).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert actual == expected, name
        digests[name] = actual

    launches = {row['launch_id']: row for row in read(directory, 'coverage.json') if row.get('launch_id')}
    tracks, owners, groups, categories = {}, {}, {}, Counter()
    display_groups, lane_metadata, process_sorts, thread_sorts = {}, {}, {}, {}
    native_pids = {row['pid'] for row in launches.values()}

    def events(root, labeled):
        cross_rank = (layout if labeled else previous_layout) == 'kernel-rank-ctx-stream-v1'
        local_groups, local_lanes, group_identities = {}, {}, {}
        for kind, event in stream_timeline(root / 'timeline.json'):
            if kind == 'metadata':
                assert event.pop('lane_layout', None) == (layout if labeled else previous_layout)
                yield kind, event
                continue
            key = event['pid'], event['tid']
            if event.get('ph') == 'M':
                name, args = event['name'], event['args']
                if name == 'thread_name':
                    assert key not in local_lanes
                    local_lanes[key] = args
                    if labeled:
                        tracks[key] = args['name']
                        lane_metadata[key] = args
                    continue
                if cross_rank and name == 'process_name' and args.get('display_group') == 'kernel':
                    assert event['pid'] not in native_pids | local_groups.keys()
                    identity = args['library'], args['image'], args['kernel']
                    assert identity not in group_identities
                    group_identities[identity] = event['pid']
                    assert args['name'] == args['library'] + '/' + kernel_track_label(
                        args['library'], args['kernel'], args['image'])
                    local_groups[event['pid']] = args
                    if labeled:
                        display_groups[event['pid']] = args
                    continue
                if cross_rank and name in {'process_sort_index', 'thread_sort_index'}:
                    if labeled:
                        sorts = process_sorts if name == 'process_sort_index' else thread_sorts
                        sort_key = event['pid'] if name == 'process_sort_index' else key
                        assert sort_key not in sorts
                        sorts[sort_key] = args['sort_index']
                    continue
            category = event.get('cat', '')
            if category.endswith(('.kernel', '.sys_instruction', '.sys_loop')):
                args = event['args']
                row = launches[args['launch_id']]
                kernel = args.get('original_kernel', args['kernel'])
                label = kernel_track_label(args['library'], kernel, args['image'])
                if cross_rank:
                    group = local_groups[event['pid']]
                    assert (group['library'], group['image'], group['kernel']) == (args['library'], args['image'], kernel)
                    expected_lane = dict(source_pid=row['pid'], rank=row['rank'], context=row['context'], stream=row['stream'])
                    assert all(local_lanes[key][field] == value for field, value in expected_lane.items())
                    assert local_lanes[key]['name'] == f'rank {row["rank"]}/ctx{row["context"]}/stream{row["stream"]}'
                    event = dict(event, pid=row['pid'])
                if labeled:
                    owner = args['library'], args['image'], kernel
                    if layout in {'kernel-ctx-stream-v1', 'kernel-rank-ctx-stream-v1'}:
                        owner += (row['pid'], row['context'], row['stream'])
                        if not cross_rank:
                            assert tracks[key] == f'{args["library"]}/{label}/ctx{row["context"]}/stream{row["stream"]}'
                        assert groups.setdefault(owner, key) == key, 'Kernel/rank/context/stream split across lanes'
                    assert owners.setdefault(key, owner) == owner, 'Mixed kernel/rank/context/stream identities on one lane'
                    if not cross_rank:
                        assert label in tracks[key]
                    categories[category] += 1
            yield kind, {key: value for key, value in event.items() if key != 'tid'}

    for old, new in zip_longest(events(reference, False), events(directory, True)):
        assert old == new, 'Non-lane event data or metadata changed'
    if layout == 'kernel-rank-ctx-stream-v1':
        ordered_groups = sorted(display_groups, key=lambda pid: tuple(display_groups[pid][field]
                                                                    for field in ('library', 'kernel', 'image')))
        assert [process_sorts[pid] for pid in ordered_groups] == list(range(len(ordered_groups)))
        assert set(thread_sorts) == set(owners)
        for pid in ordered_groups:
            ordered_lanes = sorted((key for key in owners if key[0] == pid), key=lambda key: tuple(
                lane_metadata[key][field] for field in ('rank', 'context', 'stream', 'source_pid')))
            assert [thread_sorts[key] for key in ordered_lanes] == list(range(len(ordered_lanes)))
        assert all(process_sorts[pid] >= len(ordered_groups) for pid in native_pids)
    result = {'status': 'passed', 'reference': str(reference), 'directory': str(directory),
              'lane_layout': layout, 'gpu_lanes': len(owners),
              'kernel_context_stream_groups': len(groups) if layout != 'kernel-qualified-v1' else None,
              'kernel_display_groups': len(display_groups),
              'ranks_by_kernel_group': {str(pid): sorted({lane_metadata[key]['rank'] for key in owners if key[0] == pid})
                                       for pid in display_groups},
              'categories': dict(categories), 'events_and_timestamps_unchanged': True,
              'one_kernel_identity_per_gpu_lane': True, 'unchanged_file_sha256': digests,
              'gpu_recapture': False}
    with (directory / 'kernel-lane-validation.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return result


def compare_source_export(reference, directory, raw_root):
    """Allow only verified source labels/details; compare all lanes and times exactly."""
    from collections import Counter
    import gzip
    import hashlib
    from itertools import zip_longest
    from neutrino.sys_sync.export import InstructionSources

    reference, directory, raw_root = map(Path, (reference, directory, raw_root))
    read = lambda root, name: json.loads((root / name).read_text())
    before, after = read(reference, 'validation.json'), read(directory, 'validation.json')
    report = after.pop('source_annotation')
    assert before == after, 'Capture/export configuration changed'
    source_map = read(directory, 'source-locations.json')
    assert source_map['annotation'] == report
    inventory = source_map['sources']
    resolver = InstructionSources(sorted({row['ptx_file'] for row in inventory.values()}))
    digests = {}
    for name in ('events.csv', 'coverage.json', 'clocks.json', 'communication.json'):
        with (reference / name).open('rb') as stream:
            expected = hashlib.file_digest(stream, 'sha256').hexdigest()
        with (directory / name).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert actual == expected, name
        digests[name] = actual
    launches = {row['launch_id']: row for row in read(directory, 'coverage.json') if row.get('launch_id')}
    images, members_by_site, categories = {}, {}, Counter()
    durations = 0
    for old, new in zip_longest(stream_timeline(reference / 'timeline.json'),
                                stream_timeline(directory / 'timeline.json')):
        assert old is not None and new is not None, 'Event count changed'
        kind, original = old
        new_kind, event = new
        assert kind == new_kind
        if kind == 'metadata':
            assert event.pop('source_annotation') == report
            assert event.pop('source_inventory') == inventory
        elif event.get('cat', '').endswith(('.sys_loop', '.sys_instruction')):
            args = event['args']
            image_id = args['image']
            if image_id not in images:
                launch = launches[args['launch_id']]
                path = raw_root / Path(launch['raw']).parent / (image_id + '.image.json')
                image = json.loads(path.read_text())
                assert image['id'] == image_id
                images[image_id] = image
                loops = {loop['id']: loop for loop in image['probe'].get('loops', [])}
                for site in image['probe']['sites']:
                    members = (loops[site['loop_id']]['contained_sites']
                               if site.get('interval_kind') == 'loop' else [site])
                    members_by_site[image_id, site['id']] = site, members
            image = images[image_id]
            assert image['label'] == args['library']
            site, members = members_by_site[image_id, args['site_id']]
            assert site['ordinal'] == args['site'] and site['function'] == args['callee']
            expected_name, annotation = resolver.annotate(image, site, members)
            assert event['name'] == expected_name, 'Incorrect instruction/source label'
            assert args.pop('source_annotation') == annotation, 'Incorrect source annotation'
            event['name'] = original['name']
            categories[event['cat']] += 1
        assert original == event, 'Original event, lane, timestamp, duration, or argument changed'
        if kind == 'event' and event.get('ph') == 'X':
            durations += 1
    assert resolver.inventory == inventory, 'Source inventory differs from hash-verified original PTX'
    assert resolver.report() == report
    with (directory / 'timeline.json').open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    with gzip.open(directory / 'timeline.json.gz', 'rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
    result = {'status': 'passed', 'reference': str(reference), 'directory': str(directory),
              'lane_layout': after['lane_layout'], 'duration_events': durations,
              'annotated_intervals': sum(categories.values()), 'categories': dict(categories),
              'source_annotation': report, 'timing_lanes_order_and_original_args_unchanged': True,
              'unchanged_file_sha256': digests, 'gzip_roundtrip_sha256': digest,
              'gpu_recapture': False,
              'files': {name: (directory / name).stat().st_size for name in
                        ('timeline.json', 'timeline.json.gz', 'source-locations.json')}}
    with (directory / 'source-export-validation.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return result


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--compare-source-export':
        print(json.dumps(compare_source_export(*sys.argv[2:]), indent=2))
    elif len(sys.argv) == 4 and sys.argv[1] == '--compare-kernel-lanes':
        print(json.dumps(compare_kernel_lane_export(*sys.argv[2:]), indent=2))
    elif len(sys.argv) in (3, 4) and sys.argv[1] == '--validate-export':
        print(json.dumps(validate_loop_export(*sys.argv[2:]), indent=2))
    else:
        unittest.main()
