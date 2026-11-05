"""Fresh byte fixtures for strict per-launch, per-thread decoding and budgets."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neutrino.sys_sync.identity import (ArtifactIdentity, ContextIdentity, KernelIdentity,
                                        LaunchIdentity, ModuleIdentity, ProcessIdentity)
from neutrino.sys_sync.records import (BufferLayout, COMMITTED, CONTEXT, HEADER, RECORD,
                                       STATE, RecordError, decode, decode_native, encode_header)


def launch():
    process = ProcessIdentity('session', 17, 'unique-process', 0)
    context = ContextIdentity(process, 1, 'GPU-example')
    module = ModuleIdentity(context, 1, ArtifactIdentity('1' * 64, 'sm_90', 'cubin'))
    return LaunchIdentity(KernelIdentity(module, 'entry'), 1, 0)


def raw(identity, layout, rows):
    data = bytearray(encode_header(identity, layout) + bytes(layout.state_bytes + layout.record_bytes))
    for thread, records in enumerate(rows):
        STATE.pack_into(data, HEADER.size + thread * STATE.size, len(records), 0, 1, 0)
        for ordinal, (begin, end, site) in enumerate(records):
            offset = HEADER.size + layout.state_bytes + (thread * layout.capacity + ordinal) * RECORD.size
            RECORD.pack_into(data, offset, begin, end, site, ordinal, COMMITTED)
    return data


class RecordTests(unittest.TestCase):
    def test_sizes_and_default_bounded_budget(self):
        self.assertEqual((CONTEXT.size, STATE.size, RECORD.size, HEADER.size), (56, 16, 32, 64))
        layout = BufferLayout.from_dimensions((2, 3, 4), (4, 2, 2))
        self.assertEqual(layout.threads, 384)
        self.assertEqual(layout.capacity, 256)
        layout.require_budget()
        with self.assertRaisesRegex(RecordError, 'budget'):
            layout.require_budget(retained_bytes=1, budget=layout.device_bytes)
        layout.require_budget(budget=layout.device_bytes)

    def test_invalid_dimensions_capacities_and_integer_overflow(self):
        for threads, capacity in [(0, 1), (1, 0), (True, 1), (1, True), (2**63, 2**31), (1, 2**32)]:
            with self.subTest(threads=threads, capacity=capacity), self.assertRaises(RecordError):
                BufferLayout(threads, capacity)
        for grid in [(0, 1, 1), (1, 1), (1, True, 1), (2**32, 1, 1)]:
            with self.assertRaises(RecordError):
                BufferLayout.from_dimensions(grid, (32, 1, 1))

    def test_thread_ordinals_and_per_site_occurrences(self):
        identity, layout = launch(), BufferLayout(2, 4)
        data = raw(identity, layout, [[(2**60, 2**60 + 1, 7), (2**60 + 2, 2**60 + 3, 8),
                                       (2**60 + 4, 2**60 + 5, 7)], []])
        events = decode(data, launch=identity, layout=layout, sites=[7, 8])
        self.assertEqual([event.ordinal for event in events], [0, 1, 2])
        self.assertEqual([event.occurrence for event in events], [0, 0, 1])
        self.assertEqual(events[0].end - events[0].begin, 1)
        self.assertTrue(all(event.launch_id == identity.id for event in events))

    def test_launch_id_and_ordinal_are_both_checked(self):
        identity, layout = launch(), BufferLayout(1, 2)
        data = raw(identity, layout, [[]])
        for other in [replace(identity, ordinal=2), replace(identity, kernel=replace(
                identity.kernel, module=replace(identity.kernel.module, context=replace(
                    identity.kernel.module.context, process=ProcessIdentity('other-session', 17, 'unique-process')))))]:
            with self.assertRaisesRegex(RecordError, 'header'):
                decode(data, launch=other, layout=layout, sites=[])
        with self.assertRaisesRegex(RecordError, 'header'):
            decode(data[:-1], launch=identity, layout=layout, sites=[])

    def test_overflow_incomplete_and_malformed_thread_headers(self):
        identity, layout = launch(), BufferLayout(1, 2)
        for state in [(2, 1, 1, 0), (3, 0, 1, 0), (0, 0, 0, 0), (0, 0, 2, 0), (0, 0, 1, 1)]:
            data = raw(identity, layout, [[]])
            STATE.pack_into(data, HEADER.size, *state)
            with self.subTest(state=state), self.assertRaises(RecordError):
                decode(data, launch=identity, layout=layout, sites=[7])

    def test_invalid_commit_ordinal_site_time_and_unused_slot(self):
        identity, layout = launch(), BufferLayout(1, 2)
        cases = [(10, 20, 7, 0, 1), (10, 20, 7, 1, 2), (10, 20, 8, 0, 2), (20, 10, 7, 0, 2)]
        for record in cases:
            data = raw(identity, layout, [[(10, 20, 7)]])
            RECORD.pack_into(data, HEADER.size + STATE.size, *record)
            with self.subTest(record=record), self.assertRaises(RecordError):
                decode(data, launch=identity, layout=layout, sites=[7])
        data = raw(identity, layout, [[(10, 20, 7)]])
        data[-1] = 1
        with self.assertRaisesRegex(RecordError, 'unallocated'):
            decode(data, launch=identity, layout=layout, sites=[7])

    def test_native_packed_preserves_all_threads_and_occurrences(self):
        identity, layout = launch(), BufferLayout(3, 4096)
        data = (STATE.pack(2, 0, 1, 0) + STATE.pack(0, 0, 1, 0) + STATE.pack(1, 0, 1, 0)
                + RECORD.pack(10, 20, 7, 0, 2) + RECORD.pack(21, 30, 7, 1, 2)
                + RECORD.pack(15, 25, 8, 0, 2))
        events = decode_native(data, launch=identity, layout=layout, sites=[7, 8], raw_format='states-packed-v1')
        self.assertEqual([(r.thread, r.ordinal, r.occurrence) for r in events], [(0, 0, 0), (0, 1, 1), (2, 0, 0)])
        for malformed in (data[:-1], data + bytes(32), data[:10], STATE.pack(2, 1, 1, 0) + data[16:],
                          STATE.pack(4097, 0, 1, 0) + data[16:], STATE.pack(2, 0, 0, 0) + data[16:]):
            with self.assertRaises(RecordError):
                decode_native(malformed, launch=identity, layout=layout, sites=[7, 8], raw_format='states-packed-v1')
        with self.assertRaises(RecordError):
            decode_native(data, launch=identity, layout=layout, sites=[7], raw_format='states-packed-v1')

    def test_native_fixed_compatibility(self):
        identity, layout = launch(), BufferLayout(2, 8)
        data = raw(identity, layout, [[(10, 20, 7)], []])
        self.assertEqual(decode(data, launch=identity, layout=layout, sites=[7]),
                         decode_native(data[HEADER.size:], launch=identity, layout=layout, sites=[7],
                                       raw_format='states-fixed-v1'))

    def test_instruction_intervals_do_not_overlap_within_one_thread(self):
        identity, layout = launch(), BufferLayout(1, 2)
        data = raw(identity, layout, [[(10, 20, 7), (19, 30, 7)]])
        with self.assertRaisesRegex(RecordError, 'ordering'):
            decode(data, launch=identity, layout=layout, sites=[7])


if __name__ == '__main__':
    unittest.main()
