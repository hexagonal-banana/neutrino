"""Fresh two-rank native journals exercise strict export and identity failures."""
import csv
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neutrino.sys_sync.export import (InstructionSources, NativeProcess, TraceError, export_trace, load_processes,
                                      kernel_track_label, normalize_chrome_document, select_warp_events, warp_coordinates)
from neutrino.sys_sync.records import STATE, RECORD, RecordError


def fixture(root, rank):
    directory = root / f'process-{rank + 10}'
    directory.mkdir()
    gpu, host = 2**60 + rank * 999999999, 10**15
    image_id = str(rank + 1) * 64
    image = {'id': image_id, 'original_hashes': [image_id], 'target': 'sm_90',
             'entries': {'entry': {}}, 'label': ['NCCL', 'DeepEP'][rank], 'interval_mode': 'instruction-only',
             'probe': {'sampling': 'all_threads', 'sites': [{'ordinal': 1, 'id': 's1', 'function': 'callee',
             'opcode': 'ld.acquire.sys.global.u32', 'observation': 'instruction', 'memory_order': 'acquire', 'line': 42}]}}
    (directory / (image_id + '.image.json')).write_text(json.dumps(image))
    data = STATE.pack(1, 0, 1, 0) + STATE.pack(0, 0, 1, 0) + RECORD.pack(gpu + 400000, gpu + 400001, 1, 0, 2)
    (directory / 'launch-1.bin').write_bytes(data)
    rows = [dict(kind='process', session='session', pid=rank + 10, started=host - 100,
                 rank=str(rank), hostname='single-host', boot_id='same-boot', host_clock='CLOCK_MONOTONIC_RAW'),
            dict(kind='context', context=1, uuid=f'GPU-{rank}'),
            dict(kind='load', owner=1, image=image_id, coverage='replaced'),
            dict(kind='calibration', step=3, context=1, uuid=f'GPU-{rank}', phase='before',
                 samples=[[host - 10, gpu, host + 10], [host + 990, gpu + 1000, host + 1010]]),
            dict(kind='capture_begin', step=3, name='step-3'),
            dict(kind='launch', ordinal=1, step=3, context=1, uuid=f'GPU-{rank}', owner=1, image=image_id,
                 coverage='instrumented', api='cuLaunchKernelEx', kernel='entry', grid=[1, 1, 1], block=[2, 1, 1], capacity=8,
                 threads=2, stream=100, status=0, raw='launch-1.bin', raw_format='states-packed-v1'),
            dict(kind='drain', ordinal=1, maximum_used=1, overflow_threads=0, incomplete_threads=0,
                 records=1, raw_bytes=len(data)),
            dict(kind='calibration', step=3, context=1, uuid=f'GPU-{rank}', phase='after',
                 samples=[[host + 999990, gpu + 1000000, host + 1000010],
                          [host + 1000990, gpu + 1001000, host + 1001010]]),
            dict(kind='capture_end', step=3, successful=True, recorded_launches=1), dict(kind='finalized')]
    times = [-100, -50, -25, 2000, 3000, 500000, 600000, 1002000, 1003000, 1004000]
    for timestamp, row in zip(times, rows):
        row['host_ns'] = host + timestamp
    journal = directory / 'journal.jsonl'
    journal.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return journal, rows


def add_activity(journal, rows):
    host = 10**15
    activity = dict(kind='kernel_activity', ordinal=1, step=3, correlation=123, kernel='entry',
                    grid=[1, 1, 1], block=[2, 1, 1], source='CUPTI_CONCURRENT_KERNEL',
                    host_clock='CLOCK_MONOTONIC_RAW', begin_host_ns=host + 300000,
                    end_host_ns=host + 450000, cupti_stream=7, host_ns=host + 1002000)
    check = dict(kind='activity_validation', step=3, matched=1, dropped=0, host_ns=host + 1002000)
    rows[-2:-2] = [activity, check]
    journal.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return activity, check


class SourceTests(unittest.TestCase):
    HEADER = '.version 9.1\n.target sm_90\n.address_size 64\n'

    def source(self, root, source, name='original.ptx', unit='module'):
        from dataclasses import asdict
        from neutrino.sys_sync.ptx import parse
        path = root / name
        path.write_text(source)
        module = parse(source)
        sites = [dict(asdict(site), id=f's{index}', ordinal=index, unit=unit)
                 for index, site in enumerate(module.sites, 1)]
        image = dict(id=module.sha256, label='DeepEP', probe={'original_sha256': {unit: module.sha256}})
        return path, image, sites

    def test_full_mnemonic_predicate_inline_caller_and_trailing_file_table(self):
        source = self.HEADER + '''.visible .entry k() {
.reg .pred %enabled, %done;
.loc 1 81 13, function_name helper, inlined_at 2 354 17
/*
.loc 99 999 0
*/
@!%enabled mbarrier.try_wait.parity.shared::cta.b64 %done, [%ptr], %phase, 100;
ret;
}
.file 1 "/build/include/ptx.cuh"
.file 2 "/build/include/dispatch.cuh"
'''
        with tempfile.TemporaryDirectory() as temp:
            path, image, sites = self.source(Path(temp), source)
            resolver = InstructionSources([path])
            name, annotation = resolver.annotate(image, sites[0], sites)
            opcode = 'mbarrier.try_wait.parity.shared::cta.b64'
            self.assertEqual(name, opcode + ' @ ptx.cuh:81:13 <- dispatch.cuh:354:17')
            self.assertEqual(annotation['instructions'],
                             ['@!%enabled ' + opcode + ' %done, [%ptr], %phase, 100;'])
            self.assertEqual(annotation['ptx_locations'], ['original.ptx:10'])
            row = resolver.inventory[annotation['refs'][0]]
            self.assertEqual(row['source_location'], dict(file='/build/include/ptx.cuh', line=81, column=13))
            self.assertEqual(row['inlined_at'], [dict(file='/build/include/dispatch.cuh', line=354, column=17)])
            self.assertEqual(row['ptx_sha256'], image['id'])
            self.assertEqual(resolver.annotate(image, sites[0], sites), (name, annotation))
            self.assertEqual(len(resolver.inventory), 1)
            self.assertEqual(resolver.report()['counts'], {'DeepEP_cuda_mapped_intervals': 2})

    def test_unavailable_locations_do_not_leak_between_functions_or_modules(self):
        source = self.HEADER + '''.entry first() {
.loc 1 12 0
membar.sys;
.loc 0 0 0
fence.sc.sys;
.loc 1 0 0
membar.sys;
.loc 1 99 2
ret;
}
.entry second() {
membar.sys;
ret;
}
.file 1 "/first/source.cu"
'''
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, image, sites = self.source(root, source)
            other, other_image, other_sites = self.source(
                root, self.HEADER + '.entry second() { membar.sys; ret; }', 'other.ptx', 'other-unit')
            resolver = InstructionSources([root])
            locations = [resolver.inventory[resolver.resolve(image, site)]['source_location'] for site in sites]
            self.assertEqual(locations, [dict(file='/first/source.cu', line=12, column=0), None, None, None])
            merged = dict(image, probe={'original_sha256': {**image['probe']['original_sha256'],
                                                          **other_image['probe']['original_sha256']}})
            name, annotation = resolver.annotate(merged, other_sites[0], other_sites)
            self.assertEqual(name, 'membar.sys @ PTX other.ptx:4')
            self.assertEqual(annotation['locations'], ['CUDA source unavailable'])
            self.assertEqual(len(resolver.modules), 2)
            loop = dict(id='loop', interval_kind='loop', exit_index=0)
            name, annotation = resolver.annotate(image, loop, sites[:2])
            self.assertTrue(name.startswith('Loop: membar.sys @ source.cu:12:0 + fence.sc.sys @ PTX'))
            self.assertTrue(name.endswith('/ exit 0'))
            self.assertEqual(len(annotation['refs']), 2)
            self.assertEqual(resolver.report()['counts']['DeepEP_cuda_partial_intervals'], 1)

    def test_missing_changed_hash_and_conflicting_site_metadata_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(TraceError, 'No source PTX'):
                InstructionSources([root])
            with self.assertRaisesRegex(TraceError, 'Missing source PTX'):
                InstructionSources([root / 'missing.ptx'])
            source = self.HEADER + '.entry k() { membar.sys; ret; }'
            path, image, sites = self.source(root, source)
            resolver = InstructionSources([path])
            with self.assertRaisesRegex(TraceError, 'Missing captured PTX identity'):
                resolver.resolve(dict(image, probe={}), sites[0])
            with self.assertRaisesRegex(TraceError, 'Missing hash-matched'):
                resolver.resolve(dict(image, probe={'original_sha256': {'module': '0' * 64}}), sites[0])
            for field, value in (('function', 'missing'), ('instruction', 999), ('start', -1),
                                 ('end', 0), ('line', 99), ('opcode', 'fence.sc.sys'), ('predicate', '%p')):
                with self.subTest(field=field), self.assertRaisesRegex(TraceError, 'Source site'):
                    resolver.resolve(image, dict(sites[0], **{field: value}))
            resolver.resolve(image, sites[0])
            with self.assertRaisesRegex(TraceError, 'Conflicting source site'):
                resolver.resolve(image, dict(sites[0], line=999))
            fresh = InstructionSources([path])
            path.write_text(source + '\n')
            with self.assertRaisesRegex(TraceError, 'changed after indexing'):
                fresh.resolve(image, sites[0])
            with self.assertRaisesRegex(TraceError, 'Missing hash-matched'):
                InstructionSources([path]).resolve(image, sites[0])

    def test_unknown_and_conflicting_debug_files_fail_closed(self):
        for files, message in (('', 'Unknown PTX debug file'),
                               ('.file 7 "a.cu"\n.file 7 "b.cu"\n', 'Conflicting PTX debug file')):
            with tempfile.TemporaryDirectory() as temp:
                path, image, sites = self.source(Path(temp), self.HEADER +
                                                '.entry k() {\n.loc 7 10 0\nmembar.sys; ret;\n}\n' + files)
                with self.assertRaisesRegex(TraceError, message):
                    InstructionSources([path]).resolve(image, sites[0])

    def test_source_export_preserves_instruction_loop_kernel_and_lane_evidence(self):
        from copy import deepcopy
        from neutrino.sys_sync.linkage import link_group
        from neutrino.sys_sync.probe import instrument_instructions
        from test_sys_loops import compare_source_export, validate_loop_export

        for mode in ('instruction-only', 'loop-boundary'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root, paths = Path(temp), []
                for rank in range(2):
                    journal, rows = fixture(root, rank)
                    debug = '.loc 1 81 13, inlined_at 2 354 17\n' if rank else ''
                    source = self.HEADER + '.entry entry() {\n.reg .pred %p;\n' + debug + '''fence.sc.sys;
L: mbarrier.try_wait.parity.shared::cta.b64 %p, [%ptr], %phase;
membar.sys;
@%p bra L;
ret;
}
.file 1 "/build/ptx.cuh"
.file 2 "/build/dispatch.cuh"
'''
                    path, _, _ = self.source(root, source, f'rank{rank}.ptx')
                    paths.append(path)
                    result = instrument_instructions(link_group({'module': source}), max_parameter_bytes=32764,
                                                     interval_mode=mode)
                    image_path = next(journal.parent.glob('*.image.json'))
                    image = json.loads(image_path.read_text())
                    probe = dict(result.manifest, sampling='all_threads')
                    probe.pop('context_abi')
                    image.update(interval_mode=mode, probe=probe)
                    image_path.write_text(json.dumps(image))
                    sites = probe['sites']
                    begin = 2**60 + rank * 999999999 + 400000
                    data = STATE.pack(len(sites), 0, 1, 0) + STATE.pack(0, 0, 1, 0)
                    data += b''.join(RECORD.pack(begin + 200 * i, begin + 100 + 200 * i, site['ordinal'], i, 2)
                                     for i, site in enumerate(sites))
                    (journal.parent / 'launch-1.bin').write_bytes(data)
                    next(row for row in rows if row['kind'] == 'drain').update(
                        maximum_used=len(sites), records=len(sites), raw_bytes=len(data))
                    add_activity(journal, rows)
                for grouping in ('sampled-thread', 'kernel-ctx-stream', 'kernel-rank-ctx-stream'):
                    before, after = root / (grouping + '-before'), root / grouping
                    options = dict(require_kernel_activity=True, instruction_granularity='captured', lane_grouping=grouping)
                    export_trace(root, before, **options)
                    report = export_trace(root, after, source_ptx=paths, **options)
                    count = len(sites) * 2
                    self.assertEqual(sum(report['source_annotation']['counts'].values()), count)
                    checked = compare_source_export(before, after, root)
                    self.assertEqual(checked['annotated_intervals'], count)
                    self.assertEqual(checked['duration_events'], count + 4)
                    if mode == 'loop-boundary':
                        self.assertEqual(validate_loop_export(after)['exported_intervals'], count)
                    trace_path = after / 'timeline.json'
                    trace = json.loads(trace_path.read_text())
                    for event in trace['traceEvents']:
                        if event.get('cat') == 'DeepEP.sys_loop':
                            self.assertEqual(event['name'], 'Loop: mbarrier.try_wait.parity.shared::cta.b64 @ '
                                             'ptx.cuh:81:13 <- dispatch.cuh:354:17 + membar.sys @ '
                                             'ptx.cuh:81:13 <- dispatch.cuh:354:17 / exit 0')
                            self.assertEqual(len(event['args']['source_annotation']['instructions']), 2)
                    if grouping != 'kernel-rank-ctx-stream':
                        continue
                    for mutation in ('timestamp', 'lane', 'name', 'argument', 'instruction', 'ref', 'dropped'):
                        broken = deepcopy(trace)
                        event = next(e for e in broken['traceEvents'] if '.sys_' in e.get('cat', ''))
                        if mutation == 'timestamp':
                            event['ts'] += 1
                        elif mutation == 'lane':
                            event['tid'] += 1
                        elif mutation == 'name':
                            event['name'] = 'wrong'
                        elif mutation == 'argument':
                            event['args']['thread'] += 1
                        elif mutation == 'instruction':
                            event['args']['source_annotation']['instructions'][0] = 'wrong;'
                        elif mutation == 'ref':
                            event['args']['source_annotation']['refs'][0] = 'missing'
                        else:
                            broken['traceEvents'].remove(event)
                        trace_path.write_text(json.dumps(broken, separators=(',', ':')))
                        with self.subTest(mutation=mutation), self.assertRaises((AssertionError, KeyError)):
                            compare_source_export(before, after, root)
                    trace_path.write_text(json.dumps(trace, separators=(',', ':')))


class ExportTests(unittest.TestCase):
    def test_kernel_track_names_and_specialization_tags(self):
        for library, kernel, readable in (
                ('DeepEP', '_Z13dispatch_implIiE', 'dispatch_impl'),
                ('DeepEP', '_Z22dispatch_copy_epilogueIiE', 'dispatch_copy_epilogue'),
                ('DeepEP', '_Z23combine_reduce_epilogueIiE', 'combine_reduce_epilogue'),
                ('NCCL', 'ncclDevKernel_ReduceScatter', 'ReduceScatter'),
                ('NCCL', 'ncclDevKernel_Broadcast', 'Broadcast'),
                ('custom', 'custom_entry', 'custom_entry')):
            with self.subTest(kernel=kernel):
                label = kernel_track_label(library, kernel, 'image')
                self.assertRegex(label, '^' + readable + r' \[[0-9a-f]{12}\]$')
                self.assertEqual(label, kernel_track_label(library, kernel, 'image'))
                self.assertNotEqual(label, kernel_track_label(library, kernel, 'other-image'))
        self.assertNotEqual(kernel_track_label('DeepEP', 'dispatch_implIiE', 'image'),
                            kernel_track_label('DeepEP', 'dispatch_implIfE', 'image'))

    def test_kernel_lanes_separate_entries_images_and_reuse_repeated_launches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            journal, rows = fixture(root, 1)
            kernels = ['dispatch_implIiE', 'combine_implIiE', 'dispatch_implIfE',
                       'dispatch_implIiE', 'dispatch_implIiE']
            path = next(journal.parent.glob('*.image.json'))
            image = json.loads(path.read_text())
            image['entries'] = dict.fromkeys(kernels, {})
            path.write_text(json.dumps(image))
            second_id = '3' * 64
            second = dict(image, id=second_id, original_hashes=[second_id])
            (journal.parent / (second_id + '.image.json')).write_text(json.dumps(second))
            launch = next(row for row in rows if row['kind'] == 'launch')
            drain = next(row for row in rows if row['kind'] == 'drain')
            rows[5:7] = [dict(launch, ordinal=i + 1, kernel=kernel,
                              owner=2 if i == 4 else 1, image=second_id if i == 4 else image['id'])
                         for i, kernel in enumerate(kernels)] + [dict(drain, ordinal=i + 1) for i in range(5)]
            rows.insert(3, dict(rows[2], owner=2, image=second_id))
            next(row for row in rows if row['kind'] == 'capture_end')['recorded_launches'] = 5
            activity, check = add_activity(journal, rows)
            activity['kernel'] = kernels[0]
            rows[rows.index(check):rows.index(check)] = [dict(activity, ordinal=i + 1, kernel=kernels[i])
                                                        for i in range(1, 5)]
            check['matched'] = 5
            journal.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            for granularity in ('thread', 'warp', 'captured'):
                with self.subTest(granularity=granularity):
                    directory = root / granularity
                    report = export_trace(root, directory, require_kernel_activity=True,
                                          instruction_granularity=granularity)
                    trace = json.loads((directory / 'timeline.json').read_text())
                    self.assertEqual(report['lane_layout'], 'kernel-qualified-v1')
                    self.assertEqual(trace['metadata']['lane_layout'], report['lane_layout'])
                    names = {(event['pid'], event['tid']): event['args']['name']
                             for event in trace['traceEvents'] if event['name'] == 'thread_name'}
                    for category in ('DeepEP.sys_instruction', 'EP.DeepEP.kernel'):
                        events = [event for event in trace['traceEvents'] if event.get('cat') == category]
                        self.assertEqual(len(events), 5)
                        self.assertEqual(events[0]['tid'], events[3]['tid'])
                        self.assertEqual(len({event['tid'] for event in events}), 4)
                        for event, kernel in zip(events, kernels):
                            self.assertEqual(event['args']['kernel'], kernel)
                            label = names.get((event['pid'], event['tid']), event['tid'])
                            self.assertIn(kernel.split('I')[0], label)
                            self.assertEqual(event['dur'], 150 if category.endswith('.kernel') else 0.001)
                    with gzip.open(directory / 'timeline.json.gz', 'rt') as stream:
                        self.assertEqual(json.load(stream), trace)

    def test_kernel_context_stream_grouping_preserves_all_observers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            configurations = [(1, 100, 'entry'), (1, 200, 'entry'), (2, 100, 'entry'),
                              (1, 100, 'entry2'), (1, 100, 'entry')]
            for rank in range(2):
                journal, rows = fixture(root, rank)
                image_path = next(journal.parent.glob('*.image.json'))
                image = json.loads(image_path.read_text())
                image['entries']['entry2'] = {}
                image_path.write_text(json.dumps(image))
                gpu = 2**60 + rank * 999999999
                data = STATE.pack(1, 0, 1, 0) * 128
                data += b''.join(RECORD.pack(gpu + 400000 + i, gpu + 400100 + i, 1, 0, 2)
                                 for i in range(128))
                (journal.parent / 'launch-1.bin').write_bytes(data)
                add_activity(journal, rows)
                expanded = []
                for row in rows:
                    kind = row['kind']
                    if kind in {'launch', 'drain', 'kernel_activity'}:
                        for ordinal, (context, stream, kernel) in enumerate(configurations, 1):
                            item = dict(row, ordinal=ordinal)
                            if kind == 'drain':
                                item.update(records=128, raw_bytes=len(data))
                            else:
                                item.update(kernel=kernel, grid=[2, 1, 1], block=[8, 4, 2])
                                if kind == 'launch':
                                    item.update(context=context, owner=context, stream=stream, threads=128)
                            expanded.append(item)
                    else:
                        if kind == 'capture_end':
                            row['recorded_launches'] = len(configurations)
                        elif kind == 'activity_validation':
                            row['matched'] = len(configurations)
                        expanded.append(row)
                        if kind in {'context', 'calibration'}:
                            expanded.append(dict(row, context=2))
                        elif kind == 'load':
                            expanded.append(dict(row, owner=2))
                journal.write_text(''.join(json.dumps(row) + '\n' for row in expanded))
            for granularity in ('thread', 'warp', 'captured'):
                with self.subTest(granularity=granularity):
                    original, grouped = root / (granularity + '-original'), root / granularity
                    export_trace(root, original, instruction_granularity=granularity)
                    report = export_trace(root, grouped, instruction_granularity=granularity,
                                          lane_grouping='kernel-ctx-stream', require_kernel_activity=True)
                    self.assertEqual(report['lane_layout'], 'kernel-ctx-stream-v1')
                    before = json.loads((original / 'timeline.json').read_text())['traceEvents']
                    after = json.loads((grouped / 'timeline.json').read_text())['traceEvents']
                    durations = lambda events: [dict(event, tid=0) for event in events if event.get('ph') == 'X']
                    self.assertEqual(durations(before), durations(after))
                    names = {(event['pid'], event['tid']): event['args']['name']
                             for event in after if event['name'] == 'thread_name'}
                    owners, lanes = {}, {}
                    for event in after:
                        if not event.get('cat', '').endswith(('.kernel', '.sys_instruction')):
                            continue
                        args = event['args']
                        context, stream, kernel = configurations[args['launch_ordinal'] - 1]
                        owner = event['pid'], args['image'], kernel, context, stream
                        lane = event['pid'], event['tid']
                        self.assertEqual(owners.setdefault(lane, owner), owner)
                        self.assertEqual(lanes.setdefault(owner, lane), lane)
                        self.assertTrue(names[lane].endswith(f'/ctx{context}/stream{stream}'))
                        self.assertNotIn('/block', names[lane])
                        self.assertNotIn('/warp', names[lane])
                        self.assertNotIn('/thread', names[lane])
                    self.assertEqual(len(lanes), 8)
                    with gzip.open(grouped / 'timeline.json.gz', 'rt') as stream:
                        self.assertEqual(json.load(stream)['traceEvents'], after)
            with self.assertRaisesRegex(TraceError, 'lane grouping'):
                export_trace(root, root / 'invalid', lane_grouping='invalid')
            self.assertFalse((root / 'invalid').exists())

    def test_kernel_first_groups_rank_local_handles_and_preserves_events(self):
        from test_sys_loops import compare_kernel_lane_export

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shared, variant = 'a' * 64, 'b' * 64
            entry, specialized = 'dispatch_implIiE', 'dispatch_implIfE'
            configurations = [(2, 200, entry, shared), (1, 100, entry, shared),
                              (2, 100, entry, shared), (1, 100, specialized, shared),
                              (2, 200, entry, shared), (1, 100, entry, variant)]
            for index, rank in enumerate((12, 2)):
                journal, rows = fixture(root, index)
                image = json.loads(next(journal.parent.glob('*.image.json')).read_text())
                for image_id, library in ((shared, 'DeepEP'), (variant, ('DeepEP', 'NCCL')[index])):
                    manifest = dict(image, id=image_id, original_hashes=[image_id], label=library,
                                    entries={entry: {}, specialized: {}})
                    (journal.parent / (image_id + '.image.json')).write_text(json.dumps(manifest))
                gpu = 2**60 + index * 999999999
                data = STATE.pack(1, 0, 1, 0) * 128
                data += b''.join(RECORD.pack(gpu + 400000 + i, gpu + 400100 + i, 1, 0, 2)
                                 for i in range(128))
                (journal.parent / 'launch-1.bin').write_bytes(data)
                add_activity(journal, rows)
                expanded = []
                for row in rows:
                    kind = row['kind']
                    if kind in {'launch', 'drain', 'kernel_activity'}:
                        for ordinal, (context, stream, kernel, image_id) in enumerate(configurations, 1):
                            context += index * 10
                            item = dict(row, ordinal=ordinal)
                            if kind == 'drain':
                                item.update(records=128, raw_bytes=len(data))
                            else:
                                item.update(kernel=kernel, grid=[2, 1, 1], block=[8, 4, 2])
                                if kind == 'launch':
                                    item.update(context=context, owner=context + (100 if image_id == variant else 0),
                                                image=image_id, stream=stream, threads=128)
                                else:
                                    item['kernel'] = 'demangled ' + kernel
                            expanded.append(item)
                    elif kind in {'context', 'calibration'}:
                        expanded.extend(dict(row, context=context + index * 10) for context in (1, 2))
                    elif kind == 'load':
                        expanded.extend(dict(row, owner=context + index * 10, image=shared) for context in (1, 2))
                        expanded.append(dict(row, owner=101 + index * 10, image=variant))
                    else:
                        if kind == 'process':
                            row['rank'] = str(rank)
                        elif kind == 'capture_end':
                            row['recorded_launches'] = len(configurations)
                        elif kind == 'activity_validation':
                            row['matched'] = len(configurations)
                        expanded.append(row)
                journal.write_text(''.join(json.dumps(row) + '\n' for row in expanded))
            for granularity in ('thread', 'warp', 'captured'):
                with self.subTest(granularity=granularity):
                    original, grouped = root / (granularity + '-original'), root / granularity
                    export_trace(root, original, instruction_granularity=granularity)
                    report = export_trace(root, grouped, instruction_granularity=granularity,
                                          lane_grouping='kernel-rank-ctx-stream', require_kernel_activity=True)
                    self.assertEqual(report['lane_layout'], 'kernel-rank-ctx-stream-v1')
                    comparison = compare_kernel_lane_export(original, grouped)
                    self.assertEqual(comparison['gpu_lanes'], 10)
                    self.assertEqual(comparison['kernel_display_groups'], 4)
                    self.assertEqual(sorted(comparison['ranks_by_kernel_group'].values()),
                                     [[2], [2, 12], [2, 12], [12]])
                    trace = json.loads((grouped / 'timeline.json').read_text())
                    events = trace['traceEvents']
                    lanes = {(event['pid'], event['tid']): event['args'] for event in events
                             if event['name'] == 'thread_name'}
                    original_events = json.loads((original / 'timeline.json').read_text())['traceEvents']
                    durations = lambda items: [dict(event, tid=0) for event in items if event['ph'] == 'X']
                    restored = [dict(event, pid=lanes[event['pid'], event['tid']]['source_pid'])
                                if event.get('cat', '').endswith(('.kernel', '.sys_instruction')) else event
                                for event in events]
                    self.assertEqual(durations(original_events), durations(restored))
                    kernels = [event for event in events if event.get('cat', '').endswith('.kernel')]
                    self.assertEqual(kernels[0]['pid'], kernels[6]['pid'])
                    self.assertNotEqual(kernels[0]['args']['launch_id'], kernels[6]['args']['launch_id'])
                    self.assertEqual(kernels[0]['tid'], kernels[4]['tid'])
                    self.assertEqual(kernels[0]['pid'], kernels[1]['pid'])
                    self.assertNotEqual(kernels[0]['tid'], kernels[1]['tid'])
                    self.assertEqual({event['pid'] for event in events if event.get('cat') == 'host.capture'}, {10, 11})
                    with gzip.open(grouped / 'timeline.json.gz', 'rt') as stream:
                        self.assertEqual(json.load(stream), trace)
                    overview_path = grouped / 'communication.json'
                    overview = json.loads(overview_path.read_text())
                    self.assertEqual(normalize_chrome_document(overview), overview)
                    self.assertFalse(any(event.get('cat', '').endswith('.sys_instruction')
                                         for event in overview['traceEvents']))
                    self.assertEqual([event for event in overview['traceEvents'] if event['ph'] == 'M'],
                                     [event for event in events if event['ph'] == 'M'])
                    for layout in ('kernel-rank-ctx-stream', 'kernel-ctx-stream'):
                        reused = root / (granularity + '-reused-' + layout)
                        export_trace(root, reused, instruction_granularity=granularity,
                                     lane_grouping=layout, kernel_overview=overview_path)
                        reused_events = json.loads((reused / 'timeline.json').read_text())['traceEvents']
                        reused_kernels = [event for event in reused_events if event.get('cat', '').endswith('.kernel')]
                        expected = [event for event in overview['traceEvents'] if event.get('cat', '').endswith('.kernel')]
                        self.assertEqual([dict(event, pid=0, tid=0) for event in reused_kernels],
                                         [dict(event, pid=0, tid=0) for event in expected])
                        if layout == 'kernel-ctx-stream':
                            self.assertEqual({event['pid'] for event in reused_kernels}, {10, 11})

    def test_explicit_original_image_exclusion_and_corrupt_evidence(self):
        from copy import deepcopy
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, first_rows = fixture(root, 0)
            add_activity(first, first_rows)
            journal, rows = fixture(root, 1)
            image_path = next(journal.parent.glob('*.image.json'))
            image = json.loads(image_path.read_text())
            image['probe_enabled'] = False
            image_path.write_text(json.dumps(image))
            rows = [row for row in rows if row['kind'] != 'drain']
            launch = next(row for row in rows if row['kind'] == 'launch')
            launch['coverage'] = 'observed'
            for key in ('raw', 'raw_format', 'threads', 'capacity'):
                launch.pop(key)
            next(row for row in rows if row['kind'] == 'load')['coverage'] = 'original'
            next(row for row in rows if row['kind'] == 'capture_end').update(
                recorded_launches=0, observed_launches=1)
            add_activity(journal, rows)
            process = NativeProcess(journal)
            self.assertEqual(process.decode_launch(process.launches[0])[2:], ({}, [], None))
            report = export_trace(root, root / 'mixed', require_kernel_activity=True,
                                  instruction_granularity='captured')
            self.assertEqual(report['counts']['NCCL_intervals'], 1)
            self.assertEqual(report['counts']['DeepEP_input_intervals'], 0)
            self.assertEqual(report['counts']['DeepEP_excluded_launches'], 1)
            self.assertEqual(report['counts']['DeepEP_kernel_spans'], 1)
            self.assertEqual(report['instruction_export']['excluded_image_ids'], [image['id']])
            coverage = json.loads((root / 'mixed/coverage.json').read_text())[1]
            self.assertEqual(coverage['disposition'], 'probe_excluded_original_image')
            self.assertIsNone(coverage['raw'])
            self.assertNotIn('raw_sha256', coverage)
            self.assertEqual(coverage['instruction_export']['exported_records'], 0)
            for mutation in ('undeclared', 'nonboolean', 'owner', 'raw', 'allocation', 'drain',
                             'missing_activity', 'accounting', 'enabled_recording'):
                broken, manifest = deepcopy(rows), deepcopy(image)
                current = next(row for row in broken if row['kind'] == 'launch')
                if mutation == 'undeclared':
                    manifest.pop('probe_enabled')
                elif mutation == 'nonboolean':
                    manifest['probe_enabled'] = 0
                elif mutation == 'owner':
                    next(row for row in broken if row['kind'] == 'load')['coverage'] = 'replaced'
                elif mutation == 'raw':
                    current['raw'] = 'launch-1.bin'
                elif mutation == 'allocation':
                    current['capacity'] = 16
                elif mutation == 'drain':
                    broken.insert(-2, dict(kind='drain', ordinal=1, host_ns=broken[-2]['host_ns']))
                elif mutation == 'missing_activity':
                    broken = [row for row in broken if row['kind'] not in ('kernel_activity', 'activity_validation')]
                elif mutation == 'accounting':
                    next(row for row in broken if row['kind'] == 'capture_end')['observed_launches'] = 0
                else:
                    current['coverage'] = 'instrumented'
                image_path.write_text(json.dumps(manifest))
                journal.write_text(''.join(json.dumps(row) + '\n' for row in broken))
                with self.subTest(mutation=mutation), self.assertRaises(TraceError):
                    NativeProcess(journal)

    def test_disabled_manifest_cannot_decode_instrumented_image(self):
        with tempfile.TemporaryDirectory() as temp:
            journal, rows = fixture(Path(temp), 0)
            path = next(journal.parent.glob('*.image.json'))
            image = json.loads(path.read_text())
            image['probe_enabled'] = False
            path.write_text(json.dumps(image))
            process = NativeProcess(journal)
            with self.assertRaisesRegex(TraceError, 'probe exclusion'):
                process.decode_launch(process.launches[0])

    def test_coverage_policy_export_and_mixed_policy_rejection(self):
        from neutrino.sys_sync.ptx import SITE_SELECTION
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = []
            for rank in range(2):
                journal, _ = fixture(root, rank)
                path = next(journal.parent.glob('*.image.json'))
                image = json.loads(path.read_text())
                image['probe']['site_selection'] = SITE_SELECTION
                image['probe']['sites'][0].update(opcode='mbarrier.try_wait.shared.b64',
                                                 observation='barrier_instruction', memory_order='implicit')
                path.write_text(json.dumps(image))
                paths.append(path)
            report = export_trace(root, root / 'selected', instruction_granularity='captured')
            self.assertEqual(report['instruction_export']['site_selection'], SITE_SELECTION)
            trace = json.loads((root / 'selected/timeline.json').read_text())
            self.assertEqual(trace['metadata']['instruction_export']['site_selection'], SITE_SELECTION)
            events = [e for e in trace['traceEvents'] if e.get('cat', '').endswith('.sys_instruction')]
            self.assertEqual([e['name'] for e in events], ['mbarrier.try_wait.shared.b64'] * 2)
            image = json.loads(paths[1].read_text())
            image['probe'].pop('site_selection')
            paths[1].write_text(json.dumps(image))
            with self.assertRaisesRegex(TraceError, 'Mixed instruction coverage'):
                export_trace(root, root / 'mixed')

    def test_warp_coordinates_restart_at_partial_multidimensional_blocks(self):
        block = [3, 3, 4]
        self.assertEqual(warp_coordinates(31, block), (0, 0, 31))
        self.assertEqual(warp_coordinates(32, block), (0, 1, 0))
        self.assertEqual(warp_coordinates(35, block), (0, 1, 3))
        self.assertEqual(warp_coordinates(36, block), (1, 0, 0))
        self.assertEqual(warp_coordinates(68, block), (1, 1, 0))

    def test_warp_selection_preserves_nonzero_sites_and_representative_repeats(self):
        from types import SimpleNamespace
        rows = [(31, 1), (31, 1), (30, 1), (30, 1), (31, 2), (32, 1), (35, 1), (36, 1)]
        events = [SimpleNamespace(thread=thread, site=site, ordinal=i) for i, (thread, site) in enumerate(rows)]
        selected, stats = select_warp_events(events, [3, 3, 4])
        self.assertEqual([event.ordinal for event in selected], [2, 3, 4, 5, 7])
        self.assertEqual(stats, {'observed_warp_sites': 4, 'warp_sites_without_lane_zero': 2})
        self.assertEqual(select_warp_events([], [32, 1, 1])[0], [])
        self.assertEqual([event.ordinal for event in events], list(range(8)))

    def test_warp_export_timestamps_sites_kernel_labels_and_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            journal, rows = fixture(root, 0)
            gpu = 2**60
            entries = [[(gpu + 400000, gpu + 400001, 1)] for _ in range(64)]
            entries[31] += [(gpu + 400010, gpu + 400011, 2), (gpu + 400020, gpu + 400021, 2)]
            data = b''.join(STATE.pack(len(values), 0, 1, 0) for values in entries)
            data += b''.join(RECORD.pack(begin, end, site, ordinal, 2)
                             for values in entries for ordinal, (begin, end, site) in enumerate(values))
            (journal.parent / 'launch-1.bin').write_bytes(data)
            image_path = next(journal.parent.glob('*.image.json'))
            image = json.loads(image_path.read_text())
            image['probe']['sites'].append(dict(image['probe']['sites'][0], ordinal=2, id='s2',
                                                opcode='st.release.sys.global.u32', memory_order='release'))
            image_path.write_text(json.dumps(image))
            for row in rows:
                if row['kind'] == 'launch':
                    row.update(block=[8, 4, 2], threads=64)
                if row['kind'] == 'drain':
                    row.update(records=66, raw_bytes=len(data), maximum_used=3)
            activity, _ = add_activity(journal, rows)
            activity['block'] = [8, 4, 2]
            journal.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            export_trace(root, root / 'thread', require_kernel_activity=True)
            overview = json.loads((root / 'thread/communication.json').read_text())
            for event in overview['traceEvents']:
                if event.get('cat', '').endswith('.kernel'):
                    event.update(cat='TP.NCCL.kernel', name='TP entry')
                    event['args'].update(parallel_role='TP', role_evidence={'method': 'fixture'})
                if event.get('name') == 'thread_name':
                    event['args']['name'] = event['args']['name'].replace('DP NCCL', 'TP NCCL')
            source = root / 'roles.json'
            source.write_text(json.dumps(overview))
            report = export_trace(root, root / 'warp', require_kernel_activity=True,
                                  instruction_granularity='warp', kernel_overview=source)
            self.assertEqual(report['counts']['NCCL_input_intervals'], 66)
            self.assertEqual(report['counts']['NCCL_intervals'], 4)
            self.assertEqual(report['counts']['NCCL_filtered_intervals'], 62)
            self.assertEqual(report['counts']['NCCL_observed_warp_sites'], 3)
            self.assertEqual(report['counts']['NCCL_warp_sites_without_lane_zero'], 1)
            self.assertFalse(report['instruction_export']['instruction_observations_complete'])
            trace = json.loads((root / 'warp/timeline.json').read_text())
            with gzip.open(root / 'warp/timeline.json.gz', 'rt') as stream:
                self.assertEqual(json.load(stream), trace)
            self.assertEqual(trace['metadata']['instruction_export'], report['instruction_export'])
            original = json.loads((root / 'thread/timeline.json').read_text())
            selected = [event for event in trace['traceEvents'] if event.get('cat') == 'NCCL.sys_instruction']
            candidates = [event for event in original['traceEvents'] if event.get('cat') == 'NCCL.sys_instruction']
            for event in selected:
                self.assertIsInstance(event['tid'], int)
                match = next(item for item in candidates if item['args'] == event['args'])
                self.assertEqual((event['name'], event['ts'], event['dur']),
                                 (match['name'], match['ts'], match['dur']))
            self.assertEqual({event['args']['thread'] for event in selected}, {0, 31, 32})
            self.assertEqual(selected[0]['tid'], selected[1]['tid'])
            self.assertEqual(len({event['tid'] for event in selected}), 2)
            names = [event['args']['name'] for event in trace['traceEvents'] if event['name'] == 'thread_name']
            self.assertTrue(any('/block0/warp1' in name for name in names))
            kernels = [event for event in trace['traceEvents'] if event.get('cat', '').endswith('.kernel')]
            self.assertEqual(kernels[0]['cat'], 'TP.NCCL.kernel')
            self.assertEqual(kernels[0]['args']['sys_instruction_records'], 66)
            self.assertEqual(kernels[0]['dur'], 150)
            with (root / 'warp/events.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)
            coverage = json.loads((root / 'warp/coverage.json').read_text())[0]
            self.assertEqual(coverage['intervals'], 66)
            self.assertEqual(coverage['instruction_export']['exported_records'], 4)
            self.assertEqual((journal.parent / 'launch-1.bin').read_bytes(), data)
            comparison = compare_warp_export(root / 'thread', root / 'warp', kernel_overview=source)
            self.assertEqual(comparison['instruction_records'], {'thread': 66, 'warp': 4})
            self.assertEqual(comparison['instruction_tracks'], {'thread': 64, 'warp': 2})
            with self.assertRaises(FileExistsError):
                export_trace(root, root / 'warp', instruction_granularity='warp')

    def test_warp_mode_keeps_kernel_without_instruction_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            journal, rows = fixture(root, 0)
            (journal.parent / 'launch-1.bin').write_bytes(STATE.pack(0, 0, 1, 0) * 2)
            for row in rows:
                if row['kind'] == 'drain':
                    row.update(records=0, raw_bytes=32, maximum_used=0)
            add_activity(journal, rows)
            report = export_trace(root, root / 'warp', instruction_granularity='warp')
            self.assertEqual(report['counts']['NCCL_kernel_spans'], 1)
            self.assertEqual(report['counts']['NCCL_input_intervals'], 0)
            self.assertEqual(report['counts']['NCCL_observed_warp_sites'], 0)

    def test_warp_cli_and_invalid_options(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, 0)
            command = [sys.executable, '-m', 'neutrino.sys_sync.export', '--raw', str(root),
                       '--output', str(root / 'warp'), '--world-size', '1', '--steps', '3',
                       '--instruction-granularity', 'warp', '--lane-grouping', 'kernel-rank-ctx-stream']
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                    text=True, capture_output=True, check=True)
            report = json.loads(result.stdout)
            self.assertEqual(report['instruction_export']['instruction_granularity'], 'warp')
            self.assertEqual(report['lane_layout'], 'kernel-rank-ctx-stream-v1')
            with self.assertRaisesRegex(TraceError, 'granularity'):
                export_trace(root, root / 'bad', instruction_granularity='block')
            self.assertFalse((root / 'bad').exists())

    def test_kernel_overview_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            journal, rows = fixture(root, 0)
            add_activity(journal, rows)
            export_trace(root, root / 'base')
            source = root / 'base/communication.json'
            original = json.loads(source.read_text())
            for i, change in enumerate(('session', 'timestamp', 'missing', 'duplicate', 'orphan')):
                document = json.loads(json.dumps(original))
                kernels = [event for event in document['traceEvents'] if event.get('cat', '').endswith('.kernel')]
                if change == 'session':
                    document['metadata']['session'] = 'different'
                elif change == 'timestamp':
                    kernels[0]['dur'] += 1
                elif change == 'missing':
                    document['traceEvents'].remove(kernels[0])
                else:
                    extra = json.loads(json.dumps(kernels[0]))
                    if change == 'orphan':
                        extra['args']['launch_id'] = 'orphan'
                    document['traceEvents'].append(extra)
                candidate = root / f'bad-{i}.json'
                candidate.write_text(json.dumps(document))
                with self.assertRaises(TraceError):
                    export_trace(root, root / f'out-{i}', instruction_granularity='warp', kernel_overview=candidate)

    def test_portable_thread_ids_labels_and_exact_timestamps(self):
        event = dict(ph='X', pid=17, tid='DP NCCL/stream1', ts=123.125, dur=4.75,
                     cat='DP.NCCL.kernel', name='_Z28ncclDevKernel_AllGather_RING',
                     args={'library': 'NCCL', 'kernel': '_Z28ncclDevKernel_AllGather_RING'})
        source = {'traceEvents': [event], 'metadata': {'origin_host_ns': 2**60}}
        result = normalize_chrome_document(source)
        self.assertEqual(result['traceEvents'][0]['args']['name'], event['tid'])
        span = result['traceEvents'][1]
        self.assertIsInstance(span['tid'], int)
        self.assertEqual(span['name'], 'NCCL AllGather')
        self.assertEqual((span['ts'], span['dur'], span['args']), (event['ts'], event['dur'], event['args']))
        self.assertEqual(result['metadata'], source['metadata'])
        self.assertIsInstance(event['tid'], str)
        self.assertEqual(normalize_chrome_document(result), result)

    def test_independent_spans_and_no_sys_site_kernel_are_visible(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for rank in range(2):
                journal, rows = fixture(root, rank)
                if rank == 0:
                    (journal.parent / 'launch-1.bin').write_bytes(STATE.pack(0, 0, 1, 0) * 2)
                    for row in rows:
                        if row['kind'] == 'drain':
                            row.update(records=0, raw_bytes=32, maximum_used=0)
                add_activity(journal, rows)
            result = export_trace(root, root / 'export', ranks=range(2), steps=[3], require_kernel_activity=True)
            self.assertTrue(result['full_trace_acceptance'])
            self.assertEqual(result['counts']['NCCL_kernel_spans'], 1)
            trace = json.loads((root / 'export/communication.json').read_text())
            kernels = [r for r in trace['traceEvents'] if r.get('cat', '').endswith('.kernel')]
            self.assertEqual({r['cat'] for r in kernels}, {'DP.NCCL.kernel', 'EP.DeepEP.kernel'})
            self.assertEqual(kernels[0]['args']['sys_instruction_records'], 0)
            self.assertEqual(kernels[0]['dur'], 150)

    def test_activity_required_and_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            journal, rows = fixture(root, 0)
            with self.assertRaisesRegex(TraceError, 'required'):
                export_trace(root, root / 'missing', require_kernel_activity=True)
            activity, check = add_activity(journal, rows)
            for target, key, value in [(check, 'dropped', 1), (check, 'matched', 2),
                                       (activity, 'ordinal', 2), (activity, 'step', 4),
                                       (activity, 'begin_host_ns', 0)]:
                previous = target[key]
                target[key] = value
                journal.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                with self.assertRaises(TraceError):
                    NativeProcess(journal)
                target[key] = previous
            activity['end_host_ns'] = activity['begin_host_ns'] + 1
            journal.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            process = NativeProcess(journal)
            with self.assertRaisesRegex(TraceError, 'outside independent'):
                process.decode_launch(process.launches[0])

    def test_two_clocks_one_timeline_exact_ticks_and_all_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for rank in range(2):
                fixture(root, rank)
            report = export_trace(root, root / 'decoded', ranks=range(2), steps=[3])
            self.assertEqual(report['counts']['NCCL_intervals'], 1)
            self.assertEqual(report['counts']['DeepEP_intervals'], 1)
            self.assertFalse(report['full_trace_acceptance'])
            trace = json.loads((root / 'decoded/timeline.json').read_text())
            with gzip.open(root / 'decoded/timeline.json.gz', 'rt') as stream:
                self.assertEqual(json.load(stream), trace)
            instructions = [row for row in trace['traceEvents'] if '.sys_instruction' in row.get('cat', '')]
            self.assertEqual(instructions[0]['ts'], instructions[1]['ts'])
            self.assertEqual(instructions[0]['dur'], 0.001)
            self.assertIsInstance(instructions[0]['args']['begin_tick'], str)
            self.assertNotEqual(instructions[0]['args']['launch_id'], instructions[1]['args']['launch_id'])
            with (root / 'decoded/events.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_rejects_overflow_missing_rank_and_step(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            journal, rows = fixture(root, 0)
            with self.assertRaises(TraceError):
                load_processes(root, ranks=[0, 1])
            with self.assertRaises(TraceError):
                load_processes(root, steps=[3, 4])
            data = journal.parent / 'launch-1.bin'
            data.write_bytes(STATE.pack(1, 1, 1, 0) + data.read_bytes()[16:])
            process = NativeProcess(journal)
            with self.assertRaises(RecordError):
                process.decode_launch(process.launches[0])

    def test_instruction_outside_capture_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            journal, rows = fixture(Path(temp), 0)
            for row in rows:
                if row['kind'] == 'capture_begin':
                    row['host_ns'] += 450000
            journal.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            process = NativeProcess(journal)
            with self.assertRaisesRegex(TraceError, 'outside'):
                process.decode_launch(process.launches[0])

    def test_journal_failures_and_clock_domain_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            journal, original = fixture(root, 0)
            variants = [original + [dict(kind='fatal', host_ns=10**15 + 9999999)],
                        [r for r in original if r['kind'] != 'finalized'],
                        [r for r in original if r['kind'] != 'drain'],
                        [dict(r, successful=False) if r['kind'] == 'capture_end' else r for r in original],
                        [dict(r, recorded_launches=2) if r['kind'] == 'capture_end' else r for r in original]]
            for rows in variants:
                journal.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                with self.assertRaises(TraceError):
                    NativeProcess(journal)
            journal.write_text(''.join(json.dumps(r) + '\n' for r in original))
            other, rows = fixture(root, 1)
            rows[0]['boot_id'] = 'different-boot'
            other.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with self.assertRaises(TraceError):
                load_processes(root)


def compare_warp_export(reference, directory, *, kernel_overview=None):
    """Independently check the ordered CSV subset and unchanged launch evidence."""
    from collections import Counter
    import hashlib
    import math

    reference, directory = Path(reference), Path(directory)
    before = json.loads((reference / 'validation.json').read_text())
    after = json.loads((directory / 'validation.json').read_text())
    assert after['instruction_export']['instruction_granularity'] == 'warp'
    assert after['instruction_export']['selection_stage'] == 'offline_export'
    assert before['ranks'] == after['ranks'] and before['steps'] == after['steps']
    original = json.loads((reference / 'coverage.json').read_text())
    exported = json.loads((directory / 'coverage.json').read_text())
    def raw_evidence(rows):
        return [{key: value for key, value in row.items() if key != 'instruction_export'} for row in rows]

    assert raw_evidence(original) == raw_evidence(exported)
    launches = {row['launch_id']: row for row in original if row.get('launch_id')}
    expected = json.loads(Path(kernel_overview or reference / 'communication.json').read_text())
    actual = json.loads((directory / 'communication.json').read_text())
    def kernels(document):
        return {row['args']['launch_id']: {key: value for key, value in row.items() if key != 'tid'}
                for row in document['traceEvents'] if row.get('cat', '').endswith('.kernel')}

    assert kernels(expected) == kernels(actual)
    source_counts, selected_counts, lane_counts = Counter(), Counter(), Counter()
    source_tracks, warp_tracks, completed = set(), set(), set()
    current, previous_thread, representatives = None, -1, {}
    with (reference / 'events.csv').open() as source, (directory / 'events.csv').open() as target:
        selected = csv.DictReader(target)
        for row in csv.DictReader(source):
            launch_id, thread = row['launch_id'], int(row['thread'])
            if launch_id != current:
                assert launch_id not in completed
                completed.add(launch_id)
                current, previous_thread, representatives = launch_id, -1, {}
            assert thread >= previous_thread
            previous_thread = thread
            width = math.prod(launches[launch_id]['block'])
            block_id, local_thread = divmod(thread, width)
            warp, lane = divmod(local_thread, 32)
            group = (block_id, warp, row['site_id'])
            representative = representatives.setdefault(group, thread)
            source_counts[row['library']] += 1
            lane_counts[f'{row["library"]}/lane{lane}'] += 1
            prefix = (row['pid'], row['library'], row['image'], row['kernel'], row['context'], row['stream'])
            source_tracks.add((*prefix, thread))
            if thread == representative:
                assert next(selected, None) == row, 'Selected record or timestamp changed'
                selected_counts[row['library']] += 1
                warp_tracks.add((*prefix, block_id, warp))
        assert next(selected, None) is None, 'Unexpected extra sampled records'
    for library, count in source_counts.items():
        assert count == before['counts'][library + '_intervals'] == after['counts'][library + '_input_intervals']
        assert selected_counts[library] == after['counts'][library + '_intervals']
    with (directory / 'timeline.json').open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    with gzip.open(directory / 'timeline.json.gz', 'rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
    files = {}
    for name in ('timeline.json', 'timeline.json.gz', 'events.csv'):
        old_size, new_size = (reference / name).stat().st_size, (directory / name).stat().st_size
        files[name] = {'thread_bytes': old_size, 'warp_bytes': new_size,
                       'reduction_percent': (1 - new_size / old_size) * 100}
    report = {
        'status': 'passed', 'reference': str(reference), 'directory': str(directory),
        'instruction_records': {'thread': sum(source_counts.values()), 'warp': sum(selected_counts.values())},
        'records_by_library': {'thread': dict(source_counts), 'warp': dict(selected_counts)},
        'instruction_tracks': {'thread': len(source_tracks), 'warp': len(warp_tracks)},
        'source_lane_records': dict(sorted(lane_counts.items())), 'files': files,
        'kernel_spans_unchanged': len(kernels(actual)), 'rank_steps': len(after['ranks']) * len(after['steps']),
        'selected_records_and_timestamps_unchanged': True, 'raw_launch_evidence_unchanged': True,
        'gzip_roundtrip_sha256': digest,
        'caveat': 'Offline representative sampling, not a new warp-sampled GPU capture. Repeats are retained. '
                  'JSON size also reflects integer track IDs and preserved overview role metadata; '
                  'CSV record counts isolate sampling reduction.',
    }
    with (directory / 'warp-comparison.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    return report


if __name__ == '__main__':
    if len(sys.argv) in (4, 5) and sys.argv[1] == '--compare-warp-export':
        print(json.dumps(compare_warp_export(sys.argv[2], sys.argv[3],
                                             kernel_overview=sys.argv[4] if len(sys.argv) == 5 else None), indent=2))
    else:
        unittest.main()
