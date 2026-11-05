"""GPU capture selection, compact coordinates, policy parity, and legacy decoding."""
import ctypes as C
import json
from math import prod
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neutrino.sys_sync.sampling import MODES, CONTEXT_ABI, validate_policy, policy_json, policy_hash, resolve_policy, glob_match
from neutrino.sys_sync.records import BufferLayout, CONTEXT, RECORD, STATE, RecordError, decode_native
from neutrino.sys_sync.export import NativeProcess, export_trace
from test_sys_records import launch
from test_sys_loops import rewrite, execute
from test_sys_export import fixture, add_activity

POLICY = {'schema': 1, 'default': 'warp', 'rules': [{'kernel_glob': '*', 'library': 'NCCL', 'sampling': 'block'}]}


class SamplingTests(unittest.TestCase):
    def test_policy_default_precedence_and_provenance(self):
        self.assertEqual(validate_policy(), {'schema': 1, 'default': 'warp', 'rules': []})
        self.assertEqual(resolve_policy(POLICY, '_ZncclKernel', 'NCCL'), ('block', 0))
        self.assertEqual(resolve_policy(POLICY, '_Zdispatch_ncclDevComm', 'DeepEP'), ('warp', -1))
        specific = dict(POLICY, rules=[{'kernel_glob': '*AllGather*', 'sampling': 'all'}, *POLICY['rules']])
        self.assertEqual(resolve_policy(specific, '_ZAllGather', 'NCCL'), ('all', 0))
        self.assertEqual(resolve_policy(specific, '_Zallgather', 'NCCL'), ('block', 1))
        self.assertTrue(glob_match('a?*[x]', 'abzz[x]'))
        self.assertFalse(glob_match('a?*[x]', 'abzzx'))
        self.assertTrue(glob_match('*', ''))
        self.assertFalse(glob_match('?', ''))
        self.assertEqual(policy_hash({'schema': 1}), policy_hash())

    def test_invalid_policies(self):
        invalid = [None, [], {}, {'schema': True}, {'schema': 1.0}, {'schema': 2}, {'schema': 1, 'extra': 2},
                   {'schema': 1, 'default': None}, {'schema': 1, 'default': []}, {'schema': 1, 'rules': {}},
                   {'schema': 1, 'rules': [{'kernel_glob': '*', 'sampling': 'lane'}]},
                   {'schema': 1, 'rules': [{'kernel_glob': '', 'sampling': 'warp'}]},
                   {'schema': 1, 'rules': [{'kernel_glob': '*', 'sampling': 'all', 'library': 3}]},
                   {'schema': 1, 'rules': [{'kernel_glob': 'a\n', 'sampling': 'warp'}]},
                   {'schema': 1, 'rules': [{'kernel_glob': '*', 'sampling': 'warp', 'unexpected': 1}]}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_policy(value)

    def test_compact_mapping_and_strict_decode(self):
        for grid, block in [((2, 3, 2), (7, 5, 2)), ((3, 1, 1), (35, 1, 1)), ((2, 1, 1), (7, 1, 1))]:
            width = prod(block)
            for mode in MODES:
                layout = BufferLayout.from_dimensions(grid, block, 2, mode)
                expected = [t for t in range(prod(grid) * width)
                            if mode == 'all' or (t % width) % (32 if mode == 'warp' else width) == 0]
                self.assertEqual(layout.logical_threads, prod(grid) * width)
                self.assertEqual([layout.physical_thread(i) for i in range(layout.threads)], expected)
                data = STATE.pack(1, 0, 1, 0) * layout.threads + RECORD.pack(10, 20, 1, 0, 2) * layout.threads
                events = decode_native(data, launch=launch(), layout=layout, sites=[1], raw_format='states-packed-v2')
                self.assertEqual([e.thread for e in events], expected)
                for bad in (STATE.pack(1, 1, 1, 0) + data[16:], STATE.pack(1, 0, 0, 0) + data[16:], data[:-1], data + b'!'):
                    with self.assertRaises(RecordError):
                        decode_native(bad, launch=launch(), layout=layout, sites=[1], raw_format='states-packed-v2')
                if mode != 'all':
                    with self.assertRaises(RecordError):
                        decode_native(data, launch=launch(), layout=layout, sites=[1], raw_format='states-packed-v1')
        for mode in (None, True, [], 'unknown'):
            with self.assertRaises(RecordError):
                BufferLayout.from_dimensions((1, 1, 1), (32, 1, 1), sampling=mode)

    def test_emitted_gating_unsampled_zero_timers_writes(self):
        for storage in ('per-thread-v1', 'pooled-v1'):
            result = rewrite('membar.sys; ret;', storage, 3)
            for width in (7, 35, 70):
                for mode, number in MODES.items():
                    for local in (0, 1, min(32, width - 1), width - 1):
                        initial = {'%ntid.x': width, '%tid.x': local, '%nctaid.x': 2, '%ctaid.x': 1}
                        _, timers, regs = execute(result, initial=initial, sampling=number)
                        selected = mode == 'all' or local % (32 if mode == 'warp' else width) == 0
                        self.assertEqual(len(timers), 2 if selected else 0)
                        memory = regs['__memory']
                        if selected:
                            layout = BufferLayout.from_dimensions((2, 1, 1), (width, 1, 1), sampling=mode)
                            slot = layout.slots_per_block + (local // 32 if mode == 'warp' else local if mode == 'all' else 0)
                            self.assertEqual(memory[1000 + slot * 16 + 8], 1)
                            self.assertEqual(regs['%__neutrino_sys_probe_thread'], width + local)
                            if storage == 'pooled-v1':
                                self.assertEqual(memory[10000 + 32], width + local)
                        else:
                            self.assertEqual({key for key in memory if key >= 1000}, {1000, 1004, 1008, 1012, 2000})
                            self.assertTrue(all(memory[key] == 0 for key in (1000, 1004, 1008, 1012, 2000)))
                        self.assertEqual(execute(result, enabled=False, initial=initial, sampling=number)[1], [])
            with self.assertRaisesRegex(AssertionError, 'trap'):
                execute(result, sampling=99)
            skip = rewrite('.reg .pred %p; setp.eq.u32 %p, 0, 1; @%p membar.sys; ret;', storage)
            self.assertEqual(execute(skip, sampling=1)[1], [])

    def test_mixed_export_and_tampered_policy_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rank, mode in enumerate(('block', 'warp')):
                journal, rows = fixture(root, rank)
                image_path = next(journal.parent.glob('*.image.json'))
                image = json.loads(image_path.read_text())
                image['context_abi'] = CONTEXT_ABI
                image['probe'].update(context_abi=CONTEXT_ABI, sampling='launch_policy')
                image_path.write_text(json.dumps(image))
                process = next(r for r in rows if r['kind'] == 'process')
                process.update(context_abi=CONTEXT_ABI, capture_policy=POLICY, capture_policy_sha256=policy_hash(POLICY))
                row = next(r for r in rows if r['kind'] == 'launch')
                layout = BufferLayout.from_dimensions((2, 1, 1), (35, 1, 1), 8, mode)
                row.update(grid=[2, 1, 1], block=[35, 1, 1], threads=layout.threads, recorded_threads=layout.threads,
                           logical_threads=70, raw_format='states-packed-v2', context_abi=CONTEXT_ABI,
                           capture_sampling=mode, capture_policy_rule=0 if rank == 0 else -1,
                           capture_policy_sha256=policy_hash(POLICY))
                tick = 2**60 + rank * 999999999 + 400000
                data = STATE.pack(1, 0, 1, 0) * layout.threads + RECORD.pack(tick, tick + 100, 1, 0, 2) * layout.threads
                (journal.parent / row['raw']).write_bytes(data)
                next(r for r in rows if r['kind'] == 'drain').update(records=layout.threads, raw_bytes=len(data))
                activity, _ = add_activity(journal, rows)
                activity.update(grid=row['grid'], block=row['block'])
                journal.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            report = export_trace(root, root / 'decoded', require_kernel_activity=True, instruction_granularity='captured')
            self.assertEqual(report['counts']['NCCL_intervals'], 2)
            self.assertEqual(report['counts']['DeepEP_intervals'], 4)
            self.assertFalse(report['instruction_export']['all_threads_complete'])
            self.assertEqual(report['counts']['NCCL_filtered_intervals'], 0)
            document = json.loads((root / 'decoded/timeline.json').read_text())
            self.assertTrue(all(type(e['tid']) is int for e in document['traceEvents']))
            details = [e for e in document['traceEvents'] if e.get('cat', '').endswith('.sys_instruction')]
            self.assertEqual([e['args']['thread'] for e in details], [0, 35, 0, 32, 35, 67])
            row['capture_sampling'] = 'block'
            journal.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            process = NativeProcess(journal)
            with self.assertRaisesRegex(ValueError, 'sampling policy'):
                process.decode_launch(process.launches[0])

    @unittest.skipUnless(os.environ.get('UNIFIED_TOOLCHAIN'), 'Selected compiler required')
    def test_timestamp_sass_sampling_gate(self):
        import re
        from test_sys_probe import compile_fixture
        for storage in ('per-thread-v1', 'pooled-v1'):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                compile_fixture(root, 'loop-boundary', 3, storage)
                sass = (root / 'probed/linked.sass').read_text()
                clocks = [line for line in sass.splitlines() if 'SR_GLOBALTIMER' in line]
                self.assertTrue(clocks)
                self.assertTrue(all(re.search(r'@!?P\d+\s+CS2R', line) for line in clocks), clocks)

    def test_old_or_unknown_context_abi_rejected(self):
        from neutrino.sys_sync.runtime import image_manifest
        from neutrino.sys_sync.ptx import PTXError
        for value in ({}, {'context_abi': 1}, {'context_abi': 99}, {'context_abi': True}):
            with self.assertRaisesRegex(PTXError, 'context ABI'):
                image_manifest('unused-original', 'unused-replacement', value, 'fixture')

    @unittest.skipUnless(os.environ.get('UNIFIED_TOOLCHAIN'), 'Selected C++ compiler required')
    def test_native_policy_validation_and_matching_parity(self):
        source = r'''
#include "sys_sampling.hpp"
#include <iostream>
int main() {
    using namespace neutrino::sys; using J = nlohmann::json;
    J input; std::cin >> input; J output = J::array();
    for (const auto& row : input) {
        try {
            auto policy = validate_policy(row.at("policy"));
            auto result = resolve_policy(policy, row.at("kernel"), row.at("library"));
            output.push_back({policy, result.first, result.second});
        } catch (const std::exception&) { output.push_back("invalid"); }
    }
    std::cout << output.dump();
}
'''
        policies = [None, POLICY, {'schema': 1}, {'schema': 1, 'default': 'all'}, {'schema': True},
                    {'schema': 1, 'rules': [{'kernel_glob': 'a?*[x]', 'sampling': 'block'}]},
                    {'schema': 1, 'rules': [{'kernel_glob': '', 'sampling': 'warp'}]},
                    {'schema': 1, 'default': []}, {'schema': 1, 'rules': [{'kernel_glob': '*', 'sampling': 'warp', 'bad': 1}]}]
        rows = [{'policy': p, 'kernel': k, 'library': lib} for p in policies
                for k in ('_ZncclDev', 'abzz[x]', 'abzzx', '') for lib in ('NCCL', 'DeepEP')]
        expected = []
        for row in rows:
            try:
                expected.append([validate_policy(row['policy']), *resolve_policy(row['policy'], row['kernel'], row['library'])])
            except ValueError:
                expected.append('invalid')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'check.cpp').write_text(source)
            workspace = Path(__file__).resolve().parents[2]
            command = [os.environ['CXX'], '-std=c++17', '-Wall', '-Wextra', '-Werror',
                       '-I' + str(workspace / 'neutrino/neutrino/src'),
                       '-I' + str(workspace / 'third_party/nccl-unified/contrib/nccl_m2n/third_party'),
                       '-I' + os.environ['UNIFIED_TOOLCHAIN'] + '/targets/x86_64-linux/include',
                       str(root / 'check.cpp'), '-o', str(root / 'check')]
            subprocess.run(command, check=True, capture_output=True)
            actual = subprocess.check_output([str(root / 'check')], input=json.dumps(rows), text=True)
            self.assertEqual(json.loads(actual), expected)


def gpu_worker(root):
    import test_sys_probe as probe
    setup = json.loads((root / 'launch.json').read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES'] == setup['uuid']
    results = []
    for storage, build in setup['builds'].items():
        directory = root / storage
        driver = probe.FixtureDriver(build['inputs']['driver'], setup['uuid'], directory / build['native_bridge']['path'])
        try:
            baseline = driver.load(directory / build['binaries']['baseline']['path'])
            instrumented = driver.load(directory / build['binaries']['probed']['path'])
            sites = build['probe']['sites']
            stream = driver.stream()
            for block in ((7, 1, 1), (35, 1, 1), (7, 5, 2), (32, 1, 1)):
                probe.GRID, probe.BLOCK = (2, 2, 1), block
                for mode in MODES:
                    layout = BufferLayout.from_dimensions(probe.GRID, block, 32, mode)
                    total = layout.logical_threads
                    for entry, enabled in (('dispatch', True), ('no_sites', True), ('dispatch', False)):
                        out = driver.allocate(total * 4)
                        pooled = storage == 'pooled-v1'
                        capacity = layout.threads * 32 if pooled else 32
                        record_bytes = capacity * 40 if pooled else layout.record_bytes
                        context = driver.allocate(CONTEXT.size + layout.state_bytes + 8 + record_bytes)
                        states = context + CONTEXT.size
                        counter = states + layout.state_bytes
                        records = counter + (8 if pooled else 0)
                        driver.write(context, CONTEXT.pack(records, states, 1, layout.threads, capacity,
                                                          4 if pooled else 3, counter if pooled else 0, MODES[mode]))
                        blob = bytearray(4096)
                        struct.pack_into('<Q', blob, 0, out)
                        struct.pack_into('<I', blob, 4092, 20)
                        args = [C.create_string_buffer(bytes(blob), 4096)] if entry == 'dispatch' else []
                        driver.launch(baseline, entry, stream, args, 'ordinary', False)
                        driver.call('cuCtxSynchronize')
                        expected = driver.read(out, total * 4)
                        driver.write(out, bytes(total * 4))
                        driver.launch(instrumented, entry, stream, [*args, C.c_uint64(context if enabled else 0)], 'ordinary', True)
                        driver.call('cuCtxSynchronize')
                        assert driver.read(out, total * 4) == expected
                        state_bytes = driver.read(states, layout.state_bytes)
                        if not enabled:
                            assert not any(driver.read(states, layout.state_bytes + 8 + record_bytes))
                            continue
                        if pooled:
                            count = struct.unpack('<I', driver.read(counter, 4))[0]
                            bins = {layout.physical_thread(i): [] for i in range(layout.threads)}
                            payload = driver.read(records, count * 40) if count else b''
                            for i in range(count):
                                physical = struct.unpack_from('<Q', payload, i * 40 + 32)[0]
                                assert physical in bins
                                bins[physical].append(payload[i * 40:i * 40 + 32])
                            data = state_bytes + b''.join(b''.join(bins[layout.physical_thread(i)]) for i in range(layout.threads))
                            raw_format = 'states-packed-v2'
                        else:
                            data = state_bytes + driver.read(records, record_bytes)
                            raw_format = 'states-fixed-v2'
                        events = decode_native(data, launch=launch(), layout=BufferLayout(layout.threads, capacity, mode, prod(block)),
                                               sites=[s['ordinal'] for s in sites], raw_format=raw_format,
                                               loop_sites=[s['ordinal'] for s in sites if s.get('interval_kind') == 'loop'])
                        observed = {}
                        for event in events:
                            observed.setdefault(event.thread, []).append(event.site)
                        for slot in range(layout.threads):
                            physical = layout.physical_thread(slot)
                            expected_sites = probe.expected_sites(sites, physical, False, 'loop-boundary', 3) if entry == 'dispatch' else []
                            assert observed.get(physical, []) == expected_sites
                        results.append({'storage': storage, 'sampling': mode, 'block': block, 'entry': entry,
                                        'logical_threads': total, 'recorded_threads': layout.threads, 'intervals': len(events)})
            calls = struct.unpack('<I', driver.read(driver.symbol(instrumented, 'invocations', 4), 4))[0]
            assert calls == 24
        finally:
            driver.close()
    (root / 'validation.json').write_text(json.dumps({'status': 'passed', 'cases': results,
        'null_context_verified': True, 'exactly_once_verified': True, 'numerical_match': True}, indent=2) + '\n')


def gpu_main():
    import argparse
    from test_sys_probe import compile_fixture
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--allocation-confirmed', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        return gpu_worker(args.output)
    if not args.execute or not args.allocation_confirmed:
        parser.error('GPU fixtures require execution and confirmed allocation')
    workspace = Path(__file__).resolve().parents[2]
    root = args.output.resolve()
    if not root.is_relative_to(workspace / 'artifacts/unified-sys'):
        parser.error('Use an artifact directory')
    root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(workspace / 'neutrino/examples/moe_training'))
    from gpu_guard import launch_lock, require_idle, check_once
    from launch import require_clean_baseline
    require_clean_baseline(os.environ)
    builds = {}
    for storage in ('per-thread-v1', 'pooled-v1'):
        directory = root / storage
        directory.mkdir()
        builds[storage] = compile_fixture(directory, 'loop-boundary', 3, storage)
    with launch_lock() as lock:
        devices = require_idle()
        (root / 'launch.json').write_text(json.dumps({'uuid': devices[0]['uuid'], 'builds': builds}, indent=2))
        assert [(d['index'], d['uuid']) for d in check_once()] == [(d['index'], d['uuid']) for d in devices]
        with (root / 'console.log').open('x') as log:
            subprocess.run([sys.executable, __file__, '--gpu', '--worker', '--output', str(root)],
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': devices[0]['uuid']},
                           check=True, stdout=log, stderr=subprocess.STDOUT, timeout=180, pass_fds=(lock.fileno(),))
    print((root / 'validation.json').read_text())


if __name__ == '__main__':
    if '--gpu' in sys.argv:
        gpu_main()
    else:
        unittest.main()
