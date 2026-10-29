"""Decode per-thread barriers and merge calibrated GPUs on a shared host timeline."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import json
from math import prod
from pathlib import Path
import struct

EVENT = struct.Struct('<QIIII')  # timestamp, site, kind, write count, reserved
FIELDS = ['rank', 'device', 'launch_id', 'kernel', 'block_id', 'thread_id', 'barrier_site',
          'enter_ns', 'exit_ns', 'elapsed_ns', 'sync_at_end', 'status', 'pid', 'device_uuid', 'phase',
          'stream', 'context_generation', 'calibration_id', 'session_id', 'enter_host_ns', 'exit_host_ns',
          'host_elapsed_ns', 'alignment_uncertainty_ns', 'event_index', 'opcode', 'site_type', 'ptx_line', 'scope',
          'warp_id', 'lane_id', 'thread_selection']


def selected_threads(manifest, threads):
    selection = manifest.get('thread_selection', 'thread0' if manifest.get('schema', 1) == 1 else None)
    if selection not in ('all', 'thread0'):
        raise ValueError('Missing or unsupported thread selection')
    return range(threads) if selection == 'all' else range(1)


class ClockMapping:
    """Affine calibration with empirical, not hardware-certified, uncertainty."""

    def __init__(self, data):
        self.data = data
        if data.get('schema') != 1 or data.get('complete') is not True:
            raise ValueError('Missing complete final clock calibration')
        if data.get('clock') != 'CLOCK_MONOTONIC_RAW':
            raise ValueError('Unsupported host clock')
        for key in ('session_id', 'hostname', 'boot_id', 'device_uuid'):
            if not isinstance(data.get(key), str) or not data[key]:
                raise ValueError(f'Invalid clock identity: {key}')
        for key in ('pid', 'rank', 'device', 'calibration_id', 'context_generation', 'clock_resolution_ns'):
            if type(data.get(key)) is not int or data[key] < (0 if key in ('rank', 'device') else 1):
                raise ValueError(f'Invalid clock field: {key}')
        previous_host, previous_gpu = -1, -1
        samples = []
        anchors = []
        for phase in ('start', 'end'):
            batch = data.get(phase)
            if not isinstance(batch, list) or len(batch) != 32:
                raise ValueError('Expected 32 samples in each clock calibration phase')
            for sample in batch:
                before, gpu, after = (sample[k] for k in ('host_before_ns', 'gpu_ns', 'host_after_ns'))
                if any(type(v) is not int or not 0 <= v < 2**64 for v in (before, gpu, after)):
                    raise ValueError('Invalid clock sample timestamps')
                if before >= after or before < previous_host or gpu <= previous_gpu:
                    raise ValueError('Clock discontinuity or unordered samples')
                previous_host, previous_gpu = after, gpu
                samples.append(sample)
            anchors.append(min(batch, key=lambda s: s['host_after_ns'] - s['host_before_ns']))
        self.start, self.end = anchors
        self.g0, self.g1 = (s['gpu_ns'] for s in anchors)
        self.h02, self.h12 = (s['host_before_ns'] + s['host_after_ns'] for s in anchors)
        self.dg = self.g1 - self.g0
        self.dh2 = self.h12 - self.h02
        if self.dg <= 0 or self.dh2 <= 0:
            raise ValueError('Nonpositive clock mapping scale')
        self.widths = [s['host_after_ns'] - s['host_before_ns'] for s in anchors]
        self.residual = max(max(s['host_before_ns'] - self.map_ns(s['gpu_ns']),
                                self.map_ns(s['gpu_ns']) - s['host_after_ns'], 0) for s in samples)
        if self.residual > 4 * max(self.widths):
            raise ValueError('Clock samples inconsistent with an affine clock model')
        self.domain = tuple(data[k] for k in ('clock', 'hostname', 'boot_id', 'session_id'))

    def validate_identity(self, meta):
        for key in ('pid', 'rank', 'device', 'device_uuid', 'calibration_id', 'context_generation', 'session_id'):
            if meta.get(key) != self.data[key]:
                raise ValueError(f'Launch/clock identity mismatch: {key}')

    def map_ns(self, gpu):
        return (self.h02 * self.dg + self.dh2 * (gpu - self.g0) + self.dg) // (2 * self.dg)

    def uncertainty_ns(self, gpu):
        if not self.g0 <= gpu <= self.g1:
            raise ValueError('Timestamp outside clock calibration anchors; extrapolation forbidden')
        width = self.widths[0] * (self.g1 - gpu) + self.widths[1] * (gpu - self.g0)
        return ((width + 2 * self.dg - 1) // (2 * self.dg) + self.residual +
                self.data['clock_resolution_ns'] + 1)

    def align(self, row):
        enter, exit_ = row['enter_ns'], row['exit_ns']
        uncertainty = max(self.uncertainty_ns(enter), self.uncertainty_ns(exit_))
        first, last = self.map_ns(enter), self.map_ns(exit_)
        return dict(enter_host_ns=first, exit_host_ns=last, host_elapsed_ns=last - first,
                    alignment_uncertainty_ns=uncertainty)

    def report(self):
        return dict(**{k: self.data[k] for k in ('pid', 'rank', 'device_uuid', 'calibration_id',
                    'context_generation', 'session_id', 'hostname', 'boot_id', 'clock')},
                    gpu_anchors_ns=[self.g0, self.g1], host_anchor_midpoints_ns=[self.h02 // 2, self.h12 // 2],
                    scale=self.dh2 / (2 * self.dg), drift_ppm=(self.dh2 - 2 * self.dg) * 1e6 / (2 * self.dg),
                    anchor_bracket_widths_ns=self.widths, max_residual_ns=self.residual,
                    discontinuity_residual_limit_ns=4 * max(self.widths),
                    max_uncertainty_ns=max(self.uncertainty_ns(self.g0), self.uncertainty_ns(self.g1)),
                    start_wall_ns=self.data.get('start_wall_ns', 0), end_wall_ns=self.data.get('end_wall_ns', 0),
                    uncertainty_model='Interpolated anchor half-widths + host resolution + rounding + observed residual; assumes affine clocks')


def decode_launch(meta, manifest, raw):
    return list(iter_decode_launch(meta, manifest, raw))


def iter_decode_launch(meta, manifest, raw):
    if manifest.get('mode') == 'ptx_sync':
        from neutrino.utils.ptx_sync_trace import iter_decode_launch as decode_sync
        yield from decode_sync(meta, manifest, raw)
        return
    for shape in (meta['grid'], meta['block']):
        if len(shape) != 3 or any(not isinstance(n, int) or n <= 0 for n in shape):
            raise ValueError('Invalid launch dimensions')
    blocks, threads = prod(meta['grid']), prod(meta['block'])
    if not manifest['sites'] or [s['id'] for s in manifest['sites']] != list(range(len(manifest['sites']))):
        raise ValueError('Site IDs must be unique contiguous slot indices')
    slots = manifest['slots']
    if slots != 2 * len(manifest['sites']) or manifest['record_bytes'] != EVENT.size:
        raise ValueError('Invalid map capacity or record ABI')
    if len(raw) != blocks * threads * slots * EVENT.size:
        raise ValueError('Truncated or oversized fixed thread map')
    selected = selected_threads(manifest, threads)
    for index in range(blocks * threads):
        block, thread = divmod(index, threads)
        base = index * slots * EVENT.size
        if thread not in selected:
            if any(raw[base:base + slots * EVENT.size]):
                raise ValueError('Nonrepresentative thread wrote a record')
            continue
        previous = None
        for site in manifest['sites']:
            events = [EVENT.unpack_from(raw, base + (2 * site['id'] + k) * EVENT.size) for k in (0, 1)]
            problems = []
            for k, event in enumerate(events):
                if event[3] != 1:
                    problems.append('missing' if not event[3] else 'duplicate')
                elif event[1:3] != (site['id'], k) or event[4]:
                    problems.append('identity_mismatch')
            enter, exit_ = events[0][0], events[1][0]
            if not problems and (exit_ < enter or (previous is not None and enter < previous)):
                problems.append('nonmonotonic')
            if blocks != site['blocks'] or threads != site['threads']:
                problems.append('launch_shape_mismatch')
            previous = exit_
            yield dict(rank=meta['rank'], device=meta['device'], launch_id=meta['launch_id'],
                       kernel=meta['kernel'], block_id=block, thread_id=thread, barrier_site=site['id'],
                       enter_ns=enter, exit_ns=exit_, elapsed_ns=exit_ - enter if not problems else '',
                       sync_at_end=site['sync_at_end'], status='|'.join(sorted(set(problems))) or 'ok',
                       pid=meta['pid'], device_uuid=meta['device_uuid'], phase=site['phase'],
                       stream=meta.get('stream', 'unknown'), context_generation=meta.get('context_generation', 0),
                       session_id=meta.get('session_id', 'legacy'), calibration_id=meta.get('calibration_id', 0))


class TraceWriter:
    def __init__(self, file):
        self.file, self.count = file, 0
        file.write('{"traceEvents":[')

    def emit(self, event):
        if self.count:
            self.file.write(',\n')
        self.file.write(json.dumps(event, separators=(',', ':'), allow_nan=False))
        self.count += 1

    def finish(self, metadata):
        self.file.write('],"displayTimeUnit":"ns","metadata":')
        self.file.write(json.dumps(metadata, allow_nan=False))
        self.file.write('}\n')


def process_key(meta):
    return (meta.get('session_id', 'legacy'), meta['pid'], meta['rank'], meta['device_uuid'])


def lane_key(row):
    return (process_key(row), row['context_generation'], row['stream'], row['block_id'], row['thread_id'])


def export(root, output, matrix=False, clock_align='auto', per_gpu=False):
    if clock_align not in ('auto', 'host', 'none'):
        raise ValueError('Unsupported clock alignment mode')
    output.mkdir(parents=True, exist_ok=True)
    launches, errors, models = [], [], {}
    launch_records = []
    seen, lanes = set(), {}
    warp_lanes = set()
    origins, counts = {}, defaultdict(Counter)
    pairs, invalid, aligned_launches = 0, 0, 0
    sync_opcodes, sync_types, max_thread_intervals = Counter(), Counter(), 0
    domain, common_origin = None, None
    for path in sorted(root.glob('*/result/barrier-*.json')):
        try:
            meta = json.loads(path.read_text())
            key = (process_key(meta), meta.get('context_generation', 0), meta['launch_id'])
            if key in seen:
                raise ValueError(f'Duplicate launch identity {key}')
            seen.add(key)
            attributes = {a['id']: bytes.fromhex(a['raw']) for a in meta['attributes']}
            manifest = json.loads(Path(meta['manifest']).read_text())
            if manifest.get('mode') != 'ptx_sync' and int.from_bytes(attributes.get(2, b'')[:4], 'little') != 1:
                raise ValueError('Missing cooperative launch attribute')
            if meta['kernel'] != manifest['kernel']:
                raise ValueError('Kernel/manifest mismatch')
            model = None
            if meta.get('clock_alignment', 'none') not in ('host', 'none'):
                raise ValueError('Unsupported launch clock alignment')
            if clock_align != 'none' and meta.get('clock_alignment') == 'host':
                calibration_id = meta['calibration_id']
                if type(calibration_id) is not int or calibration_id <= 0:
                    raise ValueError('Invalid calibration ID')
                calibration = path.parent / f'clock-{calibration_id}.json'
                if calibration not in models:
                    models[calibration] = ClockMapping(json.loads(calibration.read_text()))
                model = models[calibration]
                model.validate_identity(meta)
                if domain is not None and domain != model.domain:
                    raise ValueError('Mixed host/boot/session clock domains')
                domain = model.domain
            elif clock_align == 'host':
                raise ValueError('Missing host clock calibration for selected launch')
            # Pass one retains only launch metadata and unique lane identities.
            raw = path.with_suffix('.bin').read_bytes()
            intervals, first, last, opcodes = 0, None, None, Counter()
            for row in iter_decode_launch(meta, manifest, raw):
                pairs += 1
                intervals += 1
                first = row['enter_ns'] if first is None else min(first, row['enter_ns'])
                last = row['exit_ns'] if last is None else max(last, row['exit_ns'])
                opcodes[row.get('opcode', 'barrier')] += 1
                if row['status'] != 'ok':
                    invalid += 1
                    continue
                if model:
                    aligned = model.align(row)
                    start = aligned['enter_host_ns']
                    common_origin = start if common_origin is None else min(common_origin, start)
                else:
                    gpu = row['device_uuid']
                    origins[gpu] = min(origins.get(gpu, row['enter_ns']), row['enter_ns'])
                if manifest.get('mode') == 'ptx_sync':
                    sync_opcodes[row['opcode']] += 1
                    sync_types[row['site_type']] += 1
                    max_thread_intervals = max(max_thread_intervals, row['event_index'] + 1)
                lane = lane_key(row)
                if row.get('thread_selection') == 'warp':
                    warp_lanes.add(lane)
                if lane not in lanes:
                    lanes[lane] = len(lanes)
            launches.append((path, meta, manifest, model))
            launch_records.append(dict(rank=meta['rank'], pid=meta['pid'], device_uuid=meta['device_uuid'],
                launch_id=meta['launch_id'], session_id=meta.get('session_id', 'legacy'),
                context_generation=meta.get('context_generation', 0),
                kernel=meta['kernel'], intervals=intervals, raw_bytes=len(raw),
                first_enter_ns=first, last_exit_ns=last, opcode_intervals=dict(opcodes)))
            counts[meta['rank']][manifest['family']] += 1
            aligned_launches += model is not None
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f'{path}: {error}')
    if clock_align != 'none':
        unreferenced = set(root.glob('*/result/clock-*.json')) - models.keys()
        if (aligned_launches and aligned_launches != len(launches)) or unreferenced:
            errors.append('Partial clock calibration or unreferenced sidecars; refusing an unaligned fallback')
    for path in root.glob('*/result/barrier-*.bin'):
        if not path.with_suffix('.json').is_file():
            errors.append(f'Raw trace without launch metadata: {path}')
    if not launches:
        errors.append('Zero instrumentation coverage')
    for path in root.glob('*/event.log'):
        if '[barrier-error]' in path.read_text(errors='replace'):
            errors.append(f'Hook reported failure: {path}')
    if matrix:
        if set(counts) != {0, 1}:
            errors.append('Expected exactly ranks 0 and 1')
        rank_devices = {rank: {m['device_uuid'] for _, m, _, _ in launches if m['rank'] == rank} for rank in (0, 1)}
        if any(len(devices) != 1 for devices in rank_devices.values()) or rank_devices[0] == rank_devices[1]:
            errors.append('Expected one distinct GPU UUID per rank')
        # Pinned unmodified test_ep.py, 36 modes, 5 dispatches and 2 combines
        # per mode, --skip-perf-test, default nondeterministic correctness path.
        for rank in (0, 1):
            if counts[rank]['dispatch'] != 180 or counts[rank]['combine'] != 72 or counts[rank]['barrier'] != 1:
                errors.append(f'Incomplete matrix launch coverage for rank {rank}: {dict(counts[rank])}')
    if invalid:
        errors.append(f'{invalid} invalid block/thread/site pairs')
    aligned = bool(aligned_launches) and not errors
    reports = [model.report() for model in models.values()]
    process_ids = {key: i for i, key in enumerate(sorted({process_key(m) for _, m, _, _ in launches},
                   key=lambda k: (k[2], k[3], k[1], k[0])))}
    warp_recording = any(s.get('thread_selection') == 'warp' for _, _, s, _ in launches)
    summary = dict(launches=len(launches), launch_records=launch_records,
        block_site_pairs=sum(prod(m['grid']) * len(s['sites']) for _, m, s, _ in launches),
        thread_site_pairs=pairs, recording_unit_intervals=pairs, valid_pairs=pairs - invalid,
        thread_selections=sorted({s.get('thread_selection', 'thread0') for _, _, s, _ in launches}),
        launches_per_rank={str(k): dict(v) for k, v in counts.items()},
        rank_gpu_groups=len(process_ids), tracks=len(lanes),
        sync_opcode_intervals=dict(sync_opcodes), sync_interval_types=dict(sync_types),
        max_intervals_per_recording_unit=max_thread_intervals,
        max_intervals_per_thread=max_thread_intervals if not warp_recording else None,
        clocks='host-aligned (estimated uncertainty)' if aligned else 'unsynchronized; independent per-GPU display origins',
        common_origin_host_ns=common_origin if aligned else None,
        gpu_origins_ns={} if aligned else origins, calibrations=reports,
        cross_gpu_uncertainty=('Sum of the two per-event uncertainties; ordering inside this interval is unresolved'
                               if aligned else 'Unavailable: independent GPU clocks must not be compared'),
        measurement=('per-warp lane-0 PTX synchronization intervals, not whole-warp or pure communication latency'
                     if warp_recording else 'per-thread PTX synchronization intervals, not pure communication latency'
                     if any(s.get('mode') == 'ptx_sync' for _, _, s, _ in launches)
                     else 'per-thread barrier residence time, not pure communication latency'),
        probe_limitations=sorted({item for _, _, s, _ in launches for item in s.get('limitations', [])}),
        limitations=['instrumentation and readback overhead',
                     'affine-clock assumption; uncertainty is empirical, not a hardware guarantee' if aligned else 'no cross-GPU clock alignment'],
        status='failed' if errors else 'passed', errors=errors)
    with ExitStack() as stack:
        csv_file = stack.enter_context((output / 'barriers.csv').open('w', newline=''))
        csv_writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        csv_writer.writeheader()
        main = TraceWriter(stack.enter_context((output / 'timeline.json').open('w')))
        gpu_writers = {gpu: TraceWriter(stack.enter_context((output / f'timeline-gpu-{gpu}.json').open('w')))
                       for gpu in sorted({k[3] for k in process_ids})} if per_gpu else {}

        def emit(event, gpu):
            main.emit(event)
            if gpu in gpu_writers:
                gpu_writers[gpu].emit(event)

        if not errors:
            for key, pid in process_ids.items():
                session, original_pid, rank, gpu = key
                label = 'host-aligned' if aligned else 'UNSYNCHRONIZED'
                emit(dict(ph='M', name='process_name', pid=pid, tid=0,
                          args=dict(name=f'Rank {rank} / GPU {gpu} / PID {original_pid} — {label}')), gpu)
                emit(dict(ph='M', name='process_sort_index', pid=pid, tid=0, args=dict(sort_index=pid)), gpu)
            for lane, tid in lanes.items():
                process, context, stream, block, thread = lane
                unit = f'warp {thread // 32} / lane 0' if lane in warp_lanes else f'thread {thread}'
                emit(dict(ph='M', name='thread_name', pid=process_ids[process], tid=tid,
                          args=dict(name=f'Context {context} / stream {stream} / block {block} / {unit}')), process[3])
            for path, meta, manifest, model in launches:
                for row in iter_decode_launch(meta, manifest, path.with_suffix('.bin').read_bytes()):
                    if model:
                        row.update(model.align(row))
                        ts = (row['enter_host_ns'] - common_origin) / 1000
                        duration = row['host_elapsed_ns'] / 1000
                    else:
                        ts = (row['enter_ns'] - origins[row['device_uuid']]) / 1000
                        duration = row['elapsed_ns'] / 1000
                    csv_writer.writerow(row)
                    # Strings retain uint64 precision in JavaScript event arguments.
                    event_args = {k: str(v) if k.endswith('_ns') and type(v) is int else v for k, v in row.items()}
                    generic = manifest.get('mode') == 'ptx_sync'
                    emit(dict(ph='X', cat=row['phase'] if generic else 'barrier residence time',
                              name=(f"{row['opcode']} / {row['site_type']} / site {row['barrier_site']}"
                                    if generic else f"{row['phase']} barrier"),
                              pid=process_ids[process_key(row)], tid=lanes[lane_key(row)],
                              ts=ts, dur=duration, args=event_args), row['device_uuid'])
        main.finish(summary)
        for writer in gpu_writers.values():
            writer.finish(summary)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tracedir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--matrix', action='store_true')
    parser.add_argument('--clock-align', choices=('auto', 'host', 'none'), default='auto')
    parser.add_argument('--per-gpu', action='store_true', help='Also write per-GPU timeline copies')
    args = parser.parse_args()
    summary = export(args.tracedir, args.output or args.tracedir / 'decoded', args.matrix,
                     args.clock_align, args.per_gpu)
    print(json.dumps(summary, indent=2))
    if summary['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
