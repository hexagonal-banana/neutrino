"""Decode fixed-thread-map barrier records; clocks are never aligned across GPUs."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from math import prod
from pathlib import Path
import struct

EVENT = struct.Struct('<QIIII')  # timestamp, site, kind, write count, reserved
FIELDS = ['rank', 'device', 'launch_id', 'kernel', 'block_id', 'barrier_site',
          'enter_ns', 'exit_ns', 'elapsed_ns', 'sync_at_end', 'status', 'pid', 'device_uuid', 'phase']


def decode_launch(meta, manifest, raw):
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
    rows = []
    for block in range(blocks):
        base = block * threads * slots * EVENT.size
        if any(raw[base + slots * EVENT.size:base + threads * slots * EVENT.size]):
            raise ValueError('Nonrepresentative thread wrote a record')
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
            rows.append(dict(rank=meta['rank'], device=meta['device'], launch_id=meta['launch_id'],
                             kernel=meta['kernel'], block_id=block, barrier_site=site['id'],
                             enter_ns=enter, exit_ns=exit_, elapsed_ns=exit_ - enter if not problems else '',
                             sync_at_end=site['sync_at_end'], status='|'.join(sorted(set(problems))) or 'ok',
                             pid=meta['pid'], device_uuid=meta['device_uuid'], phase=site['phase']))
    return rows


def export(root, output, matrix=False):
    output.mkdir(parents=True, exist_ok=True)
    rows, launches, errors = [], [], []
    seen = set()
    for path in sorted(root.glob('*/result/barrier-*.json')):
        try:
            meta = json.loads(path.read_text())
            key = (meta['pid'], meta['device_uuid'], meta['launch_id'])
            if key in seen:
                raise ValueError(f'Duplicate launch identity {key}')
            seen.add(key)
            attributes = {a['id']: bytes.fromhex(a['raw']) for a in meta['attributes']}
            if int.from_bytes(attributes.get(2, b'')[:4], 'little') != 1:
                raise ValueError('Missing cooperative launch attribute')
            manifest = json.loads(Path(meta['manifest']).read_text())
            if meta['kernel'] != manifest['kernel']:
                raise ValueError('Kernel/manifest mismatch')
            launches.append((meta, manifest))
            rows.extend(decode_launch(meta, manifest, path.with_suffix('.bin').read_bytes()))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f'{path}: {error}')
    for path in root.glob('*/result/barrier-*.bin'):
        if not path.with_suffix('.json').is_file():
            errors.append(f'Raw trace without launch metadata: {path}')
    if not launches:
        errors.append('Zero instrumentation coverage')
    for path in root.glob('*/event.log'):
        if '[barrier-error]' in path.read_text(errors='replace'):
            errors.append(f'Hook reported failure: {path}')
    counts = defaultdict(Counter)
    for meta, manifest in launches:
        counts[meta['rank']][manifest['family']] += 1
    if matrix:
        if set(counts) != {0, 1}:
            errors.append('Expected exactly ranks 0 and 1')
        rank_devices = {rank: {m['device_uuid'] for m, _ in launches if m['rank'] == rank} for rank in (0, 1)}
        if any(len(devices) != 1 for devices in rank_devices.values()) or rank_devices[0] == rank_devices[1]:
            errors.append('Expected one distinct GPU UUID per rank')
        # Pinned unmodified test_ep.py, 36 modes, 5 dispatches and 2 combines
        # per mode, --skip-perf-test, default nondeterministic correctness path.
        for rank in (0, 1):
            if counts[rank]['dispatch'] != 180 or counts[rank]['combine'] != 72 or counts[rank]['barrier'] != 1:
                errors.append(f'Incomplete matrix launch coverage for rank {rank}: {dict(counts[rank])}')
    invalid = sum(row['status'] != 'ok' for row in rows)
    if invalid:
        errors.append(f'{invalid} invalid block/site pairs')
    with (output / 'barriers.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    origins = {}
    for row in rows:
        if row['status'] == 'ok':
            gpu = row['device_uuid']
            origins[gpu] = min(origins.get(gpu, row['enter_ns']), row['enter_ns'])
    gpu_ids = {gpu: i for i, gpu in enumerate(sorted(origins))}
    events = []
    for gpu, pid in gpu_ids.items():
        events.append(dict(ph='M', name='process_name', pid=pid, tid=0,
                           args=dict(name=f'GPU {gpu} — independent clock, unsynchronized')))
    lanes = set()
    for row in rows:
        if row['status'] != 'ok':
            continue
        gpu, block = row['device_uuid'], row['block_id']
        pid = gpu_ids[gpu]
        if (gpu, block) not in lanes:
            lanes.add((gpu, block))
            events.append(dict(ph='M', name='thread_name', pid=pid, tid=block, args=dict(name=f'Block {block} / thread 0')))
        events.append(dict(ph='X', cat='barrier residence time', name=f"{row['phase']} barrier",
                           pid=pid, tid=block, ts=(row['enter_ns'] - origins[gpu]) / 1000,
                           dur=row['elapsed_ns'] / 1000, args=row))
    (output / 'timeline.json').write_text(json.dumps(dict(traceEvents=events, displayTimeUnit='ns',
        metadata=dict(clocks='unsynchronized', gpu_origins_ns=origins, measurement='representative thread barrier residence time'))))
    for gpu, pid in gpu_ids.items():
        (output / f'timeline-gpu-{gpu}.json').write_text(json.dumps(dict(
            traceEvents=[e for e in events if e['pid'] == pid], displayTimeUnit='ns')))
    summary = dict(launches=len(launches), block_site_pairs=len(rows), valid_pairs=len(rows) - invalid,
                   launches_per_rank={str(k): dict(v) for k, v in counts.items()},
                   clocks='unsynchronized; independent per-GPU display origins',
                   measurement='thread 0 barrier residence time, not pure NVLink latency',
                   limitations=['instrumentation and readback overhead', 'no cross-GPU clock alignment'],
                   status='failed' if errors else 'passed', errors=errors)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tracedir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--matrix', action='store_true')
    args = parser.parse_args()
    summary = export(args.tracedir, args.output or args.tracedir / 'decoded', args.matrix)
    print(json.dumps(summary, indent=2))
    if summary['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
