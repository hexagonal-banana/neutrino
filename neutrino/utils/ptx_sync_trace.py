"""Validate bounded PTX synchronization intervals for warp or all-thread recording."""
from math import prod
import struct

HEADER = struct.Struct('<IIIIQ')  # records, sticky overflow, launches, completions, cumulative records
INTERVAL = struct.Struct('<QQII')  # entry, exit, static site ID, dynamic ordinal


def records_per_block(meta, manifest):
    threads = prod(meta['block'])
    return (threads + 31) // 32 if manifest.get('thread_selection') == 'warp' else threads


def headers(meta, manifest, raw, executions=1):
    for shape in (meta['grid'], meta['block']):
        if len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError('Invalid launch dimensions')
    capacity = manifest['capacity']
    selection = manifest.get('thread_selection')
    if (manifest.get('mode') != 'ptx_sync' or
            (manifest.get('schema'), selection) not in ((3, 'all'), (4, 'warp')) or
            type(capacity) is not int or not 1 <= capacity <= 65536 or
            manifest['slots'] != capacity + 1 or manifest['record_bytes'] != 24):
        raise ValueError('Invalid PTX synchronization ABI')
    if selection == 'warp' and (manifest.get('record_group') != 32 or manifest.get('representative_lane') != 0):
        raise ValueError('Invalid warp sampling ABI')
    sites = manifest['sites']
    if not sites or [s['id'] for s in sites] != list(range(len(sites))):
        raise ValueError('Invalid synchronization site IDs')
    stride = (capacity + 1) * 24
    if len(raw) != prod(meta['grid']) * records_per_block(meta, manifest) * stride:
        raise ValueError('Truncated or oversized synchronization buffer')
    for base in range(0, len(raw), stride):
        count, overflow, started, finished, total = HEADER.unpack_from(raw, base)
        if overflow or count > capacity:
            raise ValueError('Synchronization buffer overflow; increase --sync-capacity')
        if started != executions or finished != executions:
            raise ValueError('Missing, duplicated, or incomplete per-recording-unit execution')
        if not count <= total <= executions * capacity:
            raise ValueError('Invalid cumulative interval count')
        if executions == 1 and total != count:
            raise ValueError('Invalid interval total')
        yield base, count, total


def iter_decode_launch(meta, manifest, raw, executions=1):
    units = records_per_block(meta, manifest)
    warp = manifest.get('thread_selection') == 'warp'
    for index, (base, count, _) in enumerate(headers(meta, manifest, raw, executions)):
        block, unit = divmod(index, units)
        thread = unit * 32 if warp else unit
        previous = None
        for ordinal in range(count):
            enter, exit_, sid, sequence = INTERVAL.unpack_from(raw, base + 24 * (ordinal + 1))
            if sid >= len(manifest['sites']) or sequence != ordinal:
                raise ValueError('Invalid site or dynamic interval identity')
            if not enter or exit_ < enter or (previous is not None and enter < previous):
                raise ValueError('Nonmonotonic PTX synchronization interval')
            site = manifest['sites'][sid]
            previous = exit_
            yield dict(rank=meta['rank'], device=meta['device'], launch_id=meta['launch_id'],
                       kernel=meta['kernel'], block_id=block, thread_id=thread, barrier_site=sid,
                       enter_ns=enter, exit_ns=exit_, elapsed_ns=exit_ - enter,
                       sync_at_end=False, status='ok', pid=meta['pid'], device_uuid=meta['device_uuid'],
                       phase=site['category'], stream=meta.get('stream', 'unknown'),
                       context_generation=meta.get('context_generation', 0),
                       session_id=meta.get('session_id', 'legacy'), calibration_id=meta.get('calibration_id', 0),
                       event_index=ordinal, opcode=site['opcode'], site_type=site['type'],
                       ptx_line=site['ptx_line'], scope=site['scope'],
                       warp_id=thread // 32, lane_id=thread % 32,
                       thread_selection=manifest['thread_selection'])
