"""Strict native-journal decoding and same-host Chrome instruction timelines.

Instruction brackets are not complete wait intervals or kernel spans. Independent
kernel activity is a separate acceptance gate, never inferred from a bracket's
envelope or the host launch timestamp.
"""
import argparse
import ast
from bisect import bisect_right
import re
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path

from .clocks import fit_clock
from .identity import (ArtifactIdentity, ContextIdentity, KernelIdentity, LaunchIdentity,
                       ModuleIdentity, ProcessIdentity)
from .records import BufferLayout, RecordError, decode_native
from .probe import loop_selection, validate_loop_depth
from .ptx import PTXError, parse
from .sampling import CONTEXT_ABI, policy_hash, resolve_policy, validate_policy


class TraceError(ValueError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text())


def _unique(rows, key, label):
    result = {}
    for row in rows:
        value = key(row)
        if value in result:
            raise TraceError('Duplicate ' + label)
        result[value] = row
    return result


def _local(directory, name):
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise TraceError('Record path escapes process directory')
    return path


class NativeProcess:
    def __init__(self, journal):
        self.path = Path(journal)
        self.directory = self.path.parent
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        if any(row['kind'] == 'fatal' for row in rows):
            raise TraceError('Native fatal event: ' + str(journal))
        processes = [r for r in rows if r['kind'] == 'process']
        if len(processes) != 1:
            raise TraceError('Expected exactly one process identity')
        self.process = processes[0]
        p = self.process
        if p.get('capture_mode', 'full') != 'full':
            raise TraceError('Observation-only capture has no GPU probe records; use the ablation validator')
        self.capture_policy = None
        if 'context_abi' in p:
            if type(p['context_abi']) is not int or p['context_abi'] != CONTEXT_ABI:
                raise TraceError('Unsupported capture context ABI')
            self.capture_policy = validate_policy(p['capture_policy'])
            if self.capture_policy != p['capture_policy'] or policy_hash(self.capture_policy) != p['capture_policy_sha256']:
                raise TraceError('Capture policy identity mismatch')
        elif any(key in p for key in ('capture_policy', 'capture_policy_sha256')):
            raise TraceError('Sampling metadata missing context ABI')
        if not p.get('hostname') or not p.get('boot_id') or p.get('host_clock') != 'CLOCK_MONOTONIC_RAW':
            raise TraceError('Missing same-host clock-domain identity')
        self.domain = (p['hostname'], p['boot_id'], p['host_clock'])
        self.identity = ProcessIdentity(p['session'], p['pid'], str(p['started']), int(p['rank']))
        self.contexts = _unique([r for r in rows if r['kind'] == 'context'], lambda r: r['context'], 'context')
        self.owners = _unique([r for r in rows if r['kind'] == 'load'], lambda r: r['owner'], 'image owner')
        self.drains = _unique([r for r in rows if r['kind'] == 'drain'], lambda r: r['ordinal'], 'drain')
        calibration = _unique([r for r in rows if r['kind'] == 'calibration'],
                              lambda r: (r['step'], r['context'], r['phase']), 'calibration')
        self.clocks = {}
        for step, context, phase in calibration:
            if phase != 'before':
                continue
            before = calibration[step, context, 'before']
            after = calibration.get((step, context, 'after'))
            if after is None or before['uuid'] != after['uuid'] or before['uuid'] != self.contexts[context]['uuid']:
                raise TraceError('Missing/mismatched clock calibration')
            self.clocks[step, context] = fit_clock(before['samples'], after['samples'])
        if len(calibration) != 2 * len(self.clocks):
            raise TraceError('Orphan clock calibration')
        self.launches, self.captures, seen = [], {}, set()
        active, recorded, observed, previous = None, 0, 0, 0
        for row in rows:
            kind = row['kind']
            if row['host_ns'] < previous:
                raise TraceError('Nonmonotonic process journal')
            previous = row['host_ns']
            if kind == 'capture_begin':
                if active is not None or row['step'] in self.captures:
                    raise TraceError('Duplicate/nested capture')
                active, recorded, observed = row['step'], 0, 0
                self.captures[active] = {'begin': row}
            elif kind == 'launch':
                if row['ordinal'] in seen or row.get('status') != 0:
                    raise TraceError('Duplicate or failed launch')
                seen.add(row['ordinal'])
                if active is None:
                    if row['step'] != -1 or 'raw' in row:
                        raise TraceError('Launch is outside its capture')
                    continue
                if row['step'] != active:
                    raise TraceError('Launch/capture step mismatch')
                if row['coverage'] not in {'instrumented', 'observed', 'pass_through'}:
                    raise TraceError('Unknown coverage disposition')
                if (row['coverage'] == 'instrumented') != ('raw' in row):
                    raise TraceError('Instrumented launch has no records')
                recorded += 'raw' in row
                observed += row['coverage'] == 'observed'
                self.launches.append(row)
            elif kind == 'capture_end':
                if (active != row['step'] or row.get('successful') is not True or
                        recorded != row['recorded_launches'] or observed != row.get('observed_launches', 0)):
                    raise TraceError('Failed/incomplete capture')
                self.captures[active]['end'] = row
                active = None
        if active is not None or not self.captures or not any(r['kind'] == 'finalized' for r in rows):
            raise TraceError('Capture was not fully drained/finalized')
        if set(self.drains) != {r['ordinal'] for r in self.launches if 'raw' in r}:
            raise TraceError('Missing/orphan native drain')
        self.images = {}
        self.activities = _unique([r for r in rows if r['kind'] == 'kernel_activity'],
                                  lambda r: r['ordinal'], 'kernel activity')
        checks = _unique([r for r in rows if r['kind'] == 'activity_validation'],
                         lambda r: r['step'], 'activity validation')
        self.activity_verified = bool(checks)
        if self.activities or checks:
            selected = {r['ordinal']: r for r in self.launches if r['coverage'] != 'pass_through'}
            if set(self.activities) != set(selected) or set(checks) != set(self.captures):
                raise TraceError('Missing/orphan independent kernel activity')
            for step, check in checks.items():
                if check['dropped'] != 0 or check['matched'] != sum(r['step'] == step for r in selected.values()):
                    raise TraceError('Dropped or mismatched CUPTI activity')
            for ordinal, activity in self.activities.items():
                launch = selected[ordinal]
                capture = self.captures[launch['step']]
                if (activity['step'] != launch['step'] or activity['grid'] != launch['grid'] or
                        activity['block'] != launch['block'] or activity['source'] != 'CUPTI_CONCURRENT_KERNEL' or
                        activity['host_clock'] != 'CLOCK_MONOTONIC_RAW' or
                        not capture['begin']['host_ns'] <= activity['begin_host_ns'] <
                        activity['end_host_ns'] <= capture['end']['host_ns']):
                    raise TraceError('Invalid independent kernel span/identity')
        for row in self.launches:
            if row['coverage'] == 'observed':
                self.decode_launch(row)

    def decode_launch(self, row):
        if row['context'] not in self.contexts or row['uuid'] != self.contexts[row['context']]['uuid']:
            raise TraceError('Unknown launch context/UUID')
        owner = self.owners.get(row['owner'])
        observed = row['coverage'] == 'observed'
        if (owner is None or owner['image'] != row['image'] or
                owner['coverage'] != ('original' if observed else 'replaced')):
            raise TraceError('Image ownership mismatch')
        image = self.images.get(row['image'])
        if image is None:
            image = read_json(_local(self.directory, row['image'] + '.image.json'))
            if image['id'] != row['image'] or image['id'] not in image['original_hashes']:
                raise TraceError('Original image identity mismatch')
            if (image['interval_mode'] not in {'instruction-only', 'loop-boundary'} or
                    image['probe']['sampling'] != ('launch_policy' if self.capture_policy is not None else 'all_threads') or
                    image['probe'].get('interval_mode', image['interval_mode']) != image['interval_mode']):
                raise TraceError('Unsupported probe semantics')
            try:
                depth = validate_loop_depth(image['interval_mode'], image.get('loop_depth', 1))
                probe_depth = validate_loop_depth(image['interval_mode'], image['probe'].get('loop_depth', 1))
            except PTXError as exc:
                raise TraceError(str(exc)) from exc
            if probe_depth != depth:
                raise TraceError('Image/probe loop depth mismatch')
            if image['interval_mode'] == 'loop-boundary':
                if (image['probe'].get('loop_selection') != loop_selection(depth) or
                        image['probe'].get('depth_fallback', 'outermost_available') != 'outermost_available'):
                    raise TraceError('Unsupported loop boundary policy')
                loops = _unique(image['probe']['loops'], lambda loop: loop['id'], 'loop identity')
                for loop in loops.values():
                    if (loop.get('loop_depth', 1) != depth or loop['selection'] != loop_selection(depth) or
                            loop.get('depth_fallback', 'outermost_available') != 'outermost_available'):
                        raise TraceError('Loop inventory depth mismatch')
                for site in image['probe']['sites']:
                    if site.get('interval_kind') == 'loop':
                        loop = loops.get(site['loop_id'])
                        if (loop is None or site['exit_edge'] not in loop['exits'] or
                                site['function'] != loop['function'] or site['unit'] != loop['unit']):
                            raise TraceError('Invalid loop exit site')
                    elif site.get('interval_kind') != 'instruction':
                        raise TraceError('Missing interval kind')
            self.images[row['image']] = image
        if type(image.get('probe_enabled', True)) is not bool:
            raise TraceError('Image probe_enabled must be Boolean')
        if observed != (image.get('probe_enabled', True) is False):
            raise TraceError('Launch does not match image probe exclusion')
        if row['kernel'] not in image['entries']:
            raise TraceError('Entry is not owned by the original image')
        context = ContextIdentity(self.identity, row['context'], row['uuid'])
        module = ModuleIdentity(context, row['owner'], ArtifactIdentity(image['id'], image['target'], 'cubin'))
        identity = LaunchIdentity(KernelIdentity(module, row['kernel']), row['ordinal'], row['stream'])
        sampling = 'all'
        if self.capture_policy is not None:
            sampling, rule = resolve_policy(self.capture_policy, row['kernel'], image['label'])
            if (any(type(value) is not int or value != CONTEXT_ABI for value in
                    (row.get('context_abi'), image.get('context_abi'), image['probe'].get('context_abi'))) or
                    (not observed and row.get('raw_format') != 'states-packed-v2') or
                    row.get('capture_sampling') != sampling or
                    type(row.get('capture_policy_rule')) is not int or row['capture_policy_rule'] != rule or
                    row.get('capture_policy_sha256') != self.process['capture_policy_sha256']):
                raise TraceError('Launch sampling policy/ABI mismatch')
        elif row.get('raw_format', '').endswith('-v2') or 'capture_sampling' in row or 'context_abi' in image:
            raise TraceError('Sampled launch lacks process policy')
        if observed:
            if (any(key in row for key in ('raw', 'raw_format', 'capacity', 'threads', 'recorded_threads',
                                           'logical_threads', 'record_storage')) or row['ordinal'] in self.drains):
                raise TraceError('Excluded image has probe allocation or records')
            if row['ordinal'] not in self.activities:
                raise TraceError('Excluded image requires independent kernel activity')
            return identity, image, {}, [], None
        layout = BufferLayout.from_dimensions(row['grid'], row['block'], row['capacity'], sampling)
        if self.capture_policy is not None and (row.get('logical_threads') != layout.logical_threads or
                                               row.get('recorded_threads') != layout.threads):
            raise TraceError('Sampled thread counts disagree with geometry')
        if row['threads'] != layout.threads:
            raise TraceError('Thread count does not match launch geometry')
        sites = _unique(image['probe']['sites'], lambda r: r['ordinal'], 'site ordinal')
        data = _local(self.directory, row['raw']).read_bytes()
        events = decode_native(data, launch=identity, layout=layout, sites=sites,
                               raw_format=row.get('raw_format', 'states-fixed-v1'),
                               loop_sites=[ordinal for ordinal, site in sites.items()
                                           if site.get('interval_kind') == 'loop'])
        drain = self.drains[row['ordinal']]
        if (drain['overflow_threads'] or drain['incomplete_threads'] or
                drain['records'] != len(events) or drain['raw_bytes'] != len(data)):
            raise TraceError('Native drain counts differ from decoded records')
        clock = self.clocks.get((row['step'], row['context']))
        if clock is None:
            raise TraceError('Launch has no contemporaneous clock model')
        capture = self.captures[row['step']]
        for event in events:
            begin, end = clock.map_ns(event.begin), clock.map_ns(event.end)
            error = max(clock.uncertainty_ns(event.begin), clock.uncertainty_ns(event.end))
            if begin + error < capture['begin']['host_ns'] or end - error > capture['end']['host_ns']:
                raise TraceError('Mapped instruction lies outside its capture range')
            activity = self.activities.get(row['ordinal'])
            if activity and (begin + error < activity['begin_host_ns'] or end - error > activity['end_host_ns']):
                raise TraceError('Instruction lies outside independent kernel activity beyond clock uncertainty')
        return identity, image, sites, events, clock


def load_processes(root, ranks=None, steps=None):
    root = Path(root)
    processes = []
    for journal in sorted(root.glob('process-*/journal.jsonl')):
        # CPU helper processes can inherit the preload but have no captures.
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        if any(r['kind'] in {'capture_begin', 'fatal'} for r in rows):
            processes.append(NativeProcess(journal))
    if not processes:
        raise TraceError('No completed native capture processes')
    if len({p.domain for p in processes}) != 1 or len({p.identity.session for p in processes}) != 1:
        raise TraceError('Incompatible host clock domains or sessions')
    actual_ranks = [p.identity.rank for p in processes]
    if len(set(actual_ranks)) != len(actual_ranks) or (ranks is not None and set(actual_ranks) != set(ranks)):
        raise TraceError('Duplicate/missing ranks')
    selected = set(processes[0].captures) if steps is None else set(steps)
    if any(set(p.captures) != selected for p in processes):
        raise TraceError('Ranks did not capture the same complete steps')
    return processes


def coordinates(thread, grid, block):
    width = block[0] * block[1] * block[2]
    block_id, lane = divmod(thread, width)
    return ([block_id % grid[0], (block_id // grid[0]) % grid[1], block_id // (grid[0] * grid[1])],
            [lane % block[0], (lane // block[0]) % block[1], lane // (block[0] * block[1])])


def kernel_track_label(library, kernel, image):
    """Readable entry family plus an image/specialization tag; full symbols stay in args."""
    families = {
        'DeepEP': ('dispatch_copy_epilogue', 'combine_reduce_epilogue', 'dispatch_impl', 'combine_impl'),
        'NCCL': ('ReduceScatter', 'AllGather', 'AllReduce', 'Broadcast', 'Reduce', 'SendRecv'),
    }
    name = next((family for family in families.get(library, ()) if family in kernel), kernel)
    tag = hashlib.sha256(json.dumps([image, kernel]).encode()).hexdigest()[:12]
    return f'{name} [{tag}]'


class InstructionSources:
    """Resolve captured site offsets only against hash-identical original PTX."""

    def __init__(self, paths):
        self.paths, self.modules, self.sites, self.presentations = {}, {}, {}, {}
        self.inventory, self.counts = {}, Counter()
        for root in map(Path, paths):
            if not root.exists():
                raise TraceError('Missing source PTX path: ' + str(root))
            candidates = sorted(root.rglob('*.ptx')) if root.is_dir() else [root]
            for path in candidates:
                digest = hashlib.sha256(path.read_text().encode()).hexdigest()
                self.paths.setdefault(digest, path.resolve())
        if not self.paths:
            raise TraceError('No source PTX files supplied')

    def module(self, digest):
        if digest not in self.modules:
            path = self.paths.get(digest)
            if path is None:
                raise TraceError('Missing hash-matched original PTX: ' + digest)
            source = path.read_text()
            if hashlib.sha256(source.encode()).hexdigest() != digest:
                raise TraceError('Source PTX changed after indexing: ' + str(path))
            functions = {function.name: function for function in parse(source).functions if function.defined}
            # Mask comments while preserving offsets, newlines, and quoted filenames.
            clean = re.sub(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/',
                           lambda match: match[0] if match[0].startswith('"') else
                           re.sub(r'[^\n]', ' ', match[0]), source)
            files = {}
            for match in re.finditer(r'(?m)^[ \t]*\.file[ \t]+(\d+)[ \t]+("(?:\\.|[^"\\])*")', clean):
                number, filename = int(match[1]), ast.literal_eval(match[2])
                if files.setdefault(number, filename) != filename:
                    raise TraceError('Conflicting PTX debug file number')
            locations = list(re.finditer(r'(?m)^[ \t]*\.loc[ \t]+(\d+)[ \t]+(\d+)[ \t]+(\d+)([^\n]*)', clean))
            self.modules[digest] = (path, source, functions, files, locations, [m.start() for m in locations])
        return self.modules[digest]

    def resolve(self, image, site):
        digest = image['probe'].get('original_sha256', {}).get(site['unit'])
        if not digest:
            raise TraceError('Missing captured PTX identity for source annotation')
        key = digest, site['id']
        descriptor = tuple(site[field] for field in ('function', 'instruction', 'start', 'end', 'line', 'opcode', 'predicate'))
        if key in self.sites:
            previous, ref = self.sites[key]
            if previous != descriptor:
                raise TraceError('Conflicting source site metadata')
            return ref
        path, source, functions, files, locations, offsets = self.module(digest)
        function = functions.get(site['function'])
        if function is None or not 0 <= site['instruction'] < len(function.instructions):
            raise TraceError('Source site function/instruction mismatch')
        instruction = function.instructions[site['instruction']]
        if tuple(getattr(instruction, field) for field in ('start', 'end', 'line', 'opcode', 'predicate')) != descriptor[2:]:
            raise TraceError('Source site offsets/opcode mismatch')
        text = ' '.join(source[instruction.start:instruction.end].split())
        # Recover complete mnemonics even in historical manifests whose lexer split :: suffixes.
        opcode = re.match(r'(?:@!?[\w%$]+\s+)?([a-z][\w.:]*)', text)
        if opcode is None:
            raise TraceError('Cannot recover original PTX instruction')
        record = {'site_id': site['id'], 'ptx_file': str(path), 'ptx_sha256': digest,
                  'ptx_line': instruction.line, 'ptx_instruction': text, 'opcode': opcode[1],
                  'function': site['function'], 'instruction_index': instruction.index,
                  'source_location': None, 'inlined_at': []}

        def location(file_id, line, column):
            if file_id == 0 or line == 0:
                return None
            if file_id not in files:
                raise TraceError('Unknown PTX debug file number')
            return dict(file=files[file_id], line=line, column=column)

        index = bisect_right(offsets, instruction.start) - 1
        if index >= 0 and locations[index].start() >= function.body_start:
            match = locations[index]
            record['source_location'] = location(*(int(match[i]) for i in (1, 2, 3)))
            record['debug_directive'] = match[0].strip()
            for inline in re.finditer(r'\binlined_at\s+(\d+)\s+(\d+)\s+(\d+)', match[4]):
                caller = location(*(int(inline[i]) for i in (1, 2, 3)))
                if caller is not None:
                    record['inlined_at'].append(caller)
        ref = str(len(self.inventory) + 1)
        self.inventory[ref] = record
        self.sites[key] = descriptor, ref
        return ref

    def annotate(self, image, site, members):
        key = image['id'], site['id']
        if key not in self.presentations:
            refs = [self.resolve(image, member) for member in members]
            records = [self.inventory[ref] for ref in refs]
            short = lambda loc: f'{Path(loc["file"]).name}:{loc["line"]}:{loc["column"]}'
            locations, ptx_locations, labels = [], [], []
            for row in records:
                ptx = f'{Path(row["ptx_file"]).name}:{row["ptx_line"]}'
                location = short(row['source_location']) if row['source_location'] else 'CUDA source unavailable'
                if row['inlined_at']:
                    location += ' <- ' + ' <- '.join(map(short, row['inlined_at']))
                locations.append(location)
                ptx_locations.append(ptx)
                labels.append(row['opcode'] + ' @ ' + (location if row['source_location'] else 'PTX ' + ptx))
            label = ' + '.join(dict.fromkeys(labels))
            if site.get('interval_kind') == 'loop':
                label = f'Loop: {label} / exit {site["exit_index"]}'
            annotation = {'instructions': [row['ptx_instruction'] for row in records],
                          'locations': locations, 'ptx_locations': ptx_locations, 'refs': refs}
            mapped = sum(row['source_location'] is not None for row in records)
            coverage = ('cuda_mapped' if mapped == len(records) else 'cuda_partial' if mapped else 'ptx_only')
            self.presentations[key] = label, annotation, coverage
        label, annotation, coverage = self.presentations[key]
        self.counts[image['label'] + '_' + coverage + '_intervals'] += 1
        return label, annotation

    def report(self):
        return {'format': 'instruction-source-v1', 'unique_sites': len(self.inventory),
                'unique_ptx_modules': len(self.modules), 'counts': dict(self.counts),
                'ptx_sha256_verified': True, 'full_locations': 'source-locations.json',
                'location_basis': 'Original PTX .file/.loc and immediate inlined_at; no inferred CUDA locations',
                'loop_semantics': 'Listed sites are static members; duration is the structural loop, not each instruction',
                'kernel_spans': 'Whole-kernel spans unchanged; not instruction durations'}


def normalize_chrome_document(document):
    """Use portable integer thread IDs and retain readable track metadata."""
    threads, events = {}, []
    for original in document['traceEvents']:
        event = dict(original)
        tid = event.get('tid', 0)
        if isinstance(tid, str):
            key = (event['pid'], tid)
            if key not in threads:
                threads[key] = len(threads) + 1
                events.append({'ph': 'M', 'name': 'thread_name', 'pid': event['pid'],
                               'tid': threads[key], 'args': {'name': tid}})
            event['tid'] = threads[key]
        if event.get('cat', '').endswith('.kernel'):
            library = event['args']['library']
            name = event['name']
            labels = (('ReduceScatter', 'ReduceScatter'), ('AllGather', 'AllGather'), ('AllReduce', 'AllReduce'))
            if library == 'DeepEP':
                labels = (('dispatch_copy_epilogue', 'dispatch copy epilogue'),
                          ('combine_reduce_epilogue', 'combine reduce epilogue'),
                          ('dispatch_impl', 'dispatch'), ('combine_impl', 'combine'))
            for needle, label in labels:
                if needle in name:
                    event['name'] = library + ' ' + label
                    break
        events.append(event)
    return {**document, 'traceEvents': events}


def warp_coordinates(thread, block):
    """Block-local warps restart at every block, including partial warps."""
    width = block[0] * block[1] * block[2]
    block_id, local_thread = divmod(thread, width)
    warp, lane = divmod(local_thread, 32)
    return block_id, warp, lane


def select_warp_events(events, block):
    """Keep one observed thread per launch/block/warp/site, including its repeats.

    Selection is offline and fixed over the launch for each site. It is not
    dynamic active-lane election, a warp-wide envelope, or a completion bound.
    """
    representatives = {}
    for event in events:
        block_id, warp, lane = warp_coordinates(event.thread, block)
        key = (block_id, warp, event.site)
        representatives[key] = min(representatives.get(key, event.thread), event.thread)
    selected = [event for event in events
                if event.thread == representatives[(*warp_coordinates(event.thread, block)[:2], event.site)]]
    return selected, {
        'observed_warp_sites': len(representatives),
        'warp_sites_without_lane_zero': sum(warp_coordinates(thread, block)[2] != 0
                                            for thread in representatives.values()),
    }


def export_trace(root, destination, *, ranks=None, steps=None, require_kernel_activity=False,
                 instruction_granularity='thread', kernel_overview=None, lane_grouping='sampled-thread',
                 source_ptx=None):
    if instruction_granularity not in {'thread', 'warp', 'captured'}:
        raise TraceError('Unsupported instruction granularity')
    if lane_grouping not in {'sampled-thread', 'kernel-ctx-stream', 'kernel-rank-ctx-stream'}:
        raise TraceError('Unsupported lane grouping')
    grouped = lane_grouping != 'sampled-thread'
    cross_rank = lane_grouping == 'kernel-rank-ctx-stream'
    lane_layout = lane_grouping + '-v1' if grouped else 'kernel-qualified-v1'
    sources = InstructionSources(source_ptx) if source_ptx is not None else None
    processes = load_processes(root, ranks, steps)
    policies = {p.process.get('capture_policy_sha256') for p in processes}
    if len(policies) != 1:
        raise TraceError('Mixed capture policies across ranks')
    policy_capture = processes[0].capture_policy is not None
    verified = all(p.activity_verified for p in processes)
    if require_kernel_activity and not verified:
        raise TraceError('Independent GPU kernel activity is required')
    overview = []
    origin = min(c['begin']['host_ns'] for p in processes for c in p.captures.values())
    kernel_labels = None
    if kernel_overview is not None:
        document = read_json(kernel_overview)
        metadata = document['metadata']
        if (metadata['session'] != processes[0].identity.session or metadata['origin_host_ns'] != origin):
            raise TraceError('Kernel overview session/origin mismatch')
        names = {(event['pid'], event['tid']): event['args']['name']
                 for event in document['traceEvents'] if event.get('name') == 'thread_name'}
        source_pids = {(event['pid'], event['tid']): event['args']['source_pid']
                       for event in document['traceEvents']
                       if event.get('name') == 'thread_name' and 'source_pid' in event['args']}
        kernel_labels = _unique([dict(event, pid=source_pids.get((event['pid'], event['tid']), event['pid']),
                                     tid=names.get((event['pid'], event['tid']), event['tid']))
                                 for event in document['traceEvents']
                                 if event.get('cat', '').endswith('.kernel')],
                                lambda event: event['args']['launch_id'], 'overview kernel')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    sampling = {'capture_sampling': 'all_threads', 'instruction_granularity': instruction_granularity,
                'selection_stage': 'offline_export',
                'representative_policy': ('lowest_observed_lane_per_launch_block_warp_site'
                                          if instruction_granularity == 'warp' else 'all_threads'),
                'repeated_observations': 'preserved_for_selected_thread',
                'whole_warp_completion': False,
                'instruction_observations_complete': instruction_granularity == 'thread'}
    if policy_capture:
        sampling.update(capture_sampling='kernel_policy', capture_policy=processes[0].capture_policy,
                        capture_policy_sha256=processes[0].process['capture_policy_sha256'],
                        instruction_observations_complete=False, all_threads_complete=False,
                        selected_threads_complete=True, whole_block_completion=False)
    if instruction_granularity == 'captured':
        sampling.update(selection_stage='gpu_capture' if policy_capture else 'none',
                        representative_policy='per_kernel_capture_policy' if policy_capture else 'all_threads',
                        captured_intervals_complete=True, second_offline_sampling=False)
    capture_modes = set()
    coverage, clocks, counts, uncertainty = [], [], Counter(), 0
    modes, depths, loop_inventory = set(), {}, {}
    site_selections = set()
    image_dispositions = {}
    timeline = destination / 'timeline.json'
    fields = ['session', 'pid', 'rank', 'gpu_uuid', 'context', 'stream', 'step', 'launch_id',
              'launch_ordinal', 'image', 'kernel', 'library', 'thread', 'site', 'site_id',
              'callee', 'opcode', 'ordinal', 'occurrence', 'begin_tick', 'end_tick',
              'begin_host_ns', 'end_host_ns', 'uncertainty_ns']
    with timeline.open('x') as output, (destination / 'events.csv').open('x', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        output.write('{"traceEvents":[')
        first = True
        tracks, kernel_groups, rank_order = {}, {}, {}
        group_pid_base = max(process.identity.pid for process in processes) + 1

        def write_event(event):
            nonlocal first
            if not first:
                output.write(',')
            output.write(json.dumps(event, separators=(',', ':')))
            first = False

        def emit(event, group_track=None, rank_lane=None):
            if kernel_labels is not None and event.get('cat', '').endswith('.kernel'):
                label = kernel_labels.pop(event['args']['launch_id'], None)
                if (label is None or any(label[key] != event[key] for key in ('pid', 'ts', 'dur')) or
                        any(label['args'][key] != event['args'][key]
                            for key in ('rank', 'step', 'library', 'kernel', 'cupti_stream'))):
                    raise TraceError('Kernel overview identity/timestamp mismatch')
                event = label
            if cross_rank and rank_lane is not None:
                args = event['args']
                kernel = args.get('original_kernel', args['kernel'])
                # Display groups use the original image hash and entry, never rank-local module IDs.
                group_key = args['library'], args['image'], kernel
                if group_key not in kernel_groups:
                    group_pid = group_pid_base + len(kernel_groups)
                    kernel_groups[group_key] = group_pid
                    emit({'ph': 'M', 'name': 'process_name', 'pid': group_pid, 'tid': 0,
                          'args': {'name': args['library'] + '/' + kernel_track_label(
                              args['library'], kernel, args['image']), 'display_group': 'kernel',
                                   'library': args['library'], 'image': args['image'], 'kernel': kernel}})
                event = dict(event, pid=kernel_groups[group_key],
                             tid=f'rank {rank_lane["rank"]}/ctx{rank_lane["context"]}/stream{rank_lane["stream"]}')
            elif group_track is not None:
                event = dict(event, tid=group_track)
            elif event.get('cat', '').endswith('.kernel'):
                args = event['args']
                label = kernel_track_label(args['library'], args.get('original_kernel', args['kernel']), args['image'])
                prefix = label + ' | '
                if not str(event['tid']).startswith(prefix):
                    event = dict(event, tid=prefix + str(event['tid']))
            if not cross_rank and not event.get('cat', '').endswith(('.sys_instruction', '.sys_loop')):
                overview.append(event)
            if (grouped or instruction_granularity in {'warp', 'captured'}) and isinstance(event.get('tid'), str):
                args = event.get('args', {})
                key = (event['pid'], event['tid'], args.get('image'),
                       args.get('original_kernel', args.get('kernel')))
                if key not in tracks:
                    tracks[key] = len(tracks) + 1
                    metadata = {'ph': 'M', 'name': 'thread_name', 'pid': event['pid'],
                                'tid': tracks[key], 'args': {'name': event['tid']}}
                    if cross_rank:
                        if rank_lane is not None:
                            # Keep native ownership in lane metadata without changing event arguments.
                            metadata['args'].update(rank_lane)
                            rank_order[event['pid'], tracks[key]] = tuple(rank_lane[name] for name in
                                                                         ('rank', 'context', 'stream', 'source_pid'))
                        overview.append(metadata)
                    write_event(metadata)
                event = dict(event, tid=tracks[key])
            if cross_rank and not event.get('cat', '').endswith(('.sys_instruction', '.sys_loop')):
                overview.append(event)
            write_event(event)

        for process in processes:
            p = process.identity
            emit({'ph': 'M', 'name': 'process_name', 'pid': p.pid, 'tid': 0,
                  'args': {'name': f'rank {p.rank}'}})
            for step, capture in process.captures.items():
                emit({'ph': 'X', 'name': f'Capture step {step} (includes drain)', 'cat': 'host.capture',
                      'pid': p.pid, 'tid': 'host', 'ts': (capture['begin']['host_ns'] - origin) / 1000,
                      'dur': (capture['end']['host_ns'] - capture['begin']['host_ns']) / 1000})
            for (step, context), clock in process.clocks.items():
                clocks.append({'process': p.id, 'rank': p.rank, 'step': step, 'context': context, **clock.manifest()})
            for row in process.launches:
                item = {'pid': p.pid, 'rank': p.rank, **row}
                item['raw'] = str(Path(process.directory.name) / row['raw']) if 'raw' in row else None
                if row['coverage'] == 'pass_through':
                    coverage.append(item)
                    continue
                identity, image, sites, events, clock = process.decode_launch(row)
                site_selection = image['probe'].get('site_selection', 'explicit_sys_v1')
                site_selections.add(site_selection)
                if len(site_selections) != 1:
                    raise TraceError('Mixed instruction coverage policies in capture')
                sampling['site_selection'] = site_selection
                modes.add(image['interval_mode'])
                if len(modes) != 1:
                    raise TraceError('Mixed interval modes in capture')
                image_depth = image.get('loop_depth', 1)
                if depths.setdefault(image['label'], image_depth) != image_depth:
                    raise TraceError('Mixed loop depths within library: ' + image['label'])
                probe_enabled = image.get('probe_enabled', True)
                if image_dispositions.setdefault(image['id'], probe_enabled) != probe_enabled:
                    raise TraceError('Conflicting image probe exclusion across ranks')
                for loop in (image['probe'].get('loops', []) if probe_enabled else []):
                    if loop_inventory.setdefault(loop['id'], loop) != loop:
                        raise TraceError('Conflicting loop inventory')
                if probe_enabled:
                    item['raw_sha256'] = hashlib.sha256(_local(process.directory, row['raw']).read_bytes()).hexdigest()
                item.update(launch_id=identity.id, kernel_id=identity.kernel.id, library=image['label'],
                            probe_enabled=probe_enabled, intervals=len(events),
                            disposition=('probe_excluded_original_image' if not probe_enabled else
                            'observed_sys_instructions' if events else
                            'no_dynamic_sys_observations' if sites else 'no_sys_sites'))
                counts[image['label'] + ('_instrumented_launches' if probe_enabled else '_excluded_launches')] += 1
                coverage.append(item)
                counts[image['label'] + '_launches'] += 1
                counts[image['label'] + '_input_intervals'] += len(events)
                capture_mode = row.get('capture_sampling', 'all')
                capture_modes.add(capture_mode)
                if instruction_granularity == 'warp' and capture_mode == 'block':
                    raise TraceError('Block-captured data requires captured or thread display granularity')
                granularity = (('thread' if capture_mode == 'all' else capture_mode)
                               if instruction_granularity == 'captured' else instruction_granularity)
                selected, selection = (select_warp_events(events, row['block'])
                                       if instruction_granularity == 'warp' else (events, {}))
                item['instruction_export'] = {'granularity': granularity,
                                              'capture_sampling': capture_mode,
                                              'input_records': len(events), 'exported_records': len(selected),
                                              **selection}
                counts[image['label'] + '_filtered_intervals'] += len(events) - len(selected)
                for key, value in selection.items():
                    counts[image['label'] + '_' + key] += value
                kernel_label = kernel_track_label(image['label'], row['kernel'], row['image'])
                base_track = f'{image["label"]}/{kernel_label}/ctx{row["context"]}/stream{row["stream"]}'
                rank_lane = (dict(source_pid=p.pid, rank=p.rank, context=row['context'], stream=row['stream'])
                             if cross_rank else None)
                activity = process.activities.get(row['ordinal'])
                if activity:
                    role = {'NCCL': ('Unattributed' if image['interval_mode'] == 'loop-boundary' else 'DP'),
                            'DeepEP': 'EP'}.get(image['label'], image['label'])
                    item['kernel_activity'] = activity
                    emit({'ph': 'X', 'cat': role + '.' + image['label'] + '.kernel',
                          'name': activity['kernel'], 'pid': p.pid,
                          'tid': f'{role} {image["label"]}/stream{activity["cupti_stream"]}',
                          'ts': (activity['begin_host_ns'] - origin) / 1000,
                          'dur': (activity['end_host_ns'] - activity['begin_host_ns']) / 1000,
                          'args': {'rank': p.rank, 'step': row['step'], 'gpu_uuid': row['uuid'],
                                   'launch_id': identity.id, 'launch_ordinal': row['ordinal'],
                                   'image': row['image'], 'original_kernel': row['kernel'],
                                   'library': image['label'], 'launch_api': row['api'],
                                   'probe_enabled': probe_enabled,
                                   'sys_instruction_records': len(events), **activity}},
                         group_track=base_track if grouped else None, rank_lane=rank_lane)
                    counts[image['label'] + '_kernel_spans'] += 1
                for event in selected:
                    site = sites[event.site]
                    begin, end = clock.map_ns(event.begin), clock.map_ns(event.end)
                    error = max(clock.uncertainty_ns(event.begin), clock.uncertainty_ns(event.end))
                    uncertainty = max(uncertainty, error)
                    record = dict(session=p.session, pid=p.pid, rank=p.rank, gpu_uuid=row['uuid'],
                                  context=row['context'], stream=row['stream'], step=row['step'],
                                  launch_id=identity.id, launch_ordinal=row['ordinal'], image=row['image'],
                                  kernel=row['kernel'], library=image['label'], thread=event.thread,
                                  site=event.site, site_id=site['id'], callee=site['function'], opcode=site['opcode'],
                                  ordinal=event.ordinal, occurrence=event.occurrence, begin_tick=event.begin,
                                  end_tick=event.end, begin_host_ns=begin, end_host_ns=end, uncertainty_ns=error)
                    writer.writerow(record)
                    block, lane = coordinates(event.thread, row['grid'], row['block'])
                    # Decimal strings preserve exact GPU ticks in JavaScript viewers.
                    args = {**record, 'begin_tick': str(event.begin), 'end_tick': str(event.end),
                            'block': block, 'thread_xyz': lane, 'observation': site['observation'],
                            'memory_order': site['memory_order'], 'original_ptx_line': site['line']}
                    if policy_capture:
                        args.update(capture_sampling=capture_mode, display_granularity=granularity,
                                    capture_policy_rule=row['capture_policy_rule'])
                    kind = site.get('interval_kind', 'instruction')
                    if kind == 'loop':
                        args.update(interval_kind=kind, loop_id=site['loop_id'], loop_depth=image.get('loop_depth', 1),
                                    exit_edge=site['exit_edge'], exit_index=site['exit_index'],
                                    exit_outcome=site['exit_outcome'], synchronization_success='unclassified',
                                    contained_sys_sites=[{'opcode': value['opcode'], 'line': value['line'],
                                                          'predicate': value['predicate']}
                                                         for value in loop_inventory[site['loop_id']]['contained_sites']])
                    if image['interval_mode'] == 'loop-boundary':
                        counts[image['label'] + '_' + kind + '_intervals'] += 1
                    track = base_track
                    if not grouped:
                        if granularity in {'warp', 'block'}:
                            block_id, warp, _ = warp_coordinates(event.thread, row['block'])
                            track += f'/block{block_id}'
                            if granularity == 'warp':
                                track += f'/warp{warp}'
                        else:
                            track += f'/thread{event.thread}'
                    source_name = None
                    if sources is not None:
                        members = loop_inventory[site['loop_id']]['contained_sites'] if kind == 'loop' else [site]
                        source_name, args['source_annotation'] = sources.annotate(image, site, members)
                    emit({'ph': 'X', 'cat': image['label'] + ('.sys_loop' if kind == 'loop' else '.sys_instruction'),
                          'name': source_name or (f'Sync loop L{site["line"]} / exit {site["exit_index"]}'
                                                  if kind == 'loop' else site['opcode']),
                          'pid': p.pid, 'tid': track,
                          'ts': (begin - origin) / 1000, 'dur': (end - begin) / 1000, 'args': args},
                         rank_lane=rank_lane)
                    counts[image['label'] + '_intervals'] += 1
        if kernel_labels:
            raise TraceError('Orphan kernel overview events')
        if cross_rank:
            for index, key in enumerate(sorted(kernel_groups, key=lambda key: (key[0], key[2], key[1]))):
                emit({'ph': 'M', 'name': 'process_sort_index', 'pid': kernel_groups[key], 'tid': 0,
                      'args': {'sort_index': index}})
            for process in processes:
                emit({'ph': 'M', 'name': 'process_sort_index', 'pid': process.identity.pid, 'tid': 0,
                      'args': {'sort_index': len(kernel_groups) + process.identity.rank}})
            indexes = Counter()
            for (pid, tid), order in sorted(rank_order.items(), key=lambda item: (item[0][0], item[1])):
                emit({'ph': 'M', 'name': 'thread_sort_index', 'pid': pid, 'tid': tid,
                      'args': {'sort_index': indexes[pid]}})
                indexes[pid] += 1
        interval_mode = next(iter(modes), 'instruction-only')
        unique_depths = set(depths.values())
        depth = next(iter(unique_depths), 1) if len(unique_depths) <= 1 else None
        excluded_images = sorted(key for key, enabled in image_dispositions.items() if not enabled)
        if excluded_images:
            sampling.update(excluded_image_ids=excluded_images,
                            exclusion_stage='original_image_execution',
                            excluded_kernel_spans_retained=True,
                            instruction_observations_complete=False)
        if policy_capture:
            sampling['effective_capture_modes'] = sorted(capture_modes)
            sampling['all_threads_complete'] = capture_modes == {'all'} and not excluded_images
        if interval_mode == 'loop-boundary':
            sampling['instruction_observations_complete'] = False
            sampling.update(loop_selection=loop_selection(depth) if depth is not None else 'per_library',
                            loop_depth=depth, loop_depths=dict(sorted(depths.items())),
                            depth_fallback='outermost_available')
            sampling['loop_intervals_complete'] = (capture_modes == {'all'} and not excluded_images and
                                                   instruction_granularity in {'thread', 'captured'})
            sampling['captured_intervals_complete'] = instruction_granularity in {'thread', 'captured'}
        output.write('],"displayTimeUnit":"ns","metadata":')
        json.dump({'session': processes[0].identity.session, 'origin_host_ns': origin,
                   'host_clock_domain': processes[0].domain, 'interval_mode': interval_mode,
                   'lane_layout': lane_layout,
                   **({'source_annotation': sources.report(), 'source_inventory': sources.inventory} if sources else {}),
                   'loop_inventory': loop_inventory, 'loop_depth': depth, 'loop_depths': dict(sorted(depths.items())),
                   'kernel_activity_verified': verified, 'clock_models': clocks,
                   'instruction_export': sampling,
                   'kernel_overview': str(kernel_overview) if kernel_overview is not None else None}, output)
        output.write('}\n')
    with timeline.open('rb') as source, gzip.open(destination / 'timeline.json.gz', 'wb') as compressed:
        while chunk := source.read(1024 * 1024):
            compressed.write(chunk)
    (destination / 'communication.json').write_text(json.dumps(normalize_chrome_document({
        'traceEvents': overview, 'displayTimeUnit': 'ns', 'metadata': {
            'session': processes[0].identity.session, 'origin_host_ns': origin,
            'host_clock_domain': processes[0].domain, 'kernel_activity_verified': verified,
            'lane_layout': lane_layout,
            'kernel_timestamps': 'CUPTI GPU activity normalized to CLOCK_MONOTONIC_RAW',
            'instruction_detail': 'timeline.json', 'clock_models': clocks}}), separators=(',', ':')) + '\n')
    (destination / 'coverage.json').write_text(json.dumps(coverage, indent=2) + '\n')
    (destination / 'clocks.json').write_text(json.dumps(clocks, indent=2) + '\n')
    result = {'schema': 1, 'status': 'instruction_records_validated', 'ranks': sorted(p.identity.rank for p in processes),
              'lane_layout': lane_layout,
              'steps': sorted(processes[0].captures), 'counts': dict(counts),
              'maximum_estimated_uncertainty_ns': uncertainty, 'kernel_activity_verified': verified,
              'full_trace_acceptance': verified, 'interval_mode': interval_mode, 'loop_depth': depth,
              'loop_depths': dict(sorted(depths.items())),
              'kernel_activity_source': 'CUPTI_CONCURRENT_KERNEL' if verified else None,
              'instruction_export': sampling,
              'kernel_overview': str(kernel_overview) if kernel_overview is not None else None}
    if sources is not None:
        result['source_annotation'] = sources.report()
        (destination / 'source-locations.json').write_text(json.dumps(
            {'annotation': sources.report(), 'sources': sources.inventory}, indent=2) + '\n')
    (destination / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--steps', type=int, nargs='+', required=True)
    parser.add_argument('--require-kernel-activity', action='store_true')
    parser.add_argument('--instruction-granularity', choices=('thread', 'warp', 'captured'), default='thread',
                        help='captured preserves every GPU record on its policy track; warp selects one observed lane per warp/site offline')
    parser.add_argument('--lane-grouping', choices=('sampled-thread', 'kernel-ctx-stream', 'kernel-rank-ctx-stream'),
                        default='sampled-thread',
                        help='Group GPU events by kernel/context/stream, optionally placing kernels above ranks; intervals stay intact')
    parser.add_argument('--source-ptx', type=Path, nargs='+',
                        help='Original PTX files/directories for hash-verified instruction and source labels')
    parser.add_argument('--kernel-overview', type=Path,
                        help='Reuse role labels from a same-capture overview; kernel identity/timing must match')
    args = parser.parse_args()
    print(json.dumps(export_trace(args.raw, args.output, ranks=range(args.world_size), steps=args.steps,
                                 require_kernel_activity=args.require_kernel_activity,
                                 instruction_granularity=args.instruction_granularity,
                                 kernel_overview=args.kernel_overview, lane_grouping=args.lane_grouping,
                                 source_ptx=args.source_ptx), indent=2))


if __name__ == '__main__':
    main()
