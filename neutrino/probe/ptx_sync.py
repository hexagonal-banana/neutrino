"""Opcode/CFG-only PTX synchronization probes; no application or debug metadata.

One entry per file-backed CUDA library, with its exact .ptx sidecar. Polling
regions are conservative natural loops, not reconstructed application barriers.
Each selected thread records visits into a bounded, nonwrapping buffer.
The default selects lane 0 of each warp, with compact per-warp storage.
Local thread-barrier probes are excluded by default; original PTX stays intact.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

RECORD_BYTES = 24
REGISTER = r'%?[A-Za-z_$][\w$]*'


class SyncError(ValueError):
    pass


@dataclass
class Instruction:
    start: int
    end: int
    line: int
    predicate: str | None
    opcode: str
    operands: str
    scope: tuple


def parse(text):
    # Preserve offsets, including newlines, while removing comments and strings.
    clean = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
                   lambda m: re.sub(r'[^\n]', ' ', m[0]), text)
    entries = list(re.finditer(r'\.entry\s+([\w$]+)\s*\((.*?)\)[^{;]*\{', clean, re.S))
    if len(entries) != 1:
        raise SyncError('Exactly one PTX entry definition per artifact is supported')
    entry = entries[0]
    pos, depth = entry.end(), 1
    instructions, labels = [], {}
    scopes, scope_id = [0], 0
    while pos < len(clean):
        if clean[pos].isspace():
            pos += 1
            continue
        if clean[pos] in '{}':
            if clean[pos] == '{':
                depth += 1
                scope_id += 1
                scopes.append(scope_id)
            else:
                depth -= 1
                scopes.pop()
            if not depth:
                break
            pos += 1
            continue
        label = re.match(r'([\w$]+):', clean[pos:])
        if label:
            key = (tuple(scopes), label[1])
            if key in labels:
                raise SyncError('Duplicate PTX label')
            labels[key] = len(instructions)
            pos += label.end()
            continue
        if clean[pos] == '.':
            directive = re.match(r'\.\w+', clean[pos:])[0]
            if directive in ('.loc', '.pragma'):
                stop = clean.find('\n', pos)
            else:
                stop = clean.find(';', pos)
            if stop < 0:
                raise SyncError('Unterminated PTX directive')
            pos = stop + 1
            continue
        stop = clean.find(';', pos)
        if stop < 0:
            raise SyncError('Unterminated PTX instruction')
        match = re.fullmatch(r'(?:(@!?%?[\w$]+)\s+)?([\w.:]+)(?:\s+([\s\S]*?))?\s*;', clean[pos:stop + 1])
        if not match:
            raise SyncError(f'Unparsed PTX at line {text.count(chr(10), 0, pos) + 1}: {clean[pos:stop + 1]!r}')
        instructions.append(Instruction(pos, stop + 1, text.count('\n', 0, pos) + 1,
                                        match[1], match[2], (match[3] or '').strip(), tuple(scopes)))
        pos = stop + 1
    if depth or not instructions:
        raise SyncError('Missing or unbalanced PTX entry body')
    return entry, instructions, labels


def classify(opcode):
    parts = opcode.split('.')
    if parts[0] in ('ld', 'atom') and 'acquire' in parts:
        return 'acquire'
    if parts[0] in ('st', 'red', 'atom') and any(p in parts for p in ('release', 'acq_rel')):
        return 'publication'
    if opcode.startswith(('mbarrier.try_wait.', 'mbarrier.test_wait.')):
        return 'mbarrier_wait'
    if opcode.startswith('cp.async.bulk.wait_group'):
        return 'source_read_wait' if 'read' in parts else 'async_completion_wait'
    if opcode == 'cp.async.bulk.commit_group':
        return 'async_commit_nonwaiting'
    if opcode.startswith('fence.proxy.async'):
        return 'proxy_fence_nonwaiting'
    if opcode.startswith(('bar.sync', 'barrier.sync', 'bar.warp.sync')):
        return 'local_barrier'
    if parts[0] == 'membar' or (parts[0] == 'fence' and
            any(p in parts for p in ('acquire', 'release', 'acq_rel', 'sc'))):
        return 'memory_fence_nonwaiting'
    return None


def memory_scope(opcode):
    return next((p for p in re.split(r'\.|::', opcode)
                 if p in ('sys', 'gpu', 'cta', 'cluster')), 'implicit')


def cfg(instructions, labels):
    edges = []
    for i, inst in enumerate(instructions):
        op = inst.opcode
        if op in ('bra', 'bra.uni'):
            target = next((labels[inst.scope[:depth], inst.operands]
                           for depth in range(len(inst.scope), 0, -1)
                           if (inst.scope[:depth], inst.operands) in labels), None)
            if target is None:
                raise SyncError('Unknown direct branch target')
            edges.append((i, target, 'taken', inst.predicate))
            if inst.predicate:
                edges.append((i, i + 1, 'fallthrough', None))
        elif op in ('ret', 'exit', 'trap'):
            edges.append((i, None, op, inst.predicate))
            if inst.predicate:
                edges.append((i, i + 1, 'fallthrough', None))
        elif op.startswith(('brx', 'jmp')) or op == 'call' or op.startswith('call.'):
            if op.startswith('call'):
                # Device callees are outside entry-body coverage; calls return.
                edges.append((i, i + 1, 'fallthrough', None))
            else:
                raise SyncError('Indirect control flow is unsupported')
        else:
            edges.append((i, i + 1, 'fallthrough', None))
    if any(d is not None and d >= len(instructions) for _, d, _, _ in edges):
        raise SyncError('Unterminated control flow or label outside entry body')
    return edges


def natural_loops(instructions, edges):
    n = len(instructions)
    succ, pred = [set() for _ in range(n)], [set() for _ in range(n)]
    for src, dst, _, _ in edges:
        if dst is not None:
            succ[src].add(dst)
            pred[dst].add(src)
    reachable, todo = set(), [0]
    while todo:
        node = todo.pop()
        if node not in reachable:
            reachable.add(node)
            todo.extend(succ[node])
    all_bits = sum(1 << i for i in reachable)
    dom = [all_bits if i in reachable else 0 for i in range(n)]
    dom[0] = 1
    changed = True
    while changed:
        changed = False
        for node in sorted(reachable - {0}):
            value = all_bits
            for parent in pred[node] & reachable:
                value &= dom[parent]
            value |= 1 << node
            if dom[node] != value:
                dom[node], changed = value, True
    loops = {}
    for tail in reachable:
        for head in succ[tail]:
            if not dom[tail] & (1 << head):
                continue
            region = {head, tail}
            todo = [tail] if tail != head else []
            while todo:
                node = todo.pop()
                for parent in pred[node] & reachable:
                    if parent not in region:
                        region.add(parent)
                        todo.append(parent)
            loops.setdefault(head, set()).update(region)
    return loops, dom


def polling_loop(index, region, head, ins, edges, dom):
    inst = ins[index]
    if inst.predicate or classify(inst.opcode) not in ('acquire', 'mbarrier_wait'):
        return False
    if any(classify(ins[i].opcode) and i != index for i in region):
        return False
    incoming = {d for s, d, _, _ in edges if s not in region and d in region}
    if incoming - {head}:
        return False
    # The signal address must be invariant, and the wait result must feed an
    # exit decision. This avoids treating ordinary payload-processing loops as waits.
    address = re.search(r'\[([^]]+)\]', inst.operands)
    result = re.match(REGISTER, inst.operands)
    if not address or not result:
        return False
    address_regs = set(re.findall(REGISTER, address[1]))
    dependent = {result[0]}
    for i in region:
        item = ins[i]
        parts = item.opcode.split('.')
        if ((parts[0] == 'st' and not {'local', 'param'} & set(parts)) or
                parts[0] in ('atom', 'red') or item.opcode.startswith('cp.async')):
            return False
        dest = re.match(REGISTER, item.operands)
        if dest and dest[0] in address_regs:
            return False
    changed = True
    while changed:
        changed = False
        for i in region:
            regs = re.findall(REGISTER, ins[i].operands)
            if len(regs) > 1 and set(regs[1:]) & dependent and regs[0] not in dependent:
                dependent.add(regs[0])
                changed = True
    exits = [(s, p) for s, d, _, p in edges if s in region and d not in region]
    exit_depends = any(ins[s].predicate and ins[s].predicate.lstrip('@!') in dependent for s, _ in exits)
    latches = [s for s, d, _, _ in edges if s in region and d == head]
    return exit_depends and all(dom[s] & (1 << index) for s in latches)


def episode_controls(region, head, ins, edges):
    """Control dependencies for one acyclic loop iteration; reject nested cycles."""
    sink = -1
    successors = {node: set() for node in region}
    indegree = dict.fromkeys(region, 0)
    for src, dst, _, _ in edges:
        if src in region:
            successors[src].add(dst if dst in region and dst != head else sink)
    for destinations in successors.values():
        for dst in destinations - {sink}:
            indegree[dst] += 1
    pending = [node for node in region if not indegree[node]]
    order = []
    while pending:
        node = pending.pop()
        order.append(node)
        for dst in successors[node] - {sink}:
            indegree[dst] -= 1
            if not indegree[dst]:
                pending.append(dst)
    if len(order) != len(region):
        return None
    post = {sink: {sink}}
    for node in reversed(order):
        post[node] = {node} | set.intersection(*(post[dst] for dst in successors[node]))
    controls = {node: set() for node in region}
    for src in region:
        if len(successors[src]) > 1:
            controlled = set.union(*(post[dst] for dst in successors[src])) - post[src] - {sink}
            for node in controlled:
                controls[node].add(src)
    while True:
        expanded = {node: parents | set().union(*(controls[p] for p in parents))
                    for node, parents in controls.items()}
        if expanded == controls:
            break
        controls = expanded
    return {node: {ins[parent].predicate.lstrip('@!') for parent in parents if ins[parent].predicate}
            for node, parents in controls.items()}


def polling_episode(index, region, head, ins, edges):
    """Recognize a single-signal, warp-cooperative wait without payload effects."""
    inst = ins[index]
    if not inst.opcode.startswith('ld.acquire.') or memory_scope(inst.opcode) != 'sys':
        return False
    if {d for s, d, _, _ in edges if s not in region and d in region} - {head}:
        return False
    address = re.search(r'\[([^]]+)\]', inst.operands)
    result = re.match(REGISTER, inst.operands)
    if not address or not result:
        return False
    address_regs = set(re.findall(REGISTER, address[1]))
    controls = episode_controls(region, head, ins, edges)
    if controls is None:
        return False
    dependencies = []
    shuffles = []
    has_warp_barrier = False
    arithmetic = {'mov', 'cvt', 'cvta', 'add', 'sub', 'mul', 'mad', 'setp', 'selp',
                  'and', 'or', 'xor', 'not', 'shl', 'shr', 'min', 'max', 'neg', 'abs'}
    for i in sorted(region):
        item = ins[i]
        op = item.opcode
        parts = op.split('.')
        if op == 'bar.warp.sync':
            has_warp_barrier = True
            continue
        if i != index and classify(op):
            return False
        if parts[0] in ('bra', 'ret', 'exit', 'trap'):
            continue
        if parts[0] == 'st' and set(parts) & {'local', 'param'}:
            continue
        if parts[0] == 'call' and re.match(r'(?:\([^)]*\)\s*,\s*)?vprintf\s*,', item.operands):
            continue
        private_load = parts[0] == 'ld' and set(parts) & {'local', 'param'}
        if i != index and not (parts[0] in arithmetic or private_load or op == 'shfl.sync.idx.b32'):
            return False
        # Vector and paired-predicate destinations must all be considered writes.
        operands = re.match(r'(\{[^}]*\}|[^,]+)(?:,\s*([\s\S]*))?$', item.operands)
        if not operands:
            return False
        destinations = set(re.findall(REGISTER, operands[1]))
        if destinations & address_regs:
            return False
        sources = set(re.findall(REGISTER, operands[2] or '')) | controls[i]
        if item.predicate:
            sources.add(item.predicate.lstrip('@!'))
        dependencies.append((destinations, sources))
        if op == 'shfl.sync.idx.b32':
            shuffles.append((destinations, sources))
    if not has_warp_barrier or not shuffles:
        return False
    dependent = {result[0]}
    while True:
        extended = dependent | set().union(*(dst for dst, src in dependencies if src & dependent), set())
        if extended == dependent:
            break
        dependent = extended
    # The polled value must pass through a shuffle and affect an exit condition.
    shuffled = set().union(*(dst for dst, src in shuffles if src & dependent), set())
    while True:
        extended = shuffled | set().union(*(dst for dst, src in dependencies if src & shuffled), set())
        if extended == shuffled:
            break
        shuffled = extended
    return any(ins[s].predicate and ins[s].predicate.lstrip('@!') in shuffled
               for s, d, _, _ in edges if s in region and d not in region)


def resolve(text, capacity=256, waits='loop', threads='warp', local_barriers='exclude', sync_scope='all'):
    if (type(capacity) is not int or not 1 <= capacity <= 65536 or
            waits not in ('loop', 'instruction', 'episode') or threads not in ('all', 'warp') or
            local_barriers not in ('include', 'exclude') or sync_scope not in ('all', 'sys')):
        raise SyncError('Invalid synchronization capacity, selection, or probe mode')
    if waits == 'episode' and (sync_scope != 'sys' or local_barriers != 'exclude'):
        raise SyncError('Polling episodes require sys scope and excluded local-barrier probes')
    if '__ns_' in text:
        raise SyncError('Reserved probe prefix or already instrumented PTX')
    entry, ins, labels = parse(text)
    edges = cfg(ins, labels)
    loops, dom = natural_loops(ins, edges) if waits != 'instruction' else ({}, [])
    sites = []
    for i, inst in enumerate(ins):
        family = classify(inst.opcode)
        if (family is None or (family == 'local_barrier' and local_barriers == 'exclude') or
                (sync_scope == 'sys' and memory_scope(inst.opcode) != 'sys')):
            continue
        matches = [(head, region) for head, region in loops.items() if i in region
                   and polling_loop(i, region, head, ins, edges, dom)]
        site = dict(id=len(sites), opcode=inst.opcode, instruction=inst.operands, ptx_line=inst.line,
                    category=family, type='instruction', instruction_index=i,
                    scope=memory_scope(inst.opcode))
        episode = False
        if not matches and waits == 'episode':
            matches = [(head, region) for head, region in loops.items() if i in region
                       and polling_episode(i, region, head, ins, edges)]
            episode = bool(matches)
        if matches:
            head, region = min(matches, key=lambda item: len(item[1]))
            site.update(type='polling_episode' if episode else 'polling_loop', header=head, nodes=sorted(region),
                        entry_edges=[list(e) for e in edges if e[0] not in region and e[1] in region],
                        exit_edges=[list(e) for e in edges if e[0] in region and e[1] not in region],
                        includes_calls=any(ins[node].opcode.startswith('call') for node in region))
            if episode:
                site.update(includes_warp_coordination=True,
                    participants='Only lanes executing the selected acquire at least once in this episode',
                    bounds='Loop entry to loop exit, including warp coordination and bookkeeping; zero-load episodes omitted')
        sites.append(site)
    for site in sites:
        if site['type'] == 'polling_episode' and any(
                other['id'] != site['id'] and other['instruction_index'] in site['nodes'] for other in sites):
            raise SyncError('Overlapping polling episode and selected probe')
    target = re.search(r'(?m)^\s*\.target\s+(sm_\w+)', text)
    if not target:
        raise SyncError('Missing architecture')
    return dict(schema=4 if threads == 'warp' else 3, mode='ptx_sync', family='ptx_sync', kernel=entry[1], target=target[1],
                params=len(re.findall(r'\.param\b', entry[2])), sites=sites,
                capacity=capacity, slots=capacity + 1, record_bytes=RECORD_BYTES,
                thread_selection=threads, record_group=32 if threads == 'warp' else 1,
                representative_lane=0 if threads == 'warp' else None,
                waits=waits, local_barriers=local_barriers, sync_scope=sync_scope,
                identification='PTX opcodes and CFG only',
                storage='per-warp header followed by bounded dynamic intervals' if threads == 'warp' else
                        'per-thread header followed by bounded dynamic intervals',
                measurement='PTX instruction brackets and recognized polling-loop/episode residence',
                limitations=['No transport inference from PTX scope; validate transport externally',
                             'Relaxed/volatile/MMIO signal protocols are not captured; not complete RDMA/NVLink synchronization',
                             'No API-level stream memory operations or implicit operand-dependency waits',
                             'No traversal into device callees; only the selected entry body',
                             'Unrecognized polling loops retain individual instruction attempts',
                             'Loop intervals may include loop bookkeeping and diagnostic calls',
                             'Release/commit/fence brackets are not remote completion latency',
                             'Standalone acquire brackets are not whole protocol waits or payload completion',
                             'PTX boundaries undergo assembly scheduling; not exact isolated SASS latency'] +
                           (['Cooperative polling episodes include internal warp synchronization/shuffles and bookkeeping',
                             'Episode entry precedes the first load; only lanes that actually load emit an interval; zero-load episodes omitted']
                            if waits == 'episode' else []) +
                           (['Only explicit system-scope acquire/release/fence sites; GPU/grid/CTA and async probes excluded']
                            if sync_scope == 'sys' else
                            ['Local thread-barrier probes excluded; acquire/release and async completion/order probes retained']
                            if local_barriers == 'exclude' else []) +
                           (['Lane 0 represents each warp; sites skipped by lane 0 are not observed',
                             'Warp intervals are lane-0 residence, not earliest-lane entry to latest-lane exit']
                            if threads == 'warp' else []))


def instrument(text, manifest):
    entry, ins, _ = parse(text)
    capacity = manifest['capacity']
    warp = manifest['thread_selection'] == 'warp'
    insertions = {}
    initial_probes = []
    episodes = any(s['type'] == 'polling_episode' for s in manifest['sites'])

    def put(offset, code):
        insertions.setdefault(offset, []).append('\n' + code + '\n')

    def guard(predicate):
        if predicate:
            code = f"{'not' if predicate.startswith('@!') else 'mov'}.pred %__ns_guard, {predicate.lstrip('@!')};"
            return code + ('\nand.pred %__ns_guard, %__ns_guard, %__ns_lane0;' if warp else '')
        return 'mov.pred %__ns_guard, %__ns_lane0;' if warp else 'mov.pred %__ns_guard, 1;'

    def begin(site, predicate=None):
        return (f'// PTX sync {site["id"]} entry\n' + guard(predicate) +
                '\n@%__ns_guard mov.u64 %__ns_begin, %globaltimer;')

    def end(site, predicate=None, saved_guard=False):
        sid = site['id']
        prefix = '' if saved_guard else guard(predicate) + '\n'
        return f'''// PTX sync {sid} exit
{prefix}@%__ns_guard mov.u64 %__ns_end, %globaltimer;
setp.lt.u32 %__ns_room, %__ns_count, {capacity};
and.pred %__ns_write, %__ns_room, %__ns_guard;
@%__ns_write mul.wide.u32 %__ns_offset, %__ns_count, 24;
@%__ns_write add.u64 %__ns_record, %__ns_base, %__ns_offset;
@%__ns_write st.global.u64 [%__ns_record+24], %__ns_begin;
@%__ns_write st.global.u64 [%__ns_record+32], %__ns_end;
@%__ns_write st.global.u32 [%__ns_record+40], {sid};
@%__ns_write st.global.u32 [%__ns_record+44], %__ns_count;
@%__ns_guard add.u32 %__ns_count, %__ns_count, 1;
@%__ns_guard min.u32 %__ns_count, %__ns_count, {capacity + 1};'''

    def edge_probe(edge, code):
        src, _, kind, _ = edge
        put(ins[src].end if kind == 'fallthrough' else ins[src].start, code)

    for site in manifest['sites']:
        if site['type'] == 'instruction':
            item = ins[site['instruction_index']]
            put(item.start, begin(site, item.predicate))
            put(item.end, end(site, saved_guard=True))
        else:
            episode = site['type'] == 'polling_episode'
            reset = '\n@%__ns_guard mov.pred %__ns_seen, 0;' if episode else ''
            if site['header'] == 0:
                initial_probes.append(begin(site) + reset)
            for edge in site['entry_edges']:
                edge_probe(edge, begin(site, edge[3]) + reset)
            if episode:
                item = ins[site['instruction_index']]
                put(item.start, guard(item.predicate) + '\n@%__ns_guard mov.pred %__ns_seen, 1;')
            for edge in site['exit_edges']:
                if edge[2] != 'trap':
                    code = (guard(edge[3]) + '\nand.pred %__ns_guard, %__ns_guard, %__ns_seen;\n' +
                            end(site, saved_guard=True)) if episode else end(site, edge[3])
                    edge_probe(edge, code)
    grouping = '''and.b32 %__ns_x, %__ns_t, 31;
setp.eq.u32 %__ns_lane0, %__ns_x, 0;
shr.u32 %__ns_t, %__ns_t, 5;
add.u32 %__ns_n, %__ns_n, 31;
shr.u32 %__ns_n, %__ns_n, 5;''' if warp else ''
    owner = '@%__ns_lane0 ' if warp else ''
    prologue = f'''
.reg .b32 %__ns_x, %__ns_y, %__ns_z, %__ns_b, %__ns_n, %__ns_t;
.reg .b32 %__ns_count, %__ns_calls, %__ns_done, %__ns_overflow;
.reg .b64 %__ns_base, %__ns_offset, %__ns_record, %__ns_begin, %__ns_end, %__ns_total, %__ns_wide;
.reg .pred %__ns_guard, %__ns_room, %__ns_write, %__ns_lane0;
{'.reg .pred %__ns_seen;' if episodes else ''}
mov.u32 %__ns_t, %tid.z;
mov.u32 %__ns_n, %ntid.y;
mov.u32 %__ns_y, %tid.y;
mad.lo.u32 %__ns_t, %__ns_t, %__ns_n, %__ns_y;
mov.u32 %__ns_n, %ntid.x;
mov.u32 %__ns_x, %tid.x;
mad.lo.u32 %__ns_t, %__ns_t, %__ns_n, %__ns_x;
mov.u32 %__ns_b, %ctaid.z;
mov.u32 %__ns_n, %nctaid.y;
mov.u32 %__ns_y, %ctaid.y;
mad.lo.u32 %__ns_b, %__ns_b, %__ns_n, %__ns_y;
mov.u32 %__ns_n, %nctaid.x;
mov.u32 %__ns_x, %ctaid.x;
mad.lo.u32 %__ns_b, %__ns_b, %__ns_n, %__ns_x;
mov.u32 %__ns_x, %ntid.x;
mov.u32 %__ns_y, %ntid.y;
mov.u32 %__ns_z, %ntid.z;
mul.lo.u32 %__ns_n, %__ns_x, %__ns_y;
mul.lo.u32 %__ns_n, %__ns_n, %__ns_z;
{grouping}
mad.lo.u32 %__ns_b, %__ns_b, %__ns_n, %__ns_t;
mul.wide.u32 %__ns_offset, %__ns_b, {(capacity + 1) * RECORD_BYTES};
ld.param.u64 %__ns_base, [__ns_buffer];
add.u64 %__ns_base, %__ns_base, %__ns_offset;
{owner}ld.global.u32 %__ns_calls, [%__ns_base+8];
{owner}add.u32 %__ns_calls, %__ns_calls, 1;
{owner}st.global.u32 [%__ns_base+8], %__ns_calls;
mov.u32 %__ns_count, 0;
'''
    insertions.setdefault(entry.end(), []).insert(0, '\n' + prologue + '\n'.join(initial_probes) + '\n')
    for inst in ins:
        if inst.opcode not in ('ret', 'exit'):
            continue
        put(inst.start, guard(inst.predicate) + f'''
@%__ns_guard ld.global.u32 %__ns_overflow, [%__ns_base+4];
setp.gt.u32 %__ns_room, %__ns_count, {capacity};
@%__ns_room mov.u32 %__ns_overflow, 1;
@%__ns_guard st.global.u32 [%__ns_base], %__ns_count;
@%__ns_guard st.global.u32 [%__ns_base+4], %__ns_overflow;
@%__ns_guard ld.global.u32 %__ns_done, [%__ns_base+12];
@%__ns_guard add.u32 %__ns_done, %__ns_done, 1;
@%__ns_guard st.global.u32 [%__ns_base+12], %__ns_done;
@%__ns_guard ld.global.u64 %__ns_total, [%__ns_base+16];
@%__ns_guard cvt.u64.u32 %__ns_wide, %__ns_count;
@%__ns_guard add.u64 %__ns_total, %__ns_total, %__ns_wide;
@%__ns_guard st.global.u64 [%__ns_base+16], %__ns_total;''')
    put(entry.end(2), (',' if manifest['params'] else '') + '\n.param .u64 __ns_buffer')
    output, pos = [], 0
    for offset, codes in sorted(insertions.items()):
        output.extend((text[pos:offset], ''.join(codes)))
        pos = offset
    output.append(text[pos:])
    return ''.join(output)


def prepare(cubin, workdir, assemble=True):
    cubin, workdir = Path(cubin).resolve(), Path(workdir)
    ptx = cubin.with_suffix('.ptx')
    if not ptx.is_file():
        raise SyncError(f'Missing exact PTX sidecar: {ptx}')
    text = ptx.read_text()
    manifest = resolve(text, int(os.environ.get('NEUTRINO_SYNC_CAPACITY', '256')),
                       os.environ.get('NEUTRINO_SYNC_WAITS', 'loop'),
                       os.environ.get('NEUTRINO_SYNC_THREADS', 'warp'),
                       os.environ.get('NEUTRINO_SYNC_LOCAL_BARRIERS', 'exclude'),
                       os.environ.get('NEUTRINO_SYNC_SCOPE', 'all'))
    level = int(os.environ.get('NEUTRINO_SYNC_REGISTER_LEVEL', '10'))
    if not 0 <= level <= 10:
        raise SyncError('Invalid PTXAS register usage level')
    flags = ['-arch=' + manifest['target'], '-O3', '-lineinfo', f'--register-usage-level={level}']
    manifest.update(artifact=str(cubin), artifact_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(),
                    ptx_sha256=hashlib.sha256(text.encode()).hexdigest(), ptxas_options=flags)
    workdir.mkdir(parents=True, exist_ok=True)
    if manifest['sites']:
        (workdir / 'probed.ptx').write_text(instrument(text, manifest))
        if assemble:
            plain = workdir / 'roundtrip.cubin'
            subprocess.run(['ptxas', *flags, str(ptx), '-o', str(plain)], check=True, capture_output=True)
            sass = [subprocess.check_output(['cuobjdump', '--dump-sass', str(p)], text=True)
                    for p in (cubin, plain)]
            (workdir / 'roundtrip.sass').write_text(sass[1])
            if sass[0] != sass[1]:
                raise SyncError('No-probe reassembly changed SASS; match assembler options/toolchain')
            manifest['uninstrumented_reassembly_identical'] = True
            result = subprocess.run(['ptxas', *flags, '-v', str(workdir / 'probed.ptx'),
                                     '-o', str(workdir / 'probed.cubin')], capture_output=True, text=True)
            (workdir / 'ptxas.log').write_text(result.stdout + result.stderr)
            result.check_returncode()
    (workdir / 'sites.json').write_text(json.dumps(manifest, indent=2) + '\n')
    slots = manifest['slots'] if manifest['sites'] else 0
    (workdir / 'barrier.info').write_text(
        f"{manifest['params']} {slots} 24\n{manifest['kernel']}\n{manifest['record_group']}\n")
    return manifest
