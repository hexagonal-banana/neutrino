"""Fail-closed resolver for optimized, inline-annotated DeepEP direct-kernel PTX.

The measured region is the control-flow envelope of an outer gpu_barrier
inline instance. Pure instructions moved into that envelope by NVVM stay in
place. No source rewriting, polling-loop sampling, or added synchronization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

RECORD_BYTES = 24


class BoundaryError(ValueError):
    pass


@dataclass
class Instruction:
    line: int
    end: int
    text: str
    site: tuple | None
    location: tuple

    @property
    def branch(self):
        return re.fullmatch(r"(?:(@!?%[\w$]+)\s+)?bra(?:\.uni)?\s+([\w$]+)\s*;", self.text)


def debug_strings(text):
    result = {}
    for m in re.finditer(r"(?m)^([\w$]+):\s*\n((?:\s*\.b8 [\d,]+\s*\n)+)", text):
        numbers = re.findall(r"\d+", m[2].replace('.b8', ''))
        result[m[1]] = bytes(map(int, numbers)).rstrip(b'\0').decode('utf-8')
    return result


def template_values(name):
    m = re.search(r"11gpu_barrierI((?:L[bil]n?\d+E){11})E", name)
    if not m:
        raise BoundaryError(f"Unsupported barrier template: {name}")
    return [int(value.replace('n', '-')) for value in re.findall(r"L[bil](n?\d+)E", m[1])]


def parse(text):
    lines = text.splitlines()
    entries = list(re.finditer(r"\.visible\s+\.entry\s+(\S+)\s*\((.*?)\)\s*[^\{]*\{", text, re.S))
    if len(entries) != 1:
        raise BoundaryError('Exactly one visible entry is required')
    entry = entries[0]
    opening = text.count('\n', 0, entry.end() - 1)
    depth, closing = 0, None
    for n in range(opening, len(lines)):
        code = lines[n].split('//')[0]
        depth += code.count('{') - code.count('}')
        if depth == 0:
            closing = n
            break
    if closing is None:
        raise BoundaryError('Unbalanced kernel body')
    strings = debug_strings(text)
    files = {int(n): p for n, p in re.findall(r'\.file\s+(\d+)\s+"([^"]+)"', text)}
    locations, sites, instructions, labels = {}, {}, [], {}
    loc, pending, start = (0, 0, 0), '', 0

    def ancestry(key, visited=None):
        visited = set() if visited is None else visited
        if key in visited or key not in locations:
            return None
        visited.add(key)
        name, parent = locations[key]
        if '11gpu_barrierI' in name:
            return (name, parent)
        return ancestry(parent, visited) if parent else None

    for n in range(opening + 1, closing):
        code = lines[n].split('//')[0].strip()
        if not code:
            continue
        m = re.match(r"\.loc\s+(\d+)\s+(\d+)\s+(\d+)(.*)", code)
        if m:
            loc = tuple(map(int, m.group(1, 2, 3)))
            fn = re.search(r'function_name\s+([\w$]+)', m[4])
            parent = re.search(r'inlined_at\s+(\d+)\s+(\d+)\s+(\d+)', m[4])
            if fn:
                if fn[1] not in strings or not parent:
                    raise BoundaryError(f'Incomplete inline metadata at PTX line {n + 1}')
                name = strings[fn[1]]
                caller = tuple(map(int, parent.groups()))
                locations[loc] = (name, caller)
                if '11gpu_barrierI' in name:
                    sites[(name, caller)] = template_values(name)
            # Bare .loc directives do not erase a previous inline definition.
            else:
                locations.setdefault(loc, ('', None))
            continue
        if code.endswith(':'):
            labels[code[:-1]] = len(instructions)
            continue
        if code.startswith('.') or code in ('{', '}'):
            continue
        if not pending:
            start = n
        pending += ' ' + code
        if ';' in code:
            instructions.append(Instruction(start, n, pending.strip(), ancestry(loc), loc))
            pending = ''
    if pending:
        raise BoundaryError('Unparsed instruction')
    return lines, entry, opening, closing, files, sites, instructions, labels


def resolve(text):
    lines, entry, opening, closing, files, candidates, ins, labels = parse(text)
    name = entry[1]
    family = next((k for k in ('dispatch', 'combine', 'barrier')
                   if re.search(rf'\d+{k}_implI', name)), None)
    transport = re.search(rf'\d+{family}_implILb([01])E', name) if family else None
    if transport is None:
        raise BoundaryError(f'Unsupported non-direct kernel: {name}')
    is_nvlink = int(transport[1])
    expected = 1 if family == 'barrier' else 2
    if len(candidates) != expected:
        raise BoundaryError(f'Expected {expected} outer barrier sites, found {len(candidates)}')
    if len({key[1] for key in candidates}) != expected:
        raise BoundaryError('Ambiguous source call-site coordinates')
    target = re.search(r'(?m)^\.target\s+(sm_\w+)', text)
    if not target:
        raise BoundaryError('Missing original architecture')
    edges = []
    for i, instruction in enumerate(ins):
        branch = instruction.branch
        if branch:
            if branch[2] not in labels:
                raise BoundaryError(f'Unknown branch target: {branch[2]}')
            edges.append((i, labels[branch[2]], 'taken', branch[1]))
            if branch[1]:
                edges.append((i, i + 1, 'fallthrough', None))
        elif re.match(r'(?:@!?%\w+\s+)?(?:ret|exit|trap)\b', instruction.text):
            if instruction.text.startswith('@'):
                raise BoundaryError('Predicated return/trap requires an explicit edge mapping')
            edges.append((i, None, 'trap' if instruction.text.startswith('trap') else 'return', None))
        else:
            edges.append((i, i + 1, 'fallthrough', None))
    def pure(inst):
        op = re.sub(r'^@!?%\w+\s+', '', inst.text).split()[0]
        return op.startswith(('mov.', 'cvt.', 'cvta.', 'add.', 'sub.', 'mul.', 'mad.',
                              'shl.', 'shr.', 'and.', 'or.', 'xor.', 'neg.', 'not.',
                              'setp.', 'selp.', 'min.', 'max.', 'ld.param.', 'bra'))

    result = []
    for key, values in candidates.items():
        marked = [i for i, inst in enumerate(ins) if inst.site == key]
        if not marked:
            raise BoundaryError('Inline instance has no mapped instructions')
        effects = [i for i in marked if not pure(ins[i])]
        if not effects:
            raise BoundaryError('No barrier effects with recoverable inline ancestry')
        # NVVM can attach a previous instance's location to a hoisted compare,
        # or attach the next instance to a transfer-only branch. Such detached
        # pure instructions do not define an interval through unrelated work.
        first, hi = min(effects), max(effects)
        previous_effect = max((i for i in range(first) if not pure(ins[i]) and ins[i].site != key), default=-1)
        lo = min(i for i in marked if previous_effect < i <= first)
        # A loop latch can follow the last effect (including a timeout trap).
        # Close only mapped backedges with a purely computational tail. The
        # ordinary single-entry, foreign-effect, and passage proofs still apply.
        while True:
            latches = [s for s, d, _, _ in edges
                       if s > hi and d is not None and lo < d <= hi
                       and ins[s].site == key]
            if not latches:
                break
            tail = max(latches)
            if any(not pure(ins[i]) or ins[i].site not in (None, key)
                   for i in range(hi + 1, tail + 1)):
                raise BoundaryError('Barrier loop tail crosses unrelated work')
            hi = tail
        detached = [ins[i].line + 1 for i in marked if not lo <= i <= hi]
        # A region is accepted only if its envelope is single-entry, includes
        # no foreign side effects, and all departing CFG edges are explicit.
        foreign = []
        for i in range(lo, hi + 1):
            inst = ins[i]
            if inst.site != key:
                if not pure(inst):
                    raise BoundaryError(f'Foreign effect inside barrier envelope at {inst.line + 1}: {inst.text}')
                foreign.append(inst.line + 1)
        incoming = [(s, d) for s, d, _, _ in edges if d is not None and lo <= d <= hi and not lo <= s <= hi]
        if {d for _, d in incoming} != {lo}:
            raise BoundaryError(f'Missing or ambiguous entry at {ins[lo].line + 1}, '
                                            f'envelope ends at {ins[hi].end + 1}: '
                                            f'{[(ins[s].line + 1, ins[d].line + 1, ins[s].text) for s, d in incoming]}')
        if any(lo <= s <= hi and d == lo for s, d, _, _ in edges):
            raise BoundaryError('Entry is a polling backedge target')
        exits = []
        for s, d, kind, predicate in edges:
            if lo <= s <= hi and (d is None or not lo <= d <= hi):
                if kind == 'trap':
                    continue
                exits.append(dict(line=ins[s].line + 1, end_line=ins[s].end + 1,
                                  target_line=ins[d].line + 1 if d is not None and d < len(ins) else None,
                                  kind=kind, predicate=predicate))
        if not exits:
            raise BoundaryError('No normal exit edges')
        nvlink, scaleout, ranks, blocks, threads, qps, timeout, tag, flush, sync_start, sync_end = values
        if nvlink != is_nvlink or scaleout != 1 or ranks < 2 or (flush and not sync_start):
            raise BoundaryError('Unsupported barrier synchronization parameters')
        source = files.get(key[1][0], '')
        if not source.endswith(f'/deep_ep/impls/{"comm" if family == "barrier" else "ep"}/{family}.cuh'):
            raise BoundaryError(f'Unexpected source call site: {source}')
        source_line = key[1][1]
        phase = 'standalone' if family == 'barrier' else ('entry' if tag in (2, 4) else 'completion')
        result.append(dict(source=source, source_line=source_line, tag=tag, phase=phase,
                           flush_stores=bool(flush), sync_at_start=bool(sync_start), sync_at_end=bool(sync_end),
                           blocks=blocks, threads=threads, ranks=ranks, qps=qps,
                           transport='nvlink' if nvlink else 'direct_rdma', entry_line=ins[lo].line + 1,
                           last_line=ins[hi].end + 1, exits=exits, interleaved_pure_lines=foreign,
                           detached_pure_lines=detached,
                           function=key[0]))
    result.sort(key=lambda s: s['entry_line'])
    tags = [s['tag'] for s in result]
    if tags != {'dispatch': [2, 3], 'combine': [4, 5], 'barrier': [1]}[family]:
        raise BoundaryError(f'Unexpected tag order: {tags}')
    for i, site in enumerate(result):
        site['id'] = i
    # Prove exactly one ordered passage through each region on every normal
    # CFG path. Loops inside a region retain their state; timeout traps stop.
    line_to_index = {inst.line + 1: i for i, inst in enumerate(ins)}
    starts = {line_to_index[s['entry_line']]: 2 * s['id'] for s in result}
    departures = {}
    for site in result:
        for edge in site['exits']:
            departures[(line_to_index[edge['line']], edge['kind'])] = 2 * site['id'] + 1
    outgoing = {}
    for edge in edges:
        outgoing.setdefault(edge[0], []).append(edge)
    todo, seen = [(0, 0)], set()
    while todo:
        node, state = todo.pop()
        if (node, state) in seen:
            continue
        seen.add((node, state))
        if node in starts:
            if state != starts[node]:
                raise BoundaryError('Repeated, skipped, or unordered barrier entry')
            state += 1
        for _, dest, kind, _ in outgoing.get(node, []):
            next_state = state
            if (node, kind) in departures:
                if state != departures[node, kind]:
                    raise BoundaryError('Barrier exit without exactly one entry')
                next_state += 1
            if kind == 'trap':
                continue
            if kind == 'return':
                if next_state != 2 * len(result):
                    raise BoundaryError('A normal return bypasses a barrier boundary')
            elif dest is not None and dest < len(ins):
                todo.append((dest, next_state))
            else:
                raise BoundaryError('Unterminated control flow')
    return dict(schema=1, kernel=name, family=family, target=target[1],
                params=len(re.findall(r'\.param\b', entry[2])), sites=result,
                record_bytes=RECORD_BYTES, slots=2 * len(result), map_level='thread',
                measurement='representative thread barrier residence time', clocks='unsynchronized per GPU')


def instrument(text, manifest):
    lines, entry, opening, closing, *_ = parse(text)
    if '__nb_' in text:
        raise BoundaryError('Already instrumented')
    capacity = manifest['slots']
    inserts = {}

    def put(line, code):
        inserts.setdefault(line, []).append(code)

    def record(site, kind, predicate=None):
        slot = site * 2 + kind
        guard = f'and.pred %__nb_take, %__nb_rep, {predicate[1:]};' if predicate and not predicate.startswith('@!') else ''
        if predicate and predicate.startswith('@!'):
            guard = f'not.pred %__nb_take, {predicate[2:]};\nand.pred %__nb_take, %__nb_rep, %__nb_take;'
        pred = '%__nb_take' if predicate else '%__nb_rep'
        offset = slot * RECORD_BYTES
        return f'''// Neutrino barrier {site} {'exit' if kind else 'entry'}
{guard}
@{pred} mov.u64 %__nb_time, %globaltimer;
@{pred} ld.global.u32 %__nb_count, [%__nb_ptr+{offset + 16}];
@{pred} add.u32 %__nb_count, %__nb_count, 1;
@{pred} st.global.u64 [%__nb_ptr+{offset}], %__nb_time;
@{pred} st.global.u32 [%__nb_ptr+{offset + 8}], {site};
@{pred} st.global.u32 [%__nb_ptr+{offset + 12}], {kind};
@{pred} st.global.u32 [%__nb_ptr+{offset + 16}], %__nb_count;'''

    for site in manifest['sites']:
        put(site['entry_line'] - 1, record(site['id'], 0))
        for edge in site['exits']:
            before = edge['kind'] in ('taken', 'return')
            put(edge['line'] - 1 if before else edge['end_line'], record(site['id'], 1, edge['predicate']))
    prolog = f'''
.reg .pred %__nb_rep, %__nb_take;
.reg .b32 %__nb_x, %__nb_y, %__nb_z, %__nb_b, %__nb_n, %__nb_count;
.reg .b64 %__nb_ptr, %__nb_off, %__nb_time;
mov.u32 %__nb_x, %tid.x;
mov.u32 %__nb_y, %tid.y;
mov.u32 %__nb_z, %tid.z;
or.b32 %__nb_x, %__nb_x, %__nb_y;
or.b32 %__nb_x, %__nb_x, %__nb_z;
setp.eq.u32 %__nb_rep, %__nb_x, 0;
mov.u32 %__nb_b, %ctaid.z;
mov.u32 %__nb_n, %nctaid.y;
mov.u32 %__nb_y, %ctaid.y;
mad.lo.u32 %__nb_b, %__nb_b, %__nb_n, %__nb_y;
mov.u32 %__nb_n, %nctaid.x;
mov.u32 %__nb_x, %ctaid.x;
mad.lo.u32 %__nb_b, %__nb_b, %__nb_n, %__nb_x;
mov.u32 %__nb_x, %ntid.x;
mov.u32 %__nb_y, %ntid.y;
mov.u32 %__nb_z, %ntid.z;
mul.lo.u32 %__nb_n, %__nb_x, %__nb_y;
mul.lo.u32 %__nb_n, %__nb_n, %__nb_z;
mul.lo.u32 %__nb_b, %__nb_b, %__nb_n;
mul.wide.u32 %__nb_off, %__nb_b, {capacity * RECORD_BYTES};
ld.param.u64 %__nb_ptr, [__nb_buffer];
add.u64 %__nb_ptr, %__nb_ptr, %__nb_off;
'''
    # Declarations and prologue precede the original body; all original module
    # text, including aggregate parameters and debug sections, is retained.
    put(opening + 1, prolog)
    output = []
    param_close = text.count('\n', 0, entry.start(2) + len(entry[2]))
    for n, line in enumerate(lines):
        output.extend(inserts.get(n, []))
        if n == param_close:
            output.append(', .param .u64 __nb_buffer')
        output.append(line)
    return '\n'.join(output) + '\n'


def prepare(cubin, workdir, assemble=True):
    cubin, workdir = Path(cubin).resolve(), Path(workdir)
    ptx = cubin.with_suffix('.ptx')
    if not ptx.is_file():
        raise BoundaryError(f'Missing exact PTX sidecar: {ptx}')
    text = ptx.read_text()
    try:
        manifest = resolve(text)
    except BoundaryError as error:
        raise BoundaryError(f'{ptx}: {error}') from error
    manifest.update(artifact=str(cubin), artifact_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(),
                    ptx_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    ptxas_options=['-arch=' + manifest['target'], '-O3', '-lineinfo',
                                   '--register-usage-level=10'])
    workdir.mkdir(parents=True, exist_ok=True)
    probed = instrument(text, manifest)
    manifest['injected_ptx_locations'] = []
    current = None
    for line, code in enumerate(probed.splitlines(), 1):
        marker = re.fullmatch(r'// Neutrino barrier (\d+) (entry|exit)', code)
        if marker:
            current = dict(site=int(marker[1]), kind=marker[2], marker_line=line)
        elif current is not None and '%globaltimer' in code:
            manifest['injected_ptx_locations'].append(dict(current, timestamp_line=line))
            current = None
    (workdir / 'probed.ptx').write_text(probed)
    if assemble:
        # Match the pinned DeepJIT register optimizer (10, not PTXAS default 5).
        # A no-probe control must reproduce original SASS, including scheduling
        # bits, before attributing any performance change to instrumentation.
        flags = ['ptxas', *manifest['ptxas_options']]
        plain = workdir / 'roundtrip.cubin'
        subprocess.run([*flags, str(ptx), '-o', str(plain)], check=True, capture_output=True)
        original_sass, plain_sass = [subprocess.check_output(
            ['cuobjdump', '--dump-sass', str(binary)], text=True) for binary in (cubin, plain)]
        (workdir / 'roundtrip.sass').write_text(plain_sass)
        if original_sass != plain_sass:
            raise BoundaryError(f'Uninstrumented PTX reassembly changed SASS: {cubin}')
        manifest['uninstrumented_reassembly_identical'] = True
        command = [*flags, '-v', str(workdir / 'probed.ptx'), '-o', str(workdir / 'probed.cubin')]
        result = subprocess.run(command, capture_output=True, text=True)
        (workdir / 'ptxas.log').write_text(result.stdout + result.stderr)
        result.check_returncode()
    (workdir / 'sites.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (workdir / 'barrier.info').write_text(f"{manifest['params']} {manifest['slots']} {manifest['record_bytes']}\n{manifest['kernel']}\n")
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cubin')
    parser.add_argument('workdir')
    parser.add_argument('--no-assemble', action='store_true')
    args = parser.parse_args()
    prepare(args.cubin, args.workdir, not args.no_assemble)
