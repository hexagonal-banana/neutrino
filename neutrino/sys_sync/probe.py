"""Kernel-selective instruction brackets or structural loop-entry/exit intervals.

Loop boundaries follow exact CFG edges without inferring polling success.
Neither mode asserts remote arrival, consumption, or hardware completion.
"""
from dataclasses import asdict, replace
import hashlib
import json

from .context import ContextRewrite, PREFIX, REGISTER, insertions, propagate_context
from .control_flow import analyze
from .ptx import PTXError, SITE_SELECTION, SITE_SELECTION_NO_MBARRIER
from .records import COMMITTED, RECORD, SCHEMA, STARTED, STATE
from .sampling import CONTEXT_ABI


P = '%' + PREFIX + 'probe_'


def _prologue(storage='per-thread-v1'):
    text = f'''
.reg .pred {P}enabled, {P}active, {P}room, {P}bad, {P}warp, {P}block_sample, {P}selected;
.reg .b32 {P}scalar, {P}capacity, {P}used, {P}expected;
.reg .b64 {P}thread, {P}block, {P}width, {P}temp, {P}limit, {P}index, {P}mode;
.reg .b64 {P}state, {P}base, {P}slot, {P}offset, {P}begin, {P}end;
setp.ne.u64 {P}enabled, {REGISTER}, 0;
'''

    def special(register, special):
        return (f'@{P}enabled mov.u32 {P}scalar, %{special};\n'
                f'@{P}enabled cvt.u64.u32 {P}{register}, {P}scalar;\n')

    text += special('thread', 'tid.z')
    for axis in ('y', 'x'):
        text += special('temp', 'ntid.' + axis)
        text += f'@{P}enabled mul.lo.u64 {P}thread, {P}thread, {P}temp;\n'
        text += special('temp', 'tid.' + axis)
        text += f'@{P}enabled add.u64 {P}thread, {P}thread, {P}temp;\n'
    text += special('block', 'ctaid.z')
    for axis in ('y', 'x'):
        text += special('temp', 'nctaid.' + axis)
        text += f'@{P}enabled mul.lo.u64 {P}block, {P}block, {P}temp;\n'
        text += special('temp', 'ctaid.' + axis)
        text += f'@{P}enabled add.u64 {P}block, {P}block, {P}temp;\n'
    text += special('width', 'ntid.x')
    for axis in ('y', 'z'):
        text += special('temp', 'ntid.' + axis)
        text += f'@{P}enabled mul.lo.u64 {P}width, {P}width, {P}temp;\n'
    text += f'''
@{P}enabled ld.global.u32 {P}scalar, [{REGISTER}+36];
mov.pred {P}bad, 0;
@{P}enabled setp.ne.u32 {P}bad, {P}scalar, 3;
@{P}bad trap;
@{P}enabled ld.global.u64 {P}mode, [{REGISTER}+48];
@{P}enabled setp.gt.u64 {P}bad, {P}mode, 2;
@{P}bad trap;
mov.pred {P}warp, 0;
mov.pred {P}block_sample, 0;
mov.pred {P}selected, 1;
@{P}enabled setp.eq.u64 {P}warp, {P}mode, 1;
@{P}enabled setp.eq.u64 {P}block_sample, {P}mode, 2;
@{P}enabled mov.u64 {P}index, {P}thread;
@{P}enabled mad.lo.u64 {P}thread, {P}block, {P}width, {P}thread;
@{P}warp and.b64 {P}temp, {P}index, 31;
@{P}warp setp.eq.u64 {P}selected, {P}temp, 0;
@{P}warp shr.u64 {P}index, {P}index, 5;
@{P}warp add.u64 {P}width, {P}width, 31;
@{P}warp shr.u64 {P}width, {P}width, 5;
@{P}block_sample setp.eq.u64 {P}selected, {P}index, 0;
@{P}block_sample mov.u64 {P}index, 0;
@{P}block_sample mov.u64 {P}width, 1;
and.pred {P}enabled, {P}enabled, {P}selected;
@{P}enabled mad.lo.u64 {P}index, {P}block, {P}width, {P}index;
@{P}enabled ld.global.u64 {P}base, [{REGISTER}];
@{P}enabled ld.global.u64 {P}state, [{REGISTER}+8];
@{P}enabled ld.global.u64 {P}limit, [{REGISTER}+24];
@{P}enabled ld.global.u32 {P}capacity, [{REGISTER}+32];
@{P}enabled setp.ge.u64 {P}bad, {P}index, {P}limit;
@{P}bad trap;
@{P}enabled mul.lo.u64 {P}offset, {P}index, {STATE.size};
@{P}enabled add.u64 {P}state, {P}state, {P}offset;
@{P}enabled cvt.u64.u32 {P}temp, {P}capacity;
@{P}enabled mul.lo.u64 {P}offset, {P}index, {P}temp;
@{P}enabled mul.lo.u64 {P}offset, {P}offset, {RECORD.size};
@{P}enabled add.u64 {P}base, {P}base, {P}offset;
'''
    if storage == 'pooled-v1':
        text = text.replace(f'{P}scalar, 3;', f'{P}scalar, 4;')
        start = text.index(f'@{P}enabled cvt.u64.u32 {P}temp, {P}capacity;')
        text = text[:start] + f'@{P}enabled ld.global.u64 {P}limit, [{REGISTER}+40];\n'
    return text


def _active(predicate):
    return (f'and.pred {P}active, {P}enabled, {predicate};\n' if predicate else
            f'mov.pred {P}active, {P}enabled;\n')


def _timestamp(address):
    # Keep the timer's temporary local to the guarded store. A predicated move
    # into a live register can become an unconditional SASS CS2R plus SEL.
    return f'''
{{
.reg .b64 {P}tick;
@!{P}room bra {PREFIX}timer_done;
mov.u64 {P}tick, %globaltimer;
st.global.u64 [{address}], {P}tick;
{PREFIX}timer_done:
}}
'''


def _before(site, predicate, storage='per-thread-v1', suffix=''):
    if storage == 'pooled-v1':
        # The per-launch allocator saturates instead of wrapping. Reservation
        # does not replay the original instruction, even when capacity is lost.
        retry = PREFIX + f'pool_retry_{site}{suffix}'
        done = PREFIX + f'pool_done_{site}{suffix}'
        return '\n' + _active(predicate) + f'''
mov.pred {P}room, 0;
@!{P}active bra {done};
ld.volatile.global.u32 {P}used, [{P}state];
{retry}:
ld.volatile.global.u32 {P}scalar, [{P}limit];
setp.lt.u32 {P}room, {P}scalar, {P}capacity;
@!{P}room st.volatile.global.u32 [{P}state+4], 1;
@!{P}room bra {done};
cvt.u64.u32 {P}offset, {P}scalar;
add.u32 {P}scalar, {P}scalar, 1;
cvt.u32.u64 {P}expected, {P}offset;
atom.global.cas.b32 {P}scalar, [{P}limit], {P}expected, {P}scalar;
setp.ne.u32 {P}bad, {P}scalar, {P}expected;
@{P}bad bra {retry};
mul.lo.u64 {P}offset, {P}offset, 40;
add.u64 {P}slot, {P}base, {P}offset;
st.global.u64 [{P}slot+16], {site};
st.global.u32 [{P}slot+24], {P}used;
st.global.u32 [{P}slot+28], {STARTED};
st.global.u64 [{P}slot+32], {P}thread;
add.u32 {P}used, {P}used, 1;
st.volatile.global.u32 [{P}state], {P}used;
{done}:
''' + _timestamp(P + 'slot')
    return '\n' + _active(predicate) + f'''
@{P}active ld.volatile.global.u32 {P}used, [{P}state];
mov.pred {P}room, 0;
@{P}active setp.lt.u32 {P}room, {P}used, {P}capacity;
not.pred {P}bad, {P}room;
and.pred {P}bad, {P}bad, {P}active;
@{P}bad st.volatile.global.u32 [{P}state+4], 1;
@{P}room cvt.u64.u32 {P}offset, {P}used;
@{P}room mul.lo.u64 {P}offset, {P}offset, {RECORD.size};
@{P}room add.u64 {P}slot, {P}base, {P}offset;
@{P}room st.global.u64 [{P}slot+16], {site};
@{P}room st.global.u32 [{P}slot+24], {P}used;
@{P}room st.global.u32 [{P}slot+28], {STARTED};
@{P}room add.u32 {P}used, {P}used, 1;
@{P}room st.volatile.global.u32 [{P}state], {P}used;
''' + _timestamp(P + 'slot')


def _after():
    return _timestamp(P + 'slot+8') + f'@{P}room st.global.u32 [{P}slot+28], {COMMITTED};\n'


def _edge_location(function, flow, edge):
    """Insert at the source edge, never at a header revisited by backedges."""
    if edge.source == -1:
        return function.body_start, None
    last = function.instructions[flow.blocks[edge.source].stop - 1]
    terminal = last.opcode.split('.')[0] in {'bra', 'ret', 'exit', 'trap'}
    predicate = edge.predicate
    if predicate and edge.taken is False:
        predicate = predicate[1:] if predicate.startswith('!') else '!' + predicate
    return (last.start if terminal else last.end), predicate


def validate_loop_depth(interval_mode, loop_depth):
    if type(loop_depth) is not int or not 1 <= loop_depth <= 2147483647:
        raise PTXError('Loop depth must be a positive 32-bit integer')
    if interval_mode != 'loop-boundary' and loop_depth != 1:
        raise PTXError('Nondefault loop depth requires loop-boundary mode')
    return loop_depth


def loop_selection(loop_depth):
    return 'innermost_natural_loop' if loop_depth == 1 else 'nth_enclosing_natural_loop'


def _loop_probes(group, by_function, sites, edits, storage, loop_depth):
    """Replace in-loop probes with configurable-depth natural-loop episodes.

    Selection is intra-function and structural, not polling-protocol inference.
    An episode includes all code on its executed path, even if a predicated sys
    instruction never executes. Exit sites identify control flow, not success.
    """
    loops_manifest = []
    remaining = {}
    for key, candidates in by_function.items():
        function = group.definitions[key].function
        flow = analyze(function)
        owner = {i: block.index for block in flow.blocks for i in range(block.first, block.stop)}
        assigned = {}
        for ordinal, site in candidates:
            containers = [loop for loop in flow.loops if owner[site.instruction] in loop.blocks]
            if not containers:
                remaining.setdefault(key, []).append((ordinal, site))
                sites[ordinal - 1]['interval_kind'] = 'instruction'
                continue
            containers.sort(key=lambda value: len(value.blocks))
            if any(not inner.blocks < outer.blocks for inner, outer in zip(containers, containers[1:])):
                raise PTXError('Overlapping non-nested natural loops')
            selected_depth = min(loop_depth, len(containers))
            loop = containers[selected_depth - 1]
            sites[ordinal - 1].update(available_loop_depth=len(containers), selected_loop_depth=selected_depth)
            assigned.setdefault(loop.header, (loop, []))[1].append(ordinal)
        # Remove suppressed instruction sites from the record registry below;
        # their full inventory remains in each loop's contained_sites metadata.
        actions = {}
        for header, (loop, members) in sorted(assigned.items()):
            if not loop.exits:
                raise PTXError('Selected system loop has no statically accounted exit')
            loop_id = hashlib.sha256(json.dumps(
                ['loop-boundary-v1', group.modules[key[0]].sha256, key[1], header] +
                (['loop-depth', loop_depth] if loop_depth != 1 else [])
            ).encode()).hexdigest()
            label = P + f'loop_{len(loops_manifest)}_'
            declaration = (f'\n.reg .pred {label}room;\n.reg .b64 {label}slot;\n'
                           f'mov.pred {label}room, 0;\n')
            edits[key[0]].append((function.body_start, declaration))
            metadata = {'id': loop_id, 'unit': key[0], 'function': key[1], 'header': header,
                        'header_line': function.instructions[flow.blocks[header].first].line,
                        'blocks': sorted(loop.blocks), 'latches': sorted(loop.latches),
                        'entries': [asdict(edge) for edge in loop.entries],
                        'exits': [asdict(edge) for edge in loop.exits],
                        'contained_sites': [sites[index - 1].copy() for index in members],
                        'selection': loop_selection(loop_depth), 'loop_depth': loop_depth,
                        'depth_fallback': 'outermost_available', 'synchronization_success': 'unclassified'}
            loops_manifest.append(metadata)
            entry_site = len(sites) + 1
            for exit_index, edge in enumerate(loop.exits):
                ordinal = len(sites) + 1
                original = sites[members[0] - 1]
                sites.append({**original, 'ordinal': ordinal,
                              'id': hashlib.sha256(f'{loop_id}:{exit_index}'.encode()).hexdigest(),
                              'opcode': 'sys.loop', 'memory_order': 'mixed_or_conditional',
                              'observation': 'loop_execution_interval', 'interval_kind': 'loop',
                              'line': metadata['header_line'], 'loop_id': loop_id,
                              'exit_index': exit_index, 'exit_edge': asdict(edge),
                              'exit_outcome': 'exceptional' if edge.kind == 'trap' else 'unclassified'})
                position, predicate = _edge_location(function, flow, edge)
                text = '\n' + _active(predicate) + f'''
and.pred {P}room, {P}active, {label}room;
''' + _timestamp(label + 'slot+8') + f'''
@{P}room st.global.u64 [{label}slot+16], {ordinal};
@{P}room st.global.u32 [{label}slot+28], {COMMITTED};
@{P}active mov.pred {label}room, 0;
'''
                actions.setdefault(position, []).append((0, len(loop.blocks), text))
            for entry_index, edge in enumerate(loop.entries):
                position, predicate = _edge_location(function, flow, edge)
                text = _before(entry_site, predicate, storage, f'_entry{entry_index}') + f'''
@{P}active mov.pred {label}room, {P}room;
@{P}room mov.u64 {label}slot, {P}slot;
'''
                actions.setdefault(position, []).append((1, -len(loop.blocks), text))
        # Close inner loops first, then enter outer loops first on shared edges.
        for position, values in actions.items():
            edits[key[0]].append((position, ''.join(text for _, _, text in sorted(values))))
    kept = {ordinal for values in remaining.values() for ordinal, _ in values}
    sites[:] = [site for site in sites if site['ordinal'] in kept or site.get('interval_kind') == 'loop']
    return remaining, loops_manifest


def instrument_instructions(group, *, max_parameter_bytes, interval_mode, record_storage='per-thread-v1',
                            loop_depth=1, mbarrier='include'):
    if record_storage not in {'per-thread-v1', 'pooled-v1'}:
        raise PTXError('Unknown recording storage')
    if interval_mode not in {'instruction-only', 'loop-boundary'}:
        raise PTXError('Expected instruction-only or loop-boundary mode')
    validate_loop_depth(interval_mode, loop_depth)
    if mbarrier not in {'include', 'exclude'}:
        raise PTXError('Unknown mbarrier probe selection')
    if mbarrier == 'exclude':
        # Filter static candidates before loop selection; retain every original instruction.
        group = replace(group, modules={unit: replace(module, sites=tuple(
            site for site in module.sites if site.operation != 'mbarrier'))
            for unit, module in group.modules.items()})
    context = propagate_context(group, max_parameter_bytes=max_parameter_bytes)
    edits = {unit: list(values) for unit, values in context.edits.items()}
    sites, by_function = [], {}
    for unit in sorted(group.modules):
        module = group.modules[unit]
        for site in module.sites:
            ordinal = len(sites) + 1
            identity = hashlib.sha256(json.dumps([SCHEMA, module.sha256, site.function, site.instruction],
                                                 separators=(',', ':')).encode()).hexdigest()
            sites.append({'ordinal': ordinal, 'id': identity, 'unit': unit, **asdict(site)})
            by_function.setdefault((unit, site.function), []).append((ordinal, site))
    for key, definition in group.definitions.items():
        if key in by_function or definition.function.kind == 'entry':
            edits[definition.unit].append((definition.function.body_start, _prologue(record_storage)))
    loops = []
    if interval_mode == 'loop-boundary':
        # Collect edge edits separately: a non-loop instruction ending at an
        # entry edge must commit before that edge reserves another record.
        loop_edits = {unit: [] for unit in edits}
        by_function, loops = _loop_probes(group, by_function, sites, loop_edits, record_storage, loop_depth)
    for key, definition in group.definitions.items():
        function = definition.function
        unit = definition.unit
        for ordinal, site in by_function.get(key, ()):
            edits[unit].append((site.start, _before(ordinal, site.predicate, record_storage)))
            edits[unit].append((site.end, _after()))
        if function.kind == 'entry':
            for instruction in function.instructions:
                if instruction.opcode in {'ret', 'ret.uni', 'exit'}:
                    edits[unit].append((instruction.start, '\n' + _active(instruction.predicate) +
                                        f'@{P}active st.volatile.global.u32 [{P}state+8], 1;\n'))
    if interval_mode == 'loop-boundary':
        for unit, values in loop_edits.items():
            edits[unit].extend(values)
    sources = {unit: insertions(source, edits[unit]) for unit, source in group.sources.items()}
    manifest = {**context.manifest, 'kind': ('system_loop_and_instruction_intervals' if loops else
                                          'system_instruction_brackets'), 'site_instrumentation': True,
                'interval_mode': interval_mode,
                'site_selection': SITE_SELECTION if mbarrier == 'include' else SITE_SELECTION_NO_MBARRIER,
                'polling_intervals_validated': False,
                'sass_timer_placement_validated': False, 'sampling': 'launch_policy',
                'context_abi': CONTEXT_ABI, 'supported_sampling': ['all', 'warp', 'block'],
                'record_schema': SCHEMA, 'record_storage': record_storage, 'sites': sites}
    if interval_mode == 'loop-boundary':
        manifest.update(loops=loops, loop_selection=loop_selection(loop_depth), loop_depth=loop_depth,
                        depth_fallback='outermost_available',
                        loop_boundary_policy='source_edge_predicates',
                        synchronization_success='unclassified',
                        dynamic_sys_predicate_required=False)
    return ContextRewrite(sources, manifest, edits)
