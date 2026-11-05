"""Atomic whole-group hidden-context propagation; no module-global launch state.

This stage changes signatures/calls, not synchronization sites. The output must
still pass assembly/link, resource, image-ownership, and runtime launch gates.
"""
from dataclasses import asdict, dataclass

from .abi import parameter_layout
from .ptx import PTXError, tokenize


PREFIX = '__neutrino_sys_'
PARAMETER = PREFIX + 'context'
REGISTER = '%' + PREFIX + 'context'


def insertions(source, edits):
    """Apply original-coordinate insertions deterministically without deletion."""
    by_position = {}
    for position, text in edits:
        if type(position) is not int or not 0 <= position <= len(source):
            raise PTXError('Invalid original insertion coordinate')
        by_position.setdefault(position, []).append(text)
    pieces, previous = [], 0
    for position in sorted(by_position):
        pieces.extend((source[previous:position], ''.join(by_position[position])))
        previous = position
    return ''.join((*pieces, source[previous:]))


@dataclass(frozen=True)
class ContextRewrite:
    sources: dict
    manifest: dict
    edits: dict


def propagate_context(group, *, max_parameter_bytes):
    if type(max_parameter_bytes) is not int or max_parameter_bytes <= 0:
        raise PTXError('Explicit device parameter-space limit is required')
    edits = {unit: [] for unit in group.sources}
    for source in group.sources.values():
        if any(token.text.lstrip('%').startswith(PREFIX) for token in tokenize(source)):
            raise PTXError('Reserved context symbol collides with original PTX')
    exported = {definition.function.name: definition for definition in group.definitions.values()
                if definition.function.linkage != 'local'}
    layouts = []
    for key, declarations in group.declarations.items():
        unit, name = key
        resolved = group.definitions.get(key) or exported.get(name)
        if resolved is None:
            continue
        for declaration in declarations:
            function = declaration.function
            layout = parameter_layout(group.sources[unit], function)
            if layout.context_offset() + 8 > max_parameter_bytes:
                raise PTXError(f'Hidden context exceeds device parameter space: {key}')
            edits[unit].append((function.parameter_end - 1,
                                (',\n' if layout.parameters else '') + f'.param .u64 {PARAMETER}\n'))
            if function.defined:
                edits[unit].append((function.body_start,
                                    f'\n.reg .u64 {REGISTER};\nld.param.u64 {REGISTER}, [{PARAMETER}];\n'))
                layouts.append({'unit': unit, 'function': name, 'kind': function.kind,
                                'original_layout': asdict(layout), 'context_offset': layout.context_offset()})
    prototypes = set()
    guard_declarations = set()
    additional_externs = {unit: set() for unit in group.sources}
    guard_reports = []
    for sequence, call in enumerate(group.calls):
        if call.external:
            continue
        unit, name = call.caller
        function = group.definitions[call.caller].function
        instruction = function.instructions[call.instruction]
        argument = PREFIX + f'argument_{sequence}'
        edits[unit].append((function.body_start, f'\n.param .b64 {argument};\n'))
        before = f'\nst.param.b64 [{argument}], {REGISTER};\n'
        if call.indirect_operand:
            # Mutable tables and out-of-range indexing cannot dispatch into a
            # callee with the old ABI. Check the ACTUAL pointer, not the table
            # address or a guessed function name. The failure path never calls
            # or retries the original collective.
            allowed, test = '%' + PREFIX + 'allowed', '%' + PREFIX + 'test'
            candidate = '%' + PREFIX + 'candidate'
            if call.caller not in guard_declarations:
                edits[unit].append((function.body_start,
                                    f'\n.reg .pred {allowed}, {test};\n.reg .b64 {candidate};\n'))
                guard_declarations.add(call.caller)
            before += f'mov.pred {allowed}, 0;\n'
            for target in call.targets:
                definition = group.definitions[target]
                if target[0] != unit:
                    if definition.function.linkage == 'local':
                        raise PTXError('Cannot guard a module-private indirect target across units')
                    local = group.definitions.get((unit, target[1]))
                    if local is not None and local.key != target:
                        raise PTXError('Indirect guard target is shadowed by a module-private symbol')
                    if (unit, target[1]) not in group.declarations:
                        additional_externs[unit].add(target)
                before += (f'mov.u64 {candidate}, {target[1]};\n'
                           f'setp.eq.u64 {test}, {call.indirect_operand}, {candidate};\n'
                           f'or.pred {allowed}, {allowed}, {test};\n')
            before += f'not.pred {test}, {allowed};\n'
            if instruction.predicate:
                before += f'and.pred {test}, {test}, {instruction.predicate};\n'
            before += f'@{test} trap;\n'
            guard_reports.append({'unit': unit, 'function': name, 'instruction': call.instruction,
                                  'target_count': len(call.targets), 'table': call.table,
                                  'failure': 'trap_before_unsafe_call'})
        edits[unit].append((instruction.start, before))
        if call.argument_end is None:
            edits[unit].append((instruction.end - 1, f', ({argument})'))
        else:
            edits[unit].append((call.argument_end, (', ' if call.argument_count else '') + argument))
        if call.prototype_end is not None and (unit, call.prototype_end) not in prototypes:
            edits[unit].append((call.prototype_end,
                                (', ' if call.argument_count else '') + f'.param .u64 {PARAMETER}'))
            prototypes.add((unit, call.prototype_end))
    for unit, targets in additional_externs.items():
        if not targets:
            continue
        tokens = tokenize(group.sources[unit])
        address = next(index for index, token in enumerate(tokens) if token.text == '.address_size')
        position = tokens[address + 1].end
        for target in sorted(targets):
            definition = group.definitions[target]
            function = definition.function
            header = group.sources[target[0]][function.start:function.parameter_end - 1]
            comma = ', ' if definition.signature.inputs else ''
            edits[unit].append((position, f'\n.extern {header}{comma}.param .u64 {PARAMETER});\n'))
    rewritten = {unit: insertions(source, edits[unit]) for unit, source in group.sources.items()}
    manifest = {'schema': 1, 'kind': 'hidden_launch_context', 'runtime_verified': False,
                'site_instrumentation': False,
                'original_sha256': {unit: module.sha256 for unit, module in group.modules.items()},
                'layouts': layouts, 'indirect_guards': guard_reports,
                'external_leaf_contracts': sorted(group.external_functions)}
    return ContextRewrite(rewritten, manifest, edits)
