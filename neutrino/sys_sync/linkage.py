"""Whole-group call/ABI analysis, independent of library and kernel names.

A recovered writable function table is not assumed immutable. Indirect calls
require an emitted target-membership guard before their ABI may be changed.
This module performs no compilation, image replacement, or CUDA execution.
"""
from dataclasses import dataclass, replace
import re

from .abi import Layout, parameter_layout
from .control_flow import analyze
from .ptx import PTXError, parse, tokenize, _balanced


_SYMBOL = re.compile(r'[$A-Za-z_][\w$]*$')


def split_operands(tokens):
    """Split commas outside vector/address/argument-list delimiters."""
    groups, group, stack = [], [], []
    pairs = {')': '(', ']': '[', '}': '{'}
    for token in tokens:
        if token.text in ('(', '[', '{'):
            stack.append(token.text)
        elif token.text in pairs:
            if not stack or stack.pop() != pairs[token.text]:
                raise PTXError('Unbalanced operand delimiters')
        if token.text == ',' and not stack:
            if not group:
                raise PTXError('Empty operand')
            groups.append(tuple(group))
            group = []
        else:
            group.append(token)
    if stack or (groups and not group):
        raise PTXError('Incomplete operands')
    if group:
        groups.append(tuple(group))
    return tuple(groups)


def _argument_list(tokens):
    if len(tokens) < 2 or tokens[0].text != '(' or tokens[-1].text != ')':
        raise PTXError('Expected parenthesized call arguments')
    groups = split_operands(tokens[1:-1])
    if any(len(group) != 1 or not _SYMBOL.fullmatch(group[0].text) for group in groups):
        raise PTXError('Only .param call arguments are supported')
    return tuple(group[0].text for group in groups)


@dataclass(frozen=True)
class Signature:
    inputs: tuple
    returns: tuple


def _shape(layout):
    # Names are immaterial; scalar types, pointer annotations, arrays, and ABI
    # offsets are not. Do not equate an integer and floating-point register ABI.
    return tuple((parameter.offset, parameter.size, parameter.alignment,
                  tuple('_' if token.text == parameter.name else token.text
                        for token in tokenize(parameter.declaration)))
                 for parameter in layout.parameters)


def signature(source, function):
    returned = Layout((), 0) if function.return_start is None else parameter_layout(
        source, replace(function, parameter_start=function.return_start, parameter_end=function.return_end))
    return Signature(_shape(parameter_layout(source, function)), _shape(returned))


@dataclass(frozen=True)
class Definition:
    unit: str
    function: object
    signature: Signature

    @property
    def key(self):
        return self.unit, self.function.name


@dataclass(frozen=True)
class Table:
    unit: str
    name: str
    linkage: str
    storage: str
    targets: tuple[str, ...]
    has_null: bool


@dataclass(frozen=True)
class Call:
    caller: tuple[str, str]
    instruction: int
    targets: tuple[tuple[str, str], ...]
    external: str | None
    argument_end: int | None
    argument_count: int
    prototype_end: int | None
    indirect_operand: str | None
    table: tuple[str, str] | None


@dataclass
class LinkedGroup:
    sources: dict
    modules: dict
    definitions: dict
    declarations: dict
    calls: tuple[Call, ...]
    tables: tuple[Table, ...]
    external_functions: frozenset[str]


def _tables(unit, source, module):
    tokens = tokenize(source)
    token_indices = {token.start: index for index, token in enumerate(tokens)}
    ranges = []
    for function in module.functions:
        start = token_indices[function.start]
        if start and tokens[start - 1].text in {'.visible', '.extern', '.weak'}:
            start -= 1
        ranges.append((tokens[start].start, function.end))
    spans = iter(sorted(ranges))
    span = next(spans, None)
    outside = []
    for token in tokens:
        while span and token.start >= span[1]:
            span = next(spans, None)
        if not span or token.start < span[0]:
            outside.append(token)
    result = []
    for index, token in enumerate(outside):
        if token.text not in {'.global', '.const'}:
            continue
        cursor = index + 1
        if cursor < len(outside) and outside[cursor].text == '.align':
            cursor += 2
        if cursor >= len(outside) or outside[cursor].text not in {'.u64', '.b64'}:
            continue
        cursor += 1
        if cursor + 5 >= len(outside):
            continue
        name = outside[cursor].text
        suffix = [item.text for item in outside[cursor + 1:cursor + 6]]
        if not (suffix[0] == '[' and suffix[1].isdecimal() and suffix[2:] == [']', '=', '{']):
            continue
        close = _balanced(outside, cursor + 5, '{', '}')
        values = split_operands(outside[cursor + 6:close])
        # Numeric data arrays are not function tables. Mixed unsupported
        # expressions remain unresolved rather than receiving guessed targets.
        if not any(len(value) == 1 and _SYMBOL.fullmatch(value[0].text) for value in values):
            continue
        if (len(values) != int(suffix[1]) or any(len(value) != 1 or not (
                _SYMBOL.fullmatch(value[0].text) or value[0].text == '0') for value in values)):
            raise PTXError(f'Unsupported function-table initializer: {name}')
        linkage = outside[index - 1].text[1:] if index and outside[index - 1].text in {
            '.visible', '.weak'} else 'local'
        result.append(Table(unit, name, linkage, token.text[1:],
                            tuple(dict.fromkeys(value[0].text for value in values if value[0].text != '0')),
                            any(value[0].text == '0' for value in values)))
    return tuple(result)


def _directives(function):
    result = {}
    for directive in function.directives:
        if directive.opcode in {'.callprototype', '.calltargets'}:
            key = directive.scope, directive.labels[0]
            if key in result:
                raise PTXError('Duplicate call directive in one scope')
            result[key] = directive
    return result


def _visible(mapping, scope, name):
    return next((mapping[(scope[:depth], name)] for depth in range(len(scope), -1, -1)
                 if (scope[:depth], name) in mapping), None)


def _register_key(function, instruction, name):
    for depth in range(len(instruction.scope), -1, -1):
        scope = instruction.scope[:depth]
        for directive in function.directives:
            if directive.opcode != '.reg' or directive.scope != scope:
                continue
            declarations = split_operands(directive.operands[1:])
            for declaration in declarations:
                texts = [token.text for token in declaration]
                if texts == [name]:
                    return scope, name
                if len(texts) == 4 and texts[1] == '<' and texts[3] == '>' and texts[2].isdecimal():
                    prefix = texts[0]
                    suffix = name[len(prefix):] if name.startswith(prefix) else ''
                    if suffix.isdecimal() and int(suffix) < int(texts[2]):
                        return scope, name
    raise PTXError(f'Undeclared indirect-call register: {name}')


def _table_origin(function, call_instruction, lookup_table):
    """Conservative backward slice within one basic block, never nearest text.

    Only mov/cvta/add and a scalar table load are accepted. Other reaching
    definitions, predication, control-flow merges, and spilled pointers fail.
    Array bounds/mutations are NOT proved: the caller must emit a membership
    guard even when this slice identifies the original table initializer.
    """
    cfg = analyze(function)
    block = next(block for block in cfg.blocks if block.first <= call_instruction.index < block.stop)

    def origin(name, before, consumer, seen):
        table = lookup_table(name)
        if table is not None:
            return 'address', table
        if not name.startswith('%'):
            return None
        key = _register_key(function, consumer, name)
        marker = key, before
        if marker in seen:
            raise PTXError('Cyclic pointer provenance')
        seen = seen | {marker}
        for instruction in reversed(function.instructions[block.first:before]):
            groups = split_operands(instruction.operands)
            if not groups:
                continue
            # Calls may write their return arguments but never a .reg under
            # the supported .param-only ABI. Unknown multi-destination writes
            # to this register are rejected instead of ignored.
            first = groups[0]
            written = [token.text for token in first if token.text.startswith('%')]
            matches = [value for value in written
                       if _register_key(function, instruction, value) == key]
            if not matches:
                continue
            op = instruction.opcode
            if op.split('.')[0] in {'st', 'red', 'bra'}:
                continue
            if instruction.predicate or len(first) != 1:
                raise PTXError('Ambiguous indirect pointer definition')
            if op in {'mov.u64', 'mov.b64', 'cvta.global.u64', 'cvta.to.global.u64'} and len(groups) == 2:
                return origin(groups[1][0].text, instruction.index, instruction, seen) if len(groups[1]) == 1 else None
            if op in {'add.u64', 'add.s64'} and len(groups) == 3 and all(len(g) == 1 for g in groups):
                found = [origin(g[0].text, instruction.index, instruction, seen) for g in groups[1:]]
                addresses = [value for value in found if value is not None]
                return addresses[0] if len(addresses) == 1 and addresses[0][0] == 'address' else None
            if op in {'ld.global.u64', 'ld.global.b64', 'ld.const.u64', 'ld.const.b64'} and len(groups) == 2:
                address = groups[1]
                if len(address) == 3 and address[0].text == '[' and address[-1].text == ']':
                    value = origin(address[1].text, instruction.index, instruction, seen)
                    if value and value[0] == 'address':
                        return 'function', value[1]
            return None
        return None

    operands = split_operands(call_instruction.operands)
    target = operands[1] if operands[0][0].text == '(' else operands[0]
    value = origin(target[0].text, call_instruction.index, call_instruction, set())
    if not value or value[0] != 'function':
        raise PTXError(f'Unresolved indirect-call target at line {call_instruction.line}')
    return value[1]


def link_group(sources, *, external_functions=()):
    """Analyze a complete device-link group. No partial result authorizes launch.

    external_functions is an explicit ABI-unchanged leaf contract, not a name
    heuristic. Such callees remain excluded from system-site coverage and are
    listed in the result. Indirect external targets are never accepted.
    """
    sources = dict(sources)
    if not sources or any(not isinstance(key, str) or not key for key in sources):
        raise PTXError('A nonempty named PTX link group is required')
    modules = {key: parse(source) for key, source in sources.items()}
    if len({module.target for module in modules.values()}) != 1:
        raise PTXError('Cannot mix PTX target architectures in one link group')
    external_functions = frozenset(external_functions)
    definitions, exported, declarations = {}, {}, {}
    for unit, module in modules.items():
        for function in module.functions:
            item = Definition(unit, function, signature(sources[unit], function))
            if function.defined:
                definitions[item.key] = item
                if function.linkage != 'local':
                    if function.name in exported:
                        raise PTXError(f'Ambiguous exported definition: {function.name}')
                    exported[function.name] = item
            declarations.setdefault(item.key, []).append(item)

    def resolve(unit, name):
        result = definitions.get((unit, name)) or exported.get(name)
        if result is None and name not in external_functions:
            raise PTXError(f'Unresolved device function: {unit}:{name}')
        return result

    for (unit, name), items in declarations.items():
        definition = resolve(unit, name)
        expected = definition.signature if definition else items[0].signature
        if any(item.signature != expected or (definition and item.function.kind != definition.function.kind)
               for item in items):
            raise PTXError(f'Incompatible function declarations: {unit}:{name}')

    tables = tuple(table for unit, module in modules.items() for table in _tables(unit, sources[unit], module))
    local_tables, exported_tables = {}, {}
    for table in tables:
        key = table.unit, table.name
        if key in local_tables or (table.linkage != 'local' and table.name in exported_tables):
            raise PTXError(f'Ambiguous function table: {table.name}')
        local_tables[key] = table
        if table.linkage != 'local':
            exported_tables[table.name] = table

    calls = []
    for key, definition in definitions.items():
        unit, name = key
        function = definition.function
        directives = _directives(function)
        for instruction in function.instructions:
            if instruction.opcode not in {'call', 'call.uni'}:
                if instruction.opcode.startswith('call.'):
                    raise PTXError(f'Unsupported call opcode: {instruction.opcode}')
                continue
            groups = list(split_operands(instruction.operands))
            returns = _argument_list(groups.pop(0)) if groups and groups[0][0].text == '(' else ()
            if not groups or len(groups[0]) != 1:
                raise PTXError('Missing or malformed call target')
            target = groups.pop(0)[0].text
            args = groups.pop(0) if groups and groups[0][0].text == '(' else None
            arguments = _argument_list(args) if args else ()
            prototype_end = None
            table = None
            indirect = target.startswith('%')
            if indirect:
                if len(groups) != 1 or len(groups[0]) != 1:
                    raise PTXError('Indirect calls require one explicit call directive')
                directive = _visible(directives, instruction.scope, groups[0][0].text)
                if directive is None:
                    raise PTXError('Unresolved indirect-call directive')
                if directive.opcode == '.calltargets':
                    values = split_operands(directive.operands)
                    if not values or any(len(value) != 1 for value in values):
                        raise PTXError('Malformed explicit call targets')
                    target_names = tuple(value[0].text for value in values)
                    prototype_signature = None
                else:
                    tokens = directive.operands
                    return_close = _balanced(tokens, 0)
                    argument_open = return_close + 2
                    argument_close = _balanced(tokens, argument_open)
                    if argument_close != len(tokens) - 1 or tokens[return_close + 1].text != '_':
                        raise PTXError('Unsupported call prototype')
                    proto = replace(function, parameter_start=tokens[argument_open].start,
                                    parameter_end=tokens[argument_close].end,
                                    return_start=tokens[0].start, return_end=tokens[return_close].end)
                    prototype_signature = signature(sources[unit], proto)
                    prototype_end = tokens[argument_close].start
                    table = _table_origin(function, instruction, lambda symbol: local_tables.get(
                        (unit, symbol)) or exported_tables.get(symbol))
                    target_names = table.targets
                targets = tuple(resolve(table.unit if table else unit, symbol) for symbol in target_names)
                if not targets or any(item is None for item in targets):
                    raise PTXError('Indirect targets must all be defined in this link group')
                expected = prototype_signature or targets[0].signature
                if any(item.signature != expected for item in targets):
                    raise PTXError('Indirect target/prototype ABI mismatch')
            else:
                if groups:
                    raise PTXError('Unexpected direct-call operands')
                resolved = resolve(unit, target)
                targets = (resolved,) if resolved else ()
                if resolved:
                    expected = resolved.signature
                else:
                    declared = declarations.get((unit, target))
                    if not declared:
                        raise PTXError('External leaf contract requires an explicit declaration')
                    expected = declared[0].signature
            if any(item.function.kind != 'func' for item in targets):
                raise PTXError('Device-side kernel launches are unsupported')
            if len(arguments) != len(expected.inputs) or len(returns) != len(expected.returns):
                raise PTXError(f'Call argument count disagrees with ABI: {unit}:{name}:{instruction.line}')
            calls.append(Call(key, instruction.index, tuple(item.key for item in targets),
                              target if not targets else None, args[-1].start if args else None,
                              len(arguments), prototype_end, target if indirect else None,
                              (table.unit, table.name) if table else None))
    return LinkedGroup(sources, modules, definitions, declarations, tuple(calls), tables, external_functions)
