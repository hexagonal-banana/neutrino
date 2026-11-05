"""Lossless PTX inventory of acquire-system memory, system fences, and mbarrier.

This is structural analysis, not a PTX validator or a polling-loop recognizer.
No kernel/library name participates in site selection. All mbarrier operations
are selected; unsupported explicit system opcodes fail closed.
"""
from dataclasses import dataclass
import hashlib
import re


class PTXError(ValueError):
    pass


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    line: int


# Keep quoted strings and comments opaque, including braces and fake opcodes.
_LEX = re.compile(r'(?P<space>\s+)|(?P<line>//[^\n]*)|(?P<comment>/\*[\s\S]*?\*/)|'
                  r'(?P<string>"(?:\\[^\n]|[^"\\\n])*")|'
                  r'(?P<word>[.%$A-Za-z_][\w.$%]*(?:::[\w]+)*)|'
                  r'(?P<number>\d[\w.]*)|(?P<punct>[^\s])')


def tokenize(source):
    if not isinstance(source, str) or '\x00' in source:
        raise PTXError('Expected non-NUL PTX text')
    tokens, line = [], 1
    for match in _LEX.finditer(source):
        text = match.group()
        if match.lastgroup not in ('space', 'line', 'comment'):
            if text == '"' or (text == '/' and source[match.start():].startswith('/*')):
                raise PTXError(f'Unterminated string/comment at line {line}')
            tokens.append(Token(text, match.start(), match.end(), line))
        line += text.count('\n')
    return tokens


def _balanced(tokens, start, opening='(', closing=')'):
    if start >= len(tokens) or tokens[start].text != opening:
        raise PTXError(f'Expected {opening}')
    depth = 0
    for index in range(start, len(tokens)):
        text = tokens[index].text
        depth += (text == opening) - (text == closing)
        if depth == 0:
            return index
    raise PTXError(f'Unclosed {opening} at line {tokens[start].line}')


@dataclass(frozen=True)
class Instruction:
    index: int
    opcode: str
    predicate: str | None
    operands: tuple[Token, ...]
    start: int
    end: int
    line: int
    labels: tuple[str, ...]
    scope: tuple[int, ...]
    label_scopes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Directive:
    opcode: str
    operands: tuple[Token, ...]
    start: int
    end: int
    scope: tuple[int, ...]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class Function:
    name: str
    kind: str
    start: int
    end: int
    parameter_start: int
    parameter_end: int
    body_start: int | None
    body_end: int | None
    instructions: tuple[Instruction, ...]
    directives: tuple[Directive, ...]
    linkage: str
    return_start: int | None
    return_end: int | None

    @property
    def defined(self):
        return self.body_start is not None


@dataclass(frozen=True)
class Site:
    function: str
    instruction: int
    opcode: str
    predicate: str | None
    start: int
    end: int
    line: int
    operation: str
    memory_order: str
    observation: str


@dataclass(frozen=True)
class Module:
    sha256: str
    target: str
    address_size: int
    functions: tuple[Function, ...]
    sites: tuple[Site, ...]


SITE_SELECTION = 'acquire_sys_system_fences_mbarrier_try_wait_v1'


def system_scope(opcode):
    """Classify selected opcodes (historical API name, no longer sys-only).

    Select acquire/acq_rel system memory, system-scope fence/membar, and
    mbarrier.try_wait operations. Other barrier operations are not selected.
    No instruction bracket asserts remote consumption or barrier completion.
    """
    parts = opcode.split('.')
    operation = parts[0]
    fences = {'fence', 'membar'}
    barriers = {'mbarrier'}
    if operation in barriers and (len(parts) < 2 or parts[1] != 'try_wait'):
        return None
    if operation not in barriers and 'sys' not in parts:
        return None
    if operation == 'multimem' and len(parts) > 1:
        operation += '.' + parts[1]
    supported = {'ld', 'st', 'atom', 'red', 'multimem.ld_reduce', 'multimem.st',
                 'multimem.red'} | fences | barriers
    if operation not in supported:
        raise PTXError(f'Unsupported explicit system-scope opcode: {opcode}')
    orders = set(parts) & {'relaxed', 'acquire', 'release', 'acq_rel', 'sc'}
    if len(orders) > 1:
        raise PTXError(f'Conflicting memory orders: {opcode}')
    order = next(iter(orders), 'implicit')
    if operation in fences:
        observation = 'fence_instruction'
    elif operation in barriers:
        observation = 'barrier_instruction'
    elif order in {'acquire', 'acq_rel'}:
        observation = 'ordered_memory_instruction'
    else:
        return None
    return operation, order, observation


def _instructions(tokens):
    result, directives, labels, index = [], [], [], 0
    scope, scope_sequence = (), 0
    while index < len(tokens):
        token = tokens[index]
        if token.text in ('{', '}'):
            if token.text == '{':
                scope_sequence += 1
                scope = (*scope, scope_sequence)
            else:
                if not scope:
                    raise PTXError('Unmatched instruction-scope closing brace')
                scope = scope[:-1]
            index += 1
            continue
        if index + 1 < len(tokens) and tokens[index + 1].text == ':':
            labels.append((token.text, scope))
            index += 2
            continue
        # These debug directives terminate at newline, not semicolon.
        if token.text in {'.loc', '.file'}:
            index += 1
            while index < len(tokens) and tokens[index].line == token.line:
                index += 1
            continue
        end = index
        while end < len(tokens) and tokens[end].text != ';':
            end += 1
        if end == len(tokens):
            raise PTXError(f'Unterminated statement at line {token.line}')
        if token.text.startswith('.'):
            # Declarations, call prototypes, and pragmas are not instructions.
            # A label on a callprototype is not a branch target.
            directive_labels = ()
            if token.text in {'.callprototype', '.calltargets'}:
                if len(labels) != 1 or labels[0][1] != scope:
                    raise PTXError('Call directive requires one local label')
                directive_labels = (labels[0][0],)
                labels.clear()
            directives.append(Directive(token.text, tuple(tokens[index + 1:end]), token.start,
                                        tokens[end].end, scope, directive_labels))
            index = end + 1
            continue
        op_index, predicate = index, None
        if token.text == '@':
            op_index += 1
            negate = tokens[op_index].text == '!'
            op_index += int(negate)
            if op_index >= end or not re.fullmatch(r'[%$A-Za-z_][\w$%]*', tokens[op_index].text):
                raise PTXError(f'Invalid predicate at line {token.line}')
            predicate = ('!' if negate else '') + tokens[op_index].text
            op_index += 1
        if op_index >= end or not re.fullmatch(r'[a-z][\w.:]*', tokens[op_index].text):
            raise PTXError(f'Invalid instruction at line {token.line}')
        result.append(Instruction(len(result), tokens[op_index].text, predicate,
                                  tuple(tokens[op_index + 1:end]), token.start,
                                  tokens[end].end, token.line, tuple(label for label, _ in labels),
                                  scope, tuple(label_scope for _, label_scope in labels)))
        labels.clear()
        index = end + 1
    if scope:
        raise PTXError('Unclosed instruction scope')
    if labels:
        raise PTXError('Trailing labels require explicit control-flow handling')
    return tuple(result), tuple(directives)


def parse(source):
    """Inventory all entries and device functions without pruning or rewriting."""
    tokens = tokenize(source)
    functions, sites, targets, addresses = [], [], [], []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.text in {'.target', '.address_size'}:
            if index + 1 >= len(tokens):
                raise PTXError(f'Missing {token.text} value')
            (targets if token.text == '.target' else addresses).append(tokens[index + 1].text)
        if token.text not in {'.entry', '.func'}:
            if token.text == '{':
                index = _balanced(tokens, index, '{', '}')
            index += 1
            continue
        kind, start = token.text[1:], token.start
        linkage = tokens[index - 1].text[1:] if index and tokens[index - 1].text in {
            '.visible', '.extern', '.weak'} else 'local'
        cursor = index + 1
        return_start = return_end = None
        if cursor < len(tokens) and tokens[cursor].text == '(':
            close = _balanced(tokens, cursor)
            return_start, return_end = tokens[cursor].start, tokens[close].end
            cursor = close + 1  # Device-function return parameters.
        if cursor >= len(tokens) or not re.fullmatch(r'[$A-Za-z_][\w$]*', tokens[cursor].text):
            raise PTXError(f'Missing function symbol at line {token.line}')
        name = tokens[cursor].text
        cursor += 1
        parameter_start = cursor
        cursor = _balanced(tokens, cursor)
        parameter_end = cursor
        cursor += 1
        while cursor < len(tokens) and tokens[cursor].text not in {'{', ';'}:
            if tokens[cursor].text in {'.entry', '.func'}:
                raise PTXError(f'Missing function body/terminator: {name}')
            cursor += 1
        if cursor >= len(tokens):
            raise PTXError(f'Unterminated function: {name}')
        body_start = body_end = None
        instructions = directives = ()
        if tokens[cursor].text == '{':
            close = _balanced(tokens, cursor, '{', '}')
            body_start, body_end = tokens[cursor].end, tokens[close].start
            instructions, directives = _instructions(tokens[cursor + 1:close])
            cursor = close
        function = Function(name, kind, start, tokens[cursor].end,
                            tokens[parameter_start].start, tokens[parameter_end].end,
                            body_start, body_end, instructions, directives, linkage,
                            return_start, return_end)
        functions.append(function)
        for instruction in instructions:
            classification = system_scope(instruction.opcode)
            if classification:
                sites.append(Site(name, instruction.index, instruction.opcode, instruction.predicate,
                                  instruction.start, instruction.end, instruction.line, *classification))
        index = cursor + 1
    if len(targets) != 1 or not re.fullmatch(r'sm_\d+[af]?', targets[0]):
        raise PTXError('Expected exactly one explicit PTX target')
    if addresses != ['64']:
        raise PTXError('Only one 64-bit PTX address domain is supported')
    definitions = [function.name for function in functions if function.defined]
    if len(set(definitions)) != len(definitions):
        raise PTXError('Duplicate function definitions in one PTX module')
    return Module(hashlib.sha256(source.encode('utf-8')).hexdigest(), targets[0], 64,
                  tuple(functions), tuple(sites))
