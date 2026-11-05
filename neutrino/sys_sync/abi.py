"""Explicit PTX parameter layouts for safe argument marshalling.

Unsupported declarations fail closed. Layout alone is not permission to launch
an altered kernel: complete call-graph transformation and assembly validation
are separate gates.
"""
from dataclasses import dataclass
import re

from .ptx import PTXError, tokenize


@dataclass(frozen=True)
class Parameter:
    name: str
    offset: int
    size: int
    alignment: int
    declaration: str


@dataclass(frozen=True)
class Layout:
    parameters: tuple[Parameter, ...]
    size: int

    def context_offset(self):
        return (self.size + 7) // 8 * 8


_TYPE = re.compile(r'\.([busf])(8|16|32|64|128)$')
_SYMBOL = re.compile(r'[$A-Za-z_][\w$]*$')


def parameter_layout(source, function):
    text = source[function.parameter_start:function.parameter_end]
    tokens = tokenize(text)
    if not tokens or tokens[0].text != '(' or tokens[-1].text != ')':
        raise PTXError('Invalid function parameter coordinates')
    groups, group = [], []
    for token in tokens[1:-1]:
        if token.text == ',':
            if not group:
                raise PTXError('Empty parameter declaration')
            groups.append(group)
            group = []
        else:
            group.append(token)
    if group:
        groups.append(group)
    elif groups:
        raise PTXError('Trailing parameter comma')
    parameters, offset = [], 0
    for group in groups:
        parts = [token.text for token in group]
        if not parts or parts.pop(0) != '.param':
            raise PTXError('Only .param function arguments are supported')
        alignment = None
        if parts and parts[0] == '.align':
            if len(parts) < 2 or not parts[1].isdecimal():
                raise PTXError('Nonconstant parameter alignment')
            alignment = int(parts[1])
            parts = parts[2:]
        kind = _TYPE.fullmatch(parts.pop(0)) if parts else None
        if kind is None:
            raise PTXError('Unsupported parameter type/vector declaration')
        element_size = int(kind[2]) // 8
        if alignment is None:
            alignment = element_size
        if alignment < element_size or alignment > 128 or alignment & (alignment - 1):
            raise PTXError('Invalid parameter alignment')
        # Pointer pointee alignment is distinct from the parameter slot alignment.
        if parts and parts[0] == '.ptr':
            if element_size != 8 or kind[1] not in {'b', 'u'}:
                raise PTXError('Expected a 64-bit pointer parameter')
            parts.pop(0)
            if parts and parts[0] in {'.global', '.const', '.local', '.shared'}:
                parts.pop(0)
            if parts and parts[0] == '.align':
                if (len(parts) < 2 or not parts[1].isdecimal() or int(parts[1]) <= 0 or
                        int(parts[1]) & (int(parts[1]) - 1)):
                    raise PTXError('Invalid pointee alignment')
                parts = parts[2:]
        if not parts or not _SYMBOL.fullmatch(parts[0]):
            raise PTXError('Missing parameter name')
        name, parts = parts[0], parts[1:]
        count = 1
        if parts:
            if len(parts) != 3 or parts[0] != '[' or parts[2] != ']' or not parts[1].isdecimal():
                raise PTXError('Unsupported parameter array declaration')
            count = int(parts[1])
            if count <= 0:
                raise PTXError('Parameter arrays require a positive constant extent')
        size = element_size * count
        offset = (offset + alignment - 1) // alignment * alignment
        if offset + size >= 2**32:
            raise PTXError('Unrepresentable parameter layout')
        if any(parameter.name == name for parameter in parameters):
            raise PTXError('Duplicate parameter symbol')
        parameters.append(Parameter(name, offset, size, alignment, text[group[0].start:group[-1].end]))
        offset += size
    return Layout(tuple(parameters), offset)


def append_context_parameter(source, function, name, *, max_parameter_bytes):
    """Return an exact insertion and original layout, without altering user bytes.

    The caller applies all linked-group edits together; this helper does not
    rewrite an entry in isolation or choose a device's parameter-space limit.
    """
    if not isinstance(name, str) or not _SYMBOL.fullmatch(name):
        raise PTXError('Invalid hidden parameter symbol')
    if any(token.text == name for token in tokenize(source)):
        raise PTXError('Hidden context symbol collides with the original module')
    if type(max_parameter_bytes) is not int or max_parameter_bytes <= 0:
        raise PTXError('Explicit device parameter-space limit is required')
    layout = parameter_layout(source, function)
    if layout.context_offset() + 8 > max_parameter_bytes:
        raise PTXError('Hidden context exceeds device parameter space')
    insertion = (',\n' if layout.parameters else '') + f'\t.param .u64 {name}\n'
    return function.parameter_end - 1, insertion, layout
