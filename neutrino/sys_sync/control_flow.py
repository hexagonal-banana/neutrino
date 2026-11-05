"""Exact static branch edges and natural loops; no inferred synchronization protocol."""
from dataclasses import dataclass

from .ptx import PTXError


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    kind: str
    predicate: str | None = None
    taken: bool | None = None


@dataclass(frozen=True)
class Block:
    index: int
    first: int
    stop: int


@dataclass(frozen=True)
class Loop:
    header: int
    blocks: frozenset[int]
    latches: frozenset[int]
    entries: tuple[Edge, ...]
    exits: tuple[Edge, ...]


@dataclass(frozen=True)
class ControlFlow:
    blocks: tuple[Block, ...]
    edges: tuple[Edge, ...]
    reachable: frozenset[int]
    loops: tuple[Loop, ...]
    # Synthetic entry is -1; synthetic function exit is len(blocks).


def analyze(function):
    instructions = function.instructions
    if not function.defined or not instructions:
        raise PTXError('Control flow requires a nonempty function definition')
    labels = {}
    leaders = {0}
    for instruction in instructions:
        for label, scope in zip(instruction.labels, instruction.label_scopes):
            key = (scope, label)
            if key in labels:
                raise PTXError(f'Duplicate branch label in one lexical scope: {label}')
            labels[key] = instruction.index
            leaders.add(instruction.index)
    targets = {}
    for instruction in instructions:
        op = instruction.opcode.split('.')[0]
        if op in {'brx', 'jmp', 'brkpt'}:
            raise PTXError(f'Unsupported control-flow instruction: {instruction.opcode}')
        if op == 'bra':
            if len(instruction.operands) != 1:
                raise PTXError(f'Unresolved branch at line {instruction.line}')
            label = instruction.operands[0].text
            target = next((labels[(instruction.scope[:depth], label)]
                           for depth in range(len(instruction.scope), -1, -1)
                           if (instruction.scope[:depth], label) in labels), None)
            if target is None:
                raise PTXError(f'Unresolved branch at line {instruction.line}')
            targets[instruction.index] = target
            leaders.add(target)
        if op in {'bra', 'ret', 'exit', 'trap'} and instruction.index + 1 < len(instructions):
            leaders.add(instruction.index + 1)
    boundaries = sorted(leaders) + [len(instructions)]
    blocks = tuple(Block(index, start, stop) for index, (start, stop) in
                   enumerate(zip(boundaries, boundaries[1:])))
    owner = {instruction: block.index for block in blocks for instruction in range(block.first, block.stop)}
    exit_node = len(blocks)
    edges = [Edge(-1, 0, 'entry')]
    for block in blocks:
        last = instructions[block.stop - 1]
        op = last.opcode.split('.')[0]
        fallthrough = block.index + 1
        if op in {'bra', 'ret', 'exit', 'trap'}:
            # A trap is an exceptional terminal edge, never normal completion.
            target = owner[targets[last.index]] if op == 'bra' else exit_node
            edges.append(Edge(block.index, target, 'branch' if op == 'bra' else op,
                              last.predicate, True if last.predicate else None))
            if last.predicate:
                edges.append(Edge(block.index, fallthrough, 'fallthrough', last.predicate, False))
        else:
            edges.append(Edge(block.index, fallthrough, 'fallthrough'))
    successors = {node: [] for node in range(exit_node + 1)}
    predecessors = {node: [] for node in range(exit_node + 1)}
    for edge in edges:
        if edge.source >= 0:
            successors[edge.source].append(edge.target)
            predecessors[edge.target].append(edge.source)
    reachable, pending = set(), [0]
    while pending:
        node = pending.pop()
        if node not in reachable:
            reachable.add(node)
            pending.extend(successors[node])
    reachable.discard(exit_node)
    # Bit sets avoid a Python set object per block/dominator pair.
    all_bits = sum(1 << node for node in reachable)
    dominators = {node: (1 if node == 0 else all_bits) for node in reachable}
    changed = True
    while changed:
        changed = False
        for node in sorted(reachable - {0}):
            bits = all_bits
            for predecessor in predecessors[node]:
                if predecessor in reachable:
                    bits &= dominators[predecessor]
            bits |= 1 << node
            if bits != dominators[node]:
                dominators[node] = bits
                changed = True
    backedges = {}
    for edge in edges:
        if (edge.source in reachable and edge.target in reachable and
                dominators[edge.source] & (1 << edge.target)):
            backedges.setdefault(edge.target, set()).add(edge.source)
    remaining = {node: [] for node in reachable}
    indegree = dict.fromkeys(reachable, 0)
    for edge in edges:
        if (edge.source in reachable and edge.target in reachable and
                edge.source not in backedges.get(edge.target, ())):
            remaining[edge.source].append(edge.target)
            indegree[edge.target] += 1
    ready = [node for node in reachable if indegree[node] == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for target in remaining[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(reachable):
        raise PTXError('Irreducible control-flow cycle requires explicit interval analysis')
    loops = []
    for header, latches in sorted(backedges.items()):
        members, pending = {header}, list(latches - {header})
        while pending:
            node = pending.pop()
            if node not in members:
                members.add(node)
                pending.extend(predecessor for predecessor in predecessors[node] if predecessor in reachable)
        entries = tuple(edge for edge in edges if edge.target in members and edge.source not in members
                        and (edge.source == -1 or edge.source in reachable))
        if any(edge.target != header for edge in entries):
            raise PTXError('Natural loop has an unaccounted entry edge')
        exits = tuple(edge for edge in edges if edge.source in members and edge.target not in members)
        loops.append(Loop(header, frozenset(members), frozenset(latches), entries, exits))
    return ControlFlow(blocks, tuple(edges), frozenset(reachable), tuple(loops))
