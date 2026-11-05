"""Versioned per-launch, nonwrapping all-thread record ABI and strict decoding."""
from dataclasses import dataclass
import struct

from .identity import LaunchIdentity
from .sampling import MODES


SCHEMA = 1
DEFAULT_CAPACITY = 256
DEFAULT_BUDGET = 2 * 1024**3
CONTEXT = struct.Struct('<QQQQIIQQ')
STATE = struct.Struct('<IIII')  # used, overflow, entry completed, reserved
RECORD = struct.Struct('<QQQII')  # begin, end, site ordinal, thread ordinal, commit
HEADER = struct.Struct('<8sIIQQ32s')
MAGIC = b'NSYSREC1'
STARTED, COMMITTED = 1, 2


class RecordError(ValueError):
    pass


def _positive(value, bits, name):
    if type(value) is not int or not 0 < value < 2**bits:
        raise RecordError(f'Invalid {name}')


@dataclass(frozen=True)
class BufferLayout:
    threads: int
    capacity: int = DEFAULT_CAPACITY
    sampling: str = 'all'
    block_width: int = 1

    def __post_init__(self):
        _positive(self.threads, 64, 'thread count')
        _positive(self.capacity, 32, 'per-thread capacity')
        _positive(self.block_width, 32, 'block width')
        if not isinstance(self.sampling, str) or self.sampling not in MODES:
            raise RecordError('Unknown capture sampling')
        if self.threads % self.slots_per_block:
            raise RecordError('Incomplete compact block layout')
        if self.logical_threads >= 2**64:
            raise RecordError('Logical thread count overflow')
        if self.device_bytes >= 2**64:
            raise RecordError('Trace allocation size overflows the device address domain')

    @property
    def slots_per_block(self):
        return (self.block_width if self.sampling == 'all' else
                (self.block_width + 31) // 32 if self.sampling == 'warp' else 1)

    @property
    def logical_threads(self):
        return self.threads // self.slots_per_block * self.block_width

    def physical_thread(self, slot):
        if type(slot) is not int or not 0 <= slot < self.threads:
            raise RecordError('Invalid sampled slot')
        block, local = divmod(slot, self.slots_per_block)
        return block * self.block_width + (local * 32 if self.sampling == 'warp' else local)

    @property
    def state_bytes(self):
        return self.threads * STATE.size

    @property
    def record_bytes(self):
        return self.threads * self.capacity * RECORD.size

    @property
    def device_bytes(self):
        return CONTEXT.size + self.state_bytes + self.record_bytes

    def require_budget(self, retained_bytes=0, budget=DEFAULT_BUDGET):
        if type(retained_bytes) is not int or retained_bytes < 0:
            raise RecordError('Invalid retained-buffer byte count')
        _positive(budget, 64, 'retained-buffer budget')
        if retained_bytes + self.device_bytes > budget:
            raise RecordError('Retained-buffer budget exhausted; capture is invalid')

    @classmethod
    def from_dimensions(cls, grid, block, capacity=DEFAULT_CAPACITY, sampling='all'):
        if len(grid) != 3 or len(block) != 3:
            raise RecordError('Expected three-dimensional launch geometry')
        threads = 1
        for dimension in (*grid, *block):
            _positive(dimension, 32, 'launch dimension')
            threads *= dimension
        width = block[0] * block[1] * block[2]
        if not isinstance(sampling, str) or sampling not in MODES:
            raise RecordError('Unknown capture sampling')
        slots = width if sampling == 'all' else (width + 31) // 32 if sampling == 'warp' else 1
        return cls(threads // width * slots, capacity, sampling, width)


@dataclass(frozen=True)
class Interval:
    launch_id: str
    launch_ordinal: int
    thread: int
    ordinal: int
    site: int
    occurrence: int
    begin: int
    end: int


def encode_header(launch, layout):
    if not isinstance(launch, LaunchIdentity):
        raise RecordError('Raw records require an exact launch identity')
    return HEADER.pack(MAGIC, SCHEMA, layout.capacity, layout.threads,
                       launch.ordinal, bytes.fromhex(launch.id))


def decode(data, *, launch, layout, sites, loop_sites=()):
    """Validate every thread/slot before returning any intervals.

    No timestamp-nearest matching, overflow truncation, or incomplete-record
    repair is permitted. Callee/site and block/thread coordinates come from
    the immutable launch manifest; site IDs must be local to that artifact.
    """
    if not isinstance(launch, LaunchIdentity) or not isinstance(layout, BufferLayout):
        raise RecordError('Exact launch identity and buffer layout are required')
    data = memoryview(data)
    expected = HEADER.size + layout.state_bytes + layout.record_bytes
    if len(data) != expected or data[:HEADER.size].tobytes() != encode_header(launch, layout):
        raise RecordError('Raw record header/length does not match launch metadata')
    return _decode_slots(data[HEADER.size:], launch, layout, sites, packed=False, loop_sites=loop_sites)


def decode_native(data, *, launch, layout, sites, raw_format, loop_sites=()):
    """Decode native payloads whose identity is supplied by the process journal.

    states-packed-v1 stores every thread state, then only its used records in
    thread/ordinal order. It does not wrap, sample, or drop populated records.
    """
    if not isinstance(launch, LaunchIdentity) or not isinstance(layout, BufferLayout):
        raise RecordError('Exact launch identity and buffer layout are required')
    if raw_format not in {'states-packed-v1', 'states-fixed-v1', 'states-packed-v2', 'states-fixed-v2'}:
        raise RecordError('Unknown native record format')
    if raw_format.endswith('-v1') and layout.sampling != 'all':
        raise RecordError('Historical records cannot use sampled layouts')
    return _decode_slots(memoryview(data), launch, layout, sites, packed=raw_format.startswith('states-packed-'),
                         loop_sites=loop_sites)


def _decode_slots(data, launch, layout, sites, *, packed, loop_sites=()):
    if len(data) < layout.state_bytes or (not packed and len(data) != layout.state_bytes + layout.record_bytes):
        raise RecordError('Native record length does not match launch metadata')
    sites = frozenset(sites)
    if any(type(site) is not int or not 0 < site < 2**64 for site in sites):
        raise RecordError('Invalid static site registry')
    loop_sites = frozenset(loop_sites)
    if not loop_sites <= sites:
        raise RecordError('Unknown loop interval sites')
    result = []
    base = layout.state_bytes
    cursor = base
    empty = bytes(RECORD.size)
    for thread in range(layout.threads):
        used, overflow, completed, reserved = STATE.unpack_from(data, thread * STATE.size)
        if overflow:
            raise RecordError(f'Trace overflow on thread {thread}')
        if used > layout.capacity or completed != 1 or reserved:
            raise RecordError(f'Incomplete/malformed thread state: {thread}')
        occurrences = {}
        previous_begin, previous_end, stack = 0, 0, []
        if packed and cursor + used * RECORD.size > len(data):
            raise RecordError('Truncated packed records')
        for ordinal in range(used if packed else layout.capacity):
            offset = cursor if packed else base + (thread * layout.capacity + ordinal) * RECORD.size
            if packed:
                cursor += RECORD.size
            if ordinal >= used:
                if data[offset:offset + RECORD.size] != empty:
                    raise RecordError('Unexpected data in an unallocated record slot')
                continue
            begin, end, site, recorded_ordinal, commit = RECORD.unpack_from(data, offset)
            if commit != COMMITTED or recorded_ordinal != ordinal or site not in sites:
                raise RecordError('Incomplete, duplicate, or orphan interval')
            if not begin or end < begin or begin < previous_begin:
                raise RecordError('Malformed per-thread instruction timestamp ordering')
            if loop_sites:
                while stack and stack[-1][0] <= begin:
                    stack.pop()
                if stack and (not stack[-1][1] or end > stack[-1][0]):
                    raise RecordError('Crossing or non-loop enclosing intervals')
                stack.append((end, site in loop_sites))
            elif begin < previous_end:
                raise RecordError('Malformed per-thread instruction timestamp ordering')
            previous_begin, previous_end = begin, end
            occurrence = occurrences.get(site, 0)
            occurrences[site] = occurrence + 1
            result.append(Interval(launch.id, launch.ordinal, layout.physical_thread(thread),
                                   ordinal, site, occurrence, begin, end))
    if packed and cursor != len(data):
        raise RecordError('Unexpected trailing packed records')
    return result
