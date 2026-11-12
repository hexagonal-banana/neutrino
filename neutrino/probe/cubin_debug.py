"""Bounds-checked cubin ELF/DWARF attribution, independent of probe policy.

NVIDIA's two line programs join at kernel-relative SASS PCs. The embedded
PTX is NUL-delimited: retain every delimiter, including initial empty lines.
"""
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import struct
import subprocess


class DebugError(ValueError):
    pass


class Reader:
    def __init__(self, data, pos=0, end=None):
        self.data, self.pos = data, pos
        self.end = len(data) if end is None else end
        if not 0 <= pos <= self.end <= len(data):
            raise DebugError("Invalid data bounds")

    def take(self, size):
        if size < 0 or self.pos + size > self.end:
            raise DebugError(f"Truncated data at {self.pos:#x}")
        result = self.data[self.pos:self.pos + size]
        self.pos += size
        return result

    def uint(self, size):
        return int.from_bytes(self.take(size), "little")

    def leb(self, signed=False):
        value = shift = 0
        for _ in range(10):
            byte = self.uint(1)
            value |= (byte & 127) << shift
            shift += 7
            if byte < 128:
                return value - (1 << shift) if signed and byte & 64 else value
        raise DebugError("Oversized LEB128")

    def string(self):
        end = self.data.find(b"\0", self.pos, self.end)
        if end < 0:
            raise DebugError("Unterminated string")
        try:
            return self.take(end - self.pos + 1)[:-1].decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise DebugError("Invalid UTF-8 debug string") from exc


@dataclass(frozen=True)
class Section:
    name: str
    index: int
    kind: int
    offset: int
    size: int
    link: int
    info: int
    entsize: int


class Cubin:
    def __init__(self, data: bytes):
        self.data = data
        if len(data) < 64 or data[:7] != b"\x7fELF\x02\x01\x01":
            raise DebugError("Expected little-endian ELF64 cubin")
        header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
        if header[2] != 190:
            raise DebugError("ELF machine is not CUDA")
        if header[3] != 1 or header[8] != 64:
            raise DebugError("Unsupported ELF header version/size")
        offset, entry_size, count, names_idx = header[6], header[11], header[12], header[13]
        if entry_size != 64 or not 0 < names_idx < count:
            raise DebugError("Unsupported ELF section table")
        table = Reader(data, offset).take(count * entry_size)
        raw = [struct.unpack_from("<IIQQQQIIQQ", table, i * 64) for i in range(count)]
        if raw[names_idx][1] != 3:
            raise DebugError("Section names must reference a string table")
        names = Reader(data, raw[names_idx][4]).take(raw[names_idx][5])
        self.sections = []
        for i, s in enumerate(raw):
            name = Reader(names, s[0]).string()
            if s[1] != 8:
                Reader(data, s[4]).take(s[5])
            self.sections.append(Section(name, i, s[1], s[4], s[5], s[6], s[7], s[9]))
        self.by_name = {s.name: s for s in self.sections}
        if len(self.by_name) != len(self.sections):
            raise DebugError("Duplicate ELF section names")

    @classmethod
    def read(cls, path):
        return cls(Path(path).read_bytes())

    def section(self, name):
        if name not in self.by_name:
            raise DebugError(f"Missing section {name}")
        s = self.by_name[name]
        if s.kind == 8:
            raise DebugError(f"Section {name} has no file-backed data")
        return self.data[s.offset:s.offset + s.size]

    def relocations(self, name):
        """Map relocated line-program addresses to (kernel, symbol addend)."""
        self.section(name)
        target = self.by_name[name].index
        result = {}
        for rel in self.sections:
            if rel.kind != 4 or rel.info != target:
                continue
            if rel.entsize != 24 or rel.size % 24 or rel.link >= len(self.sections):
                raise DebugError("Invalid ELF RELA section")
            symtab = self.sections[rel.link]
            if symtab.kind != 2 or symtab.entsize != 24 or symtab.size % 24:
                raise DebugError("Unsupported symbol table")
            for pos in range(rel.offset, rel.offset + rel.size, 24):
                address, info, addend = struct.unpack_from("<QQq", self.data, pos)
                if info & 0xffffffff != 2:
                    raise DebugError("Unsupported CUDA line relocation (expected R_CUDA_64)")
                if address + 8 > self.by_name[name].size or address in result:
                    raise DebugError("Invalid/duplicate line address relocation")
                symbol = info >> 32
                if (symbol + 1) * 24 > symtab.size:
                    raise DebugError("Relocation symbol out of bounds")
                _, _, _, shndx, value, _ = struct.unpack_from(
                    "<IBBHQQ", self.data, symtab.offset + symbol * 24)
                if shndx >= len(self.sections):
                    raise DebugError("Invalid relocated section")
                section = self.sections[shndx].name
                if not section.startswith(".text."):
                    raise DebugError(f"Line address does not reference code: {section}")
                if not 0 <= value + addend <= self.sections[shndx].size:
                    raise DebugError("Relocated address is outside its kernel")
                result[address] = (section[6:], value + addend)
        return result

    def ptx(self):
        try:
            text = self.section(".nv_debug_ptx_txt").replace(b"\0", b"\n").decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DebugError("Invalid embedded PTX text") from exc
        if not re.search(r"^\.version \d", text, re.M):
            raise DebugError("Embedded PTX is not supported text")
        return text


@dataclass(frozen=True)
class InlineContext:
    key: tuple
    name: str
    file: str
    line: int
    parent: tuple | None = None


@dataclass(frozen=True)
class LineRow:
    unit: int
    index: int
    kernel: str
    pc: int
    file: str
    line: int
    parent: int
    function: str
    end: bool = False
    sequence_base: int = 0


def line_rows(cubin, section_name):
    data = cubin.section(section_name)
    relocations = cubin.relocations(section_name)
    strings = None
    reader = Reader(data)
    output = []
    while reader.pos < reader.end:
        unit = reader.pos
        length = reader.uint(4)
        if length == 0:
            continue
        if length >= 0xfffffff0:
            raise DebugError("DWARF64 line units are unsupported")
        end = reader.pos + length
        r = Reader(data, reader.pos, end)
        version = r.uint(2)
        if version not in (2, 3, 4):
            raise DebugError(f"Unsupported line version {version}")
        prologue_length = r.uint(4)
        program = r.pos + prologue_length
        minimum = r.uint(1)
        if version >= 4 and r.uint(1) != 1:
            raise DebugError("VLIW line tables are unsupported")
        r.uint(1)  # default is_stmt does not affect attribution
        base = int.from_bytes(r.take(1), "little", signed=True)
        line_range, opcode_base = r.uint(1), r.uint(1)
        if not minimum or not line_range or not opcode_base:
            raise DebugError("Invalid line program header")
        arities = [0] + list(r.take(opcode_base - 1))
        dirs = [""]
        while True:
            name = r.string()
            if not name:
                break
            dirs.append(name)
        files = [""]
        while True:
            name = r.string()
            if not name:
                break
            directory, _, _ = r.leb(), r.leb(), r.leb()
            if directory >= len(dirs):
                raise DebugError("Invalid source directory")
            files.append(str(Path(dirs[directory]) / name))
        # CUDA adds the compilation unit's string-table base to the prologue.
        remaining = program - r.pos
        if remaining not in (0, 4, 8):
            raise DebugError(f"Unsupported CUDA line prologue extension ({remaining})")
        string_base = r.uint(remaining) if remaining else 0
        pc, line, file, parent, function, kernel = 0, 1, 1, 0, "", ""
        index = sequence_base = 0

        def emit(is_end=False):
            nonlocal index
            if not kernel or not 0 <= file < len(files):
                raise DebugError("Line row has no kernel/file identity")
            index += 1
            if not 0 <= pc <= cubin.by_name[f".text.{kernel}"].size or line < 0:
                raise DebugError("Line row is outside its kernel/source address space")
            if output and output[-1].unit == unit and not output[-1].end:
                if output[-1].kernel != kernel or output[-1].pc > pc:
                    raise DebugError("Nonmonotonic line sequence")
            # NVIDIA inline-parent operands restart at each end_sequence, even
            # when several kernels share one DWARF compilation unit. Normalize
            # them to unit-relative row identities without losing the raw base.
            output.append(LineRow(unit, index, kernel, pc, files[file], line,
                                  parent + sequence_base if parent else 0,
                                  function, is_end, sequence_base))

        while r.pos < end:
            op = r.uint(1)
            if op == 0:
                size = r.leb()
                ext_end = r.pos + size
                if not size or ext_end > end:
                    raise DebugError("Extended line opcode exceeds its compilation unit")
                ext = Reader(data, r.pos, ext_end)
                extended = ext.uint(1)
                if extended == 1:
                    emit(True)
                    sequence_base = index
                    pc, line, file, parent, function, kernel = 0, 1, 1, 0, "", ""
                elif extended == 2:
                    if ext.pos not in relocations or ext.end - ext.pos != 8:
                        raise DebugError("Missing/unsupported kernel address relocation")
                    kernel, addend = relocations[ext.pos]
                    pc = ext.uint(8) + addend
                elif extended == 0x90:
                    parent, offset = ext.leb(), ext.leb()
                    if strings is None:
                        strings = cubin.section(".debug_str")
                    function = Reader(strings, string_base + offset).string()
                else:
                    raise DebugError(f"Unsupported extended line opcode {extended:#x}")
                if ext.pos != ext_end:
                    raise DebugError(f"Invalid extended opcode {extended:#x} payload: {data[ext.pos:ext_end].hex()}")
                r.pos = ext_end
            elif op >= opcode_base:
                adjusted = op - opcode_base
                pc += (adjusted // line_range) * minimum
                line += base + adjusted % line_range
                emit()
            elif op == 1:
                emit()
            elif op == 2:
                pc += r.leb() * minimum
            elif op == 3:
                line += r.leb(True)
            elif op == 4:
                file = r.leb()
            elif op == 8:
                pc += ((255 - opcode_base) // line_range) * minimum
            elif op == 9:
                pc += r.uint(2)
            else:
                for _ in range(arities[op]):
                    r.leb()
        if kernel:
            raise DebugError("Unterminated line sequence")
        reader.pos = end
    return output


@lru_cache(maxsize=4096)
def demangle(name):
    if not name.startswith("_Z"):
        return name
    return subprocess.check_output(["c++filt", "--", name], text=True).strip()


def base_name(name):
    """Strip balanced template lists, return types, and the parameter list."""
    operators = []

    def hide_operator(match):
        operators.append(match[0])
        return f"__operator_{len(operators) - 1}__"

    text = re.sub(r"operator\s*(?:\(\)|\[\]|<=>|<<=?|>>=?|[<>]=?)",
                  hide_operator, demangle(name))
    result, depth = [], 0
    for char in text:
        if char == "<":
            depth += 1
        elif char == ">" and depth:
            depth -= 1
        elif not depth:
            if char == "(":
                break
            result.append(char)
    result = "".join(result).strip()
    if not result:
        return ""
    operator = result.find("operator ")
    if operator >= 0:
        result = result[:operator].split()[-1] + result[operator:] if result[:operator].strip() else result
    else:
        result = result.split()[-1]
    for index, operator in enumerate(operators):
        result = result.replace(f"__operator_{index}__", operator)
    return result


def matches(selector, name):
    return selector == name if selector.startswith("_Z") else selector == base_name(name)


@dataclass(frozen=True)
class AttributionInterval:
    """Raw evidence, not a claim that a scheduled PTX instruction owns a scope."""
    start: int
    end: int
    source: LineRow
    assembly: LineRow
    chain: tuple

    @property
    def exact(self):
        return self.start == self.source.pc == self.assembly.pc

    def manifest(self):
        return {"sass_interval": [self.start, self.end],
                "source_pc": self.source.pc, "ptx_pc": self.assembly.pc,
                "source_row": [self.source.unit, self.source.index],
                "source_sequence_base": self.source.sequence_base,
                "ptx_row": [self.assembly.unit, self.assembly.index],
                "source_line": self.source.line, "ptx_line": self.assembly.line,
                "exact_pc_anchor": self.exact,
                "ancestry": [list(context.key) for context in self.chain]}


class Attribution:
    def __init__(self, cubin):
        self.ptx = cubin.ptx()
        ptx_line_count = len(self.ptx.splitlines())
        self.sources = line_rows(cubin, ".debug_line")
        self.assembly = line_rows(cubin, ".nv_debug_line_sass")
        self.rows = {(row.unit, row.index): row for row in self.sources}
        self.lines = {}
        self.conflicts = {}
        self.variants = {}
        self.evidence = {}
        self.anchors = {}
        sources = {}
        for row in self.sources:
            sources.setdefault(row.kernel, []).append(row)
        for kernel, rows in sources.items():
            # A sequence may contain zero-length inline call-site rows. Keep them
            # in self.rows; only the last row at a PC owns subsequent code.
            by_pc = {row.pc: row for row in rows}
            pcs = sorted(by_pc)
            ptx_by_pc = {row.pc: row for row in self.assembly if row.kernel == kernel}
            ptx_pcs = sorted(ptx_by_pc)
            boundaries = sorted(set(pcs) | set(ptx_pcs))
            for pc, end_pc in zip(boundaries, boundaries[1:]):
                i = bisect_right(pcs, pc) - 1
                j = bisect_right(ptx_pcs, pc) - 1
                if i < 0 or j < 0 or by_pc[pcs[i]].end or ptx_by_pc[ptx_pcs[j]].end:
                    continue
                chain = self.chain(by_pc[pcs[i]])
                key = (kernel, ptx_by_pc[ptx_pcs[j]].line)
                if key[1] == 0:
                    continue
                if not 1 <= key[1] <= ptx_line_count:
                    raise DebugError("PTX line coordinate is outside embedded text")
                interval = AttributionInterval(pc, end_pc, by_pc[pcs[i]],
                                               ptx_by_pc[ptx_pcs[j]], chain)
                self.evidence.setdefault(key, []).append(interval)
                if interval.exact:
                    self.anchors.setdefault(key, set()).add(chain)
                self.variants.setdefault(key, set()).add(chain)
        for key, variants in self.variants.items():
            common = list(next(iter(variants)))
            for chain in variants:
                length = 0
                for a, b in zip(common, chain):
                    if a != b:
                        break
                    length += 1
                common = common[:length]
            self.lines[key] = tuple(common)
            if len(variants) > 1:
                self.conflicts[key] = variants

    def chain(self, row):
        chain, seen = [], set()
        while row.parent:
            key = (row.unit, row.parent)
            if key in seen or key not in self.rows:
                raise DebugError("Invalid inline-context ancestry")
            seen.add(key)
            parent = self.rows[key]
            if parent.kernel != row.kernel or parent.end or parent.index >= row.index:
                raise DebugError("Inline parent is not an earlier row in the same kernel")
            if row.function:
                chain.append((row, parent))
            row = parent
        result = []
        for row, parent in reversed(chain):
            result.append(InlineContext((row.unit, row.parent, row.function),
                                        row.function, parent.file, parent.line,
                                        result[-1].key if result else None))
        return tuple(result)

    def selected(self, kernel, selector):
        result = {}
        for (name, line), variants in self.variants.items():
            if name != kernel:
                continue
            for chain in variants:
                for context in chain:
                    if matches(selector, context.name):
                        result.setdefault(context, set()).add(line)
        return result
