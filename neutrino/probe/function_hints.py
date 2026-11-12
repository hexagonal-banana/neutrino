"""Versioned source hints carried by embedded PTX, independent of debug tables.

Markers are compiler-visible inline assembly, not GPU synchronization. Their
ordering guarantees must be validated for the supported compiler and workload.
"""
from dataclasses import dataclass
import json
import re
import struct
import zlib

from .cubin_debug import DebugError, base_name

HINT_ABI = 1
MAGIC = 0x4E485431
END_MAGIC = 0x3144484E
MAX_PAYLOAD = 4096
MOV = re.compile(r"mov\.u32\s+([%\w$]+)\s*,\s*(0[xX][0-9a-fA-F]+|\d+)\s*;")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def validate_identity(value):
    required = {"function", "file", "line", "source_sha256", "rule_sha256"}
    if not isinstance(value, dict) or set(value) not in (required, required | {"symbol"}):
        raise DebugError("Invalid source-hint identity fields")
    if not isinstance(value["function"], str) or not re.fullmatch(
            r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)+", value["function"]):
        raise DebugError("Source hints require an exact qualified base function name")
    if not isinstance(value["file"], str) or not value["file"] or "\0" in value["file"]:
        raise DebugError("Invalid source-hint logical file")
    if type(value["line"]) is not int or value["line"] < 1:
        raise DebugError("Invalid source-hint source line")
    for key in ("source_sha256", "rule_sha256"):
        if not isinstance(value[key], str) or not re.fullmatch(r"[0-9a-f]{64}", value[key]):
            raise DebugError(f"Invalid source-hint {key}")
    if "symbol" in value and (not isinstance(value["symbol"], str) or
                               not value["symbol"].startswith("_Z") or
                               base_name(value["symbol"]) != value["function"]):
        raise DebugError("Invalid exact source-hint symbol")
    return value


def marker_ptx(identity, kind):
    """Create a scoped, state-free literal marker. No timestamps or memory I/O."""
    validate_identity(identity)
    if type(kind) is not int or kind not in (0, 1):
        raise DebugError("Invalid source-hint event kind")
    payload = canonical(identity)
    if len(payload) > MAX_PAYLOAD:
        raise DebugError("Oversized source-hint payload")
    padded = payload + b"\0" * (-len(payload) % 4)
    words = [MAGIC, HINT_ABI, kind, len(payload), *struct.unpack(f"<{len(padded) // 4}I", padded),
             zlib.crc32(payload), END_MAGIC]
    return "{\n.reg .b32 __nh_word;\n" + "".join(
        f"mov.u32 __nh_word, 0x{word:08x};\n" for word in words) + "}\n"


def marker_cpp(identity, kind):
    # JSON's escapes for this ASCII PTX string are valid C++ string escapes.
    return "asm volatile(" + json.dumps(marker_ptx(identity, kind)) + ' ::: "memory");'


@dataclass(frozen=True)
class Hint:
    first_line: int
    last_line: int
    declaration_line: int
    kind: int
    identity: dict

    def manifest(self):
        return {"first_line": self.first_line, "last_line": self.last_line,
                "kind": self.kind, "identity": self.identity, "abi": HINT_ABI}


def decode_hints(ptx):
    """Decode complete canonical blocks, accepting arbitrary local register names.

    A recognizable framing word outside a valid block is an error, including a
    damaged first word with an intact footer. Comments and debug rows carry no
    hint identity and are never interpreted as markers.
    """
    rows = [(n, line.split("//", 1)[0].strip()) for n, line in enumerate(ptx.splitlines(), 1)]
    rows = [(n, text) for n, text in rows if text]
    result, consumed = [], set()
    for index, (number, text) in enumerate(rows):
        match = MOV.fullmatch(text)
        if number in consumed or not match or int(match[2], 0) != MAGIC:
            continue
        if index < 2 or rows[index - 2][1] != "{" or not re.fullmatch(
                r"\.reg\s+\.b32\s+" + re.escape(match[1]) + r"\s*;", rows[index - 1][1]):
            raise DebugError(f"Source-hint magic outside a private block at PTX {number}")
        words, lines = [], []
        cursor = index
        while cursor < len(rows) and rows[cursor][1] != "}":
            word = MOV.fullmatch(rows[cursor][1])
            if not word or word[1] != match[1]:
                raise DebugError(f"Nonliteral or conflicting source-hint block at PTX {number}")
            value = int(word[2], 0)
            if value > 0xffffffff or len(words) > MAX_PAYLOAD // 4 + 6:
                raise DebugError("Oversized source-hint block")
            words.append(value)
            lines.append(rows[cursor][0])
            cursor += 1
        if cursor == len(rows) or len(words) < 6:
            raise DebugError(f"Truncated source-hint block at PTX {number}")
        if words[1] != HINT_ABI or words[2] not in (0, 1):
            raise DebugError(f"Unsupported source-hint ABI/kind at PTX {number}")
        length = words[3]
        if not 0 < length <= MAX_PAYLOAD or len(words) != 6 + (length + 3) // 4 or words[-1] != END_MAGIC:
            raise DebugError(f"Invalid source-hint length/footer at PTX {number}")
        padded = struct.pack(f"<{len(words) - 6}I", *words[4:-2])
        payload = padded[:length]
        if any(padded[length:]) or zlib.crc32(payload) != words[-2]:
            raise DebugError(f"Source-hint checksum mismatch at PTX {number}")
        try:
            identity = validate_identity(json.loads(payload))
        except (UnicodeError, ValueError, TypeError) as exc:
            raise DebugError(f"Invalid source-hint payload at PTX {number}: {exc}") from exc
        if canonical(identity) != payload:
            raise DebugError("Noncanonical source-hint identity")
        result.append(Hint(number, lines[-1], rows[index - 1][0], words[2], identity))
        consumed.update(lines)
    for number, text in rows:
        match = MOV.fullmatch(re.sub(r"^@!?[%\w$]+\s+", "", text))
        if match and int(match[2], 0) in (MAGIC, END_MAGIC) and number not in consumed:
            raise DebugError(f"Orphan source-hint framing word at PTX {number}")
    return result
