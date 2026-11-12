"""Automatic Code Generation based on the Map definitions for reading trace files.
TODO update to numpy.struct"""

from neutrino.common import load

__all__ = ["gen_reading_code"]

# NOTE Template for Generating Trace Reading Code
TRACE_READING_PY = """# Neutrino Auto-Generated Code for Trace Reading
import struct
from typing import NamedTuple
from neutrino import TraceHeader, TraceSection

{MAP_DEFNS}

def parse(path: str) -> tuple[TraceHeader, list[TraceSection], dict[str, list[list[NamedTuple]]]]:
    with open(path, "rb") as f:
        header: TraceHeader = TraceHeader(*struct.unpack("iiiiiiii", f.read(32)))
        sections: list[TraceSection] = []
        for _ in range(header.numProbes):
            sections.append(TraceSection(*struct.unpack("IIQ", f.read(16))))
        gridSize = header.gridDimX * header.gridDimY * header.gridDimZ
        blockSize = header.blockDimX * header.blockDimY * header.blockDimZ
        records: dict[str, list[list[NamedTuple]]] = dict()
{TRACE_READINGS}
    return header, sections, records
# END of Neutrino Auto-Generated Code for Trace Reading"""

TRACE_STRUCT_CODE_PY = """
class {MAP_NAME}(NamedTuple):
{CONTENT}
"""

TRACE_PARSING_PY = """
        # Read {MAP_NAME}
        records["{MAP_NAME}"] = []
        f.seek(sections[{INDEX}].offset)
        for i in range(gridSize):
            records["{MAP_NAME}"].append([])
            for j in range((blockSize + sections[{INDEX}].warpDiv - 1) // sections[{INDEX}].warpDiv):
                records["{MAP_NAME}"][-1].append([])
                for k in range(sections[{INDEX}].size // {BYTES}):
                    records["{MAP_NAME}"][i][j].append({MAP_NAME}(*struct.unpack("{FORMAT_STRING}", f.read({BYTES}))))
"""

def gen_reading_code(probe: dict) -> str:
    """Generate the code for reading the trace file"""
    _, maps, _ = load(probe)
    trace_structs = []
    trace_readings = []

    for index, map in enumerate(maps):
        content = []
        format_string = ""
        reading_bytes = 0
        for reg in map.regs:
            content.append(f"    {reg.name}: int")
            format_string += "Q" if reg.dtype == 'u64' else "I"
            reading_bytes += 8 if reg.dtype == 'u64' else 4
        trace_structs.append(TRACE_STRUCT_CODE_PY.format(
            MAP_NAME=map.name,
            CONTENT="\n".join(content)
        ))
        trace_readings.append(TRACE_PARSING_PY.format(
            MAP_NAME=map.name,
            FORMAT_STRING=format_string,
            BYTES=reading_bytes,
            INDEX=index,
        ))

    return TRACE_READING_PY.format(
        MAP_DEFNS="\n".join(trace_structs),
        TRACE_READINGS="\n".join(trace_readings)
    )

if __name__ == "__main__":
    # Example usage
    import sys
    import toml
    probe = toml.load(sys.argv[1])
    print(gen_reading_code(probe))