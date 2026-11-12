"""Neutrino Internal APIs, not for user import"""

from typing import Optional, Literal, Union
from dataclasses import dataclass
import re

@dataclass
class Register:
    name: str
    dtype: Literal['u32', 'u64']
    init: Optional[int] = None


@dataclass
class Probe:
    name:   str                       # name is the key in TOML
    level:  Literal["thread", "warp"] # level of the probe
    pos:    list[str]                 # := tracepoint in the paper
    before: Union[list, str] = None   # snippet inserted before, one of before and after shall be given
    after:  Union[list, str] = None   # snippet inserted after,  one of before and after shall be given
    funcname: Optional[str] = None


@dataclass
class Map:
    name:   str 
    level:  Literal["thread", "warp"]
    type:   Literal["array"]
    size:   int
    cap:    Union[int, Literal["dynamic"]]
    regs:   list[Register]


def load(raw: dict) -> tuple[list[Probe], list[Map], int]:
    """Unserialize Neutrino probes in Python dict to probes, maps, regs"""
    assert "probe" in raw.keys() and "map" in raw.keys(), "At least a probe and a map"
    probes: list[Probe] = []
    maps: list[Map] = []
    for name, probe in raw["probe"].items():
        # first validate the 
        keys = probe.keys()
        assert "position" in keys or "pos" in keys, f"[error] {name} has no position (required)"
        # assert "datamodel" in keys, f"[error] "
        assert "before" in keys or "after" in keys, f"[error] {name} is empty, one of before or after shall be given"
        assert "level" in keys and probe["level"] in ("warp", "thread"), f"[error] level must be given and one of 'warp', 'thread'"
        position = probe.get("pos", probe.get("position"))
        if isinstance(position, str):
            position = position.split(":")
        if not isinstance(position, list) or not all(isinstance(p, str) for p in position):
            raise ValueError(f"Probe {name}: pos must be a string or list of strings")
        probes.append(Probe(name=name,
                            level=probe["level"], 
                            pos=position,
                            before=probe["before"] if "before" in keys else None,
                            after=probe["after"] if "after" in keys else None,
                            funcname=probe.get("funcname")))
    for name, map_ in raw["map"].items():
        maps.append(Map(name=name, 
                        level=map_["level"], 
                        type=map_["type"], 
                        size=map_["size"], 
                        cap=map_["cap"], 
                        regs=[Register(name, val[0], init=val[1]) for name, val in map_["regs"].items()]))
    function_probes = [p for p in probes if p.funcname is not None or "func" in p.pos]
    if function_probes:
        if len(function_probes) != len(probes):
            raise ValueError("Function probes cannot be mixed with kernel/opcode probes")
        for p in probes:
            if p.pos != ["func"] or not isinstance(p.funcname, str) or not p.funcname.strip():
                raise ValueError("Function probes require pos='func' and a nonempty funcname")
            if p.level != "warp":
                raise ValueError("Function probes currently support only level='warp'")
            if not any(isinstance(s, str) and s.strip() for s in (p.before, p.after)):
                raise ValueError(f"Function probe {p.name} has no payload")
            for snippet in (p.before, p.after):
                if snippet is not None and not isinstance(snippet, str):
                    raise ValueError("Function probe payloads must be compiled assembly strings")
                if re.search(r"\b(?:ADDR|BYTES|OUT|IN[1-4])\b", snippet or ""):
                    raise ValueError("Instruction operands are unavailable in function probes")
        if not maps:
            raise ValueError("Function probes require at least one finite warp map")
        for m in maps:
            if m.level != "warp" or type(m.cap) is not int or m.cap <= 0:
                raise ValueError("Function probes require finite positive warp-map capacities")
            if m.type != "array" or any(r.dtype not in ("u64", "u32") for r in m.regs):
                raise ValueError("Function maps require array storage with u64/u32 fields")
            if (type(m.size) is not int or m.size <= 0 or m.size % 8 or
                    m.size != sum(8 if r.dtype == "u64" else 4 for r in m.regs)):
                raise ValueError(f"Map {m.name}: size must equal the 8-byte-aligned payload layout")
            if sorted(m.regs, key=lambda r: r.dtype, reverse=True) != m.regs:
                raise ValueError(f"Map {m.name}: u64 fields must precede u32 fields")
        if type(raw.get("regs")) is not int or raw["regs"] < 0:
            raise ValueError("Function probe register count must be a nonnegative integer")
    elif any(re.search(r"\b(?:SITE_ID|EVENT_KIND)\b", str(p.before or "") + str(p.after or ""))
             for p in probes):
        raise ValueError("site_id and event operands require a function probe")
    return probes, maps, raw["regs"]


def dump(probes, maps, regs, callback = "") -> dict:
    """Serialize Neutrino probes to Python dict"""
    dict_probe = { 
        "regs": regs, 
        "probe" : {p.name: {"level": p.level, "pos": ":".join(p.pos) if isinstance(p.pos, list) else p.pos,
                                      "before": p.before, "after": p.after,
                                      **({"funcname": p.funcname} if p.funcname is not None else {})} for p in probes},
        "map": {m.name: {"level": m.level, "type": m.type, "size": m.size, "cap": m.cap, "regs": {r.name: [r.dtype, r.init] for r in m.regs}} for m in maps}
    }
    if len(callback) > 0:
        dict_probe["CALLBACK"] = callback
    return dict_probe