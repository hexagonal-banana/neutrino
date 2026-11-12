"""CLI Entry of Language Submodule"""
from neutrino.common import Probe, dump, load
from neutrino.language.frontend import parse

def compile(mode: str, source: str) -> dict:
    """Compile the Tracing DSL into Assembly Probes"""
    registers = []
    regs, probes, maps, callback = parse(source, register_descriptions=registers)
    function_mode = any(p.funcname is not None or p.pos == "func" for p in probes)
    if function_mode and mode != "CUDA":
        raise ValueError("Function probes currently require CUDA")
    if mode == "CUDA":
        from neutrino.language.ptx import gencode
        probes = gencode(probes)
    elif mode == "HIP":
        from neutrino.language.gcn import gencode
        probes = gencode(probes)

    # NOTE Merge probes of the same level and pos
    merged_probes: dict[tuple[str, str, str | None], Probe] = {}
    for probe in probes:
        key = (probe.level, probe.pos, probe.funcname)
        if key not in merged_probes:
            merged_probes[key] = probe
        else: # merge
            merged_probes[key].name += "_" + probe.name
            merged_probes[key].before = (
                "\n".join(s for s in (merged_probes[key].before, probe.before) if s)
                if merged_probes[key].before or probe.before
                else None
            )
            merged_probes[key].after = (
                "\n".join(s for s in (merged_probes[key].after, probe.after) if s)
                if merged_probes[key].after or probe.after
                else None
            )
    probes = list(merged_probes.values())

    dumped = dump(probes, maps, regs, callback)

    for map_ in maps:
        if map_.cap == "dynamic":
            dumped["dynamic"] = True

    load(dumped)
    if function_mode:
        dumped["registers"] = [
            {"name": r.name, "dtype": r.dtype, "init": r.init, "index": i}
            for i, r in enumerate(registers)
        ]
    return dumped

if __name__ == "__main__": # A small test case
    import sys
    import toml
    mode, source = sys.argv[1], sys.argv[2]
    source = open(source, "r").read()
    asm_probes = compile(mode, source)
    print(toml.dumps(asm_probes))