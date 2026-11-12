"""Generate the CUDA PTX Assembly, a C-style asm"""

from neutrino.common import Register, Probe, Map

def filter_keyword(reg: str) -> str:
    if reg in {"ADDR", "BYTES", "OUT", "IN1", "IN2", "IN3", "IN4", "SITE_ID", "EVENT_KIND"}:
        return reg
    elif isinstance(reg, int):
        return str(reg)
    else:
        return "%" + reg

def cvt_inst(inst: list[str]) -> str:
    match inst[0]:
        # ALU Instructions
        case "add":
            return f"add.u64 {filter_keyword(inst[1])}, {filter_keyword(inst[2])}, {filter_keyword(inst[3])};"
        case "sub":
            return f"sub.u64 {filter_keyword(inst[1])}, {filter_keyword(inst[2])}, {filter_keyword(inst[3])};"
        case "mul":
            return f"mul.u64 {filter_keyword(inst[1])}, {filter_keyword(inst[2])}, {filter_keyword(inst[3])};"
        case "div":
            return f"div.u64 {filter_keyword(inst[1])}, {filter_keyword(inst[2])}, {filter_keyword(inst[3])};"
        case "mod":
            return f"rem.u64 {filter_keyword(inst[1])}, {filter_keyword(inst[2])}, {filter_keyword(inst[3])};"
        case "lsh":
            return f"shl.u64 {filter_keyword(inst[1])}, {filter_keyword(inst[2])}, {filter_keyword(inst[3])};"
        case "rsh":
            return f"shr.u64 {filter_keyword(inst[1])}, {filter_keyword(inst[2])}, {filter_keyword(inst[3])};"
        # Memory Instructions
        case "SAVE":
            contents = (filter_keyword(reg) for reg in inst[2:])
            contents = ", ".join(contents)
            return f"SAVE [ {inst[1]} ] {{ { contents } }}" # just return everything
        # Other Instructions
        case "mov":
            return f"mov.u64 {filter_keyword(inst[1])}, {filter_keyword(inst[2])};"
        case "clock":
            return f"mov.u64 {filter_keyword(inst[1])}, %clock64;"
        case "time":
            return f"mov.u64 {filter_keyword(inst[1])}, %globaltimer;"
        case "cuid":
            return f"""{{
                .reg .b32 %tmp;
                mov.u32 %tmp, %smid;
                cvt.u64.u32 {filter_keyword(inst[1])}, %tmp;
            }}"""
        case _:
            raise NotImplementedError(f"{inst} not yet supported")

def gencode(probes: list[Probe]) -> list[Probe]:
    # First handle the initialization of regs

    # Then handle the syntax of probes
    for probe in probes:
        # only change the instructions, i.e., before and after part
        if probe.before is not None:
            insts: list[str] = []
            for inst in probe.before:
                insts.append(cvt_inst(inst))
            probe.before = "\n".join(insts)
        elif probe.after is not None:
            insts: list[str] = []
            for inst in probe.after:
                insts.append(cvt_inst(inst))
            probe.after = "\n".join(insts)
    
    return probes

if __name__ == "__main__":
    probes = [
        Probe(name='thread_start', level='warp', pos='kernel', size=0, before=None, after=[['clock', 'R0']]), 
        Probe(name='thread_end', level='warp', pos='kernel', size=0, before=None, after=[['clock', 'R2'], ['sub', 'R1', 'R2', 'R0'], ['cuid', 'R3'], ['SAVE', 'block_sched', 'R0', 'R1', 'R3']])
    ]
    print(gencode(probes))
