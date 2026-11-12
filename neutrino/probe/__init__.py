"""Neutrino Probing Engine Protocol

NOTE just a protocol for developers, don't import / export"""

from dataclasses import dataclass
from neutrino.common import Register, Probe, Map

__all__ = ["Ref", "load_probes", "TRACE_READING_CODE_PY"]

@dataclass
class Ref:
    """Reference for replacement"""
    line: str          # Original line
    probe: str         # Probe name for matchine
    before_after: bool # True if before and False if after -> to distinguish which snippet is used
    site_id: int | None = None
    event: int | None = None


@dataclass
class KernelParam:
    dtype: str
    name: str


# NOTE following is just protocol, please implement yours, developers can 
# extend other functions for their need, just keep following implemented

def get_arch() -> str:
    """get architecture for assembler"""
    ...

def dump(workdir: str, name: str, suffix: str) -> str:
    """call objdump to extract assembly from binary"""
    ...

def prune(ptx: str, entry_name: str):
    """Prune Assembly to specific entry_name"""
    ...

def probing(asm: str, probes: list[Probe]):
    """Probe the probes into asm"""
    ...

def assemble(workdir: str, name: str):
    """call assembler to turn assembly to machine code"""
    ...

def write_kernel_info(name: str, params, probe_mem_sizes: list[int], 
    workdir: str, analyze_hook: str = "", file_name: str = "kernel.info"): 
    """write kernel info for hook driver to read back"""
    ...