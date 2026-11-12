from typing import NamedTuple, Union, Literal

# Load the inert engine package before defining the public decorator. Otherwise
# a later engine import replaces neutrino.probe with the package object.
from . import probe as _probe_engine

class TraceHeader(NamedTuple):
    gridDimX: int
    gridDimY: int
    gridDimZ: int
    blockDimX: int
    blockDimY: int
    blockDimZ: int
    sharedMemBytes: int
    numProbes: int

class TraceSection(NamedTuple):
    size:    int
    warpDiv: int
    offset:  int

def probe(pos: str, after: bool = False, level: str = "thread", size: int = 0,
          *, before: bool = False, funcname: str | None = None):
    """Neutrino Probe Entry"""
    from functools import wraps
    # Just preventing the execution as we take it as part of AST only
    def inner(func: callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
          raise RuntimeError(f"{func.__name__} shall be jit other than run")
        return wrapper
    return inner

def Map(level: Literal["warp", "thread"], type: str, size: int, cap: Union[int, Literal["dynamic"]]):
    """Neutrino Map Definition"""
    from functools import wraps
    def inner(cls): 
        @wraps(cls)
        def wrapper(*args, **kwargs):
          raise RuntimeError(f"{cls.__name__} shall be jit other than run")
        return wrapper
    return inner

# Following are internal definition