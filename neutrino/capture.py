"""Library-neutral capture control for the opt-in native system-scope tracer.

This API fails closed when the native collector is absent or incompatible. It
never launches a kernel, starts a second profiler, or emulates a successful trace.
"""
from contextlib import contextmanager
import ctypes
import os
import threading

_lock = threading.Lock()


def require_runtime():
    if not os.environ.get('NEUTRINO_SESSION_ID'):
        raise RuntimeError('System-scope capture requires a shared Neutrino session ID')
    process = ctypes.CDLL(None)
    try:
        version = process.neutrinoSysAbiVersion
        begin = process.neutrinoSysCaptureBegin
        end = process.neutrinoSysCaptureEnd
    except AttributeError as exc:
        raise RuntimeError('The native sys_sync collector is not loaded') from exc
    version.argtypes, version.restype = [], ctypes.c_uint32
    begin.argtypes, begin.restype = [ctypes.c_char_p, ctypes.c_int64], ctypes.c_int
    end.argtypes, end.restype = [ctypes.c_int], ctypes.c_int
    if version() != 1:
        raise RuntimeError('Incompatible native sys_sync capture ABI')
    return begin, end


def finalize():
    """Release recording allocations before CUDA context teardown."""
    require_runtime()
    process = ctypes.CDLL(None)
    try:
        finish = process.neutrinoSysFinalize
    except AttributeError as exc:
        raise RuntimeError('The native collector has no finalization entry point') from exc
    finish.argtypes, finish.restype = [], ctypes.c_int
    with _lock:
        if finish():
            raise RuntimeError('Native collector finalization failed')


@contextmanager
def capture_range(name, ordinal):
    if not isinstance(name, str) or not name or '\0' in name:
        raise ValueError('Capture name must be nonempty and contain no NUL')
    if type(ordinal) is not int or not 0 <= ordinal < 2**63:
        raise ValueError('Capture ordinal must be a nonnegative signed 64-bit integer')
    if not _lock.acquire(blocking=False):
        raise RuntimeError('Nested or concurrent process-wide capture ranges are unsupported')
    try:
        begin, end = require_runtime()
        status = begin(name.encode('utf-8'), ordinal)
        if status:
            raise RuntimeError(f'Native capture begin failed: {status}')
        try:
            yield
        except BaseException:
            end(0)
            raise
        else:
            status = end(1)
            if status:
                raise RuntimeError(f'Native capture end/drain failed: {status}')
    finally:
        _lock.release()
