"""Fail-closed telemetry checks; idle observations are not GPU allocation."""
import csv
import fcntl
import io
from pathlib import Path
import subprocess
import time
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[3]


class UnsafeGPUState(RuntimeError):
    pass


def query(options):
    try:
        return subprocess.check_output(['nvidia-smi', *options], text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise UnsafeGPUState(f'GPU telemetry unavailable: {exc}') from exc


def check_once(runner=query):
    text = runner(['--query-gpu=index,uuid,name,memory.used,utilization.gpu',
                   '--format=csv,noheader,nounits'])
    devices = []
    try:
        for row in csv.reader(io.StringIO(text)):
            if len(row) != 5:
                raise ValueError('Incomplete GPU telemetry')
            index, uuid, name, memory, utilization = (value.strip() for value in row)
            item = dict(index=int(index), uuid=uuid, name=name,
                        memory_mib=int(memory), utilization=int(utilization))
            if (not uuid.startswith('GPU-') or 'H800' not in name or
                    not 0 <= item['memory_mib'] <= 1024 or item['utilization'] != 0):
                raise ValueError(f'GPU is busy or unsupported: {item}')
            devices.append(item)
        if (len(devices) != 8 or {d['index'] for d in devices} != set(range(8)) or
                len({d['uuid'] for d in devices}) != 8):
            raise ValueError('Expected eight distinct H800 GPUs')
    except (ValueError, TypeError) as exc:
        raise UnsafeGPUState(str(exc)) from exc
    processes = runner(['--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    if processes.strip():
        raise UnsafeGPUState('GPU processes or unrecognized process telemetry: ' + processes.strip())
    return sorted(devices, key=lambda item: item['index'])


def require_idle(runner=query, sleep=time.sleep):
    identity = None
    for sample in range(6):
        devices = check_once(runner)
        current = [(d['index'], d['uuid']) for d in devices]
        if identity is not None and identity != current:
            raise UnsafeGPUState('GPU identities changed while checking idle state')
        identity = current
        if sample < 5:
            sleep(5)
    return devices


@contextmanager
def launch_lock():
    path = ROOT / '.cache/unified/gpu-launch.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise UnsafeGPUState('Another unified workload is running') from exc
        yield handle
