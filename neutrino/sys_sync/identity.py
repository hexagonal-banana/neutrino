"""Immutable identities shared by manifests and event correlation.

Artifact IDs describe original bytes; kernel IDs additionally identify a live
module instance. Launch IDs are never inferred from a symbol or timestamp.
These types do not load CUDA or instrument launches.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import re
import threading


class IdentityError(ValueError):
    pass


def _text(value, label):
    if not isinstance(value, str) or not value or '\x00' in value:
        raise IdentityError(f'Invalid {label}')


def _uint(value, label, positive=False):
    if type(value) is not int or not int(positive) <= value < 2**64:
        raise IdentityError(f'Invalid {label}')


def _id(kind, fields):
    payload = json.dumps([1, kind, fields], sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('ascii')).hexdigest()


@dataclass(frozen=True)
class ProcessIdentity:
    session: str
    pid: int
    instance: str
    rank: int | None = None

    def __post_init__(self):
        _text(self.session, 'session')
        _uint(self.pid, 'PID', positive=True)
        _text(self.instance, 'process instance')
        if self.rank is not None:
            _uint(self.rank, 'rank')

    @property
    def id(self):
        return _id('process', asdict(self))


@dataclass(frozen=True)
class ArtifactIdentity:
    original_sha256: str
    target: str
    image_format: str

    def __post_init__(self):
        if not isinstance(self.original_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', self.original_sha256):
            raise IdentityError('Expected original image SHA-256')
        if not isinstance(self.target, str) or not re.fullmatch(r'sm_\d+[af]?', self.target):
            raise IdentityError('Expected exact CUDA target; no architecture retargeting')
        if self.image_format not in {'ptx', 'cubin', 'fatbin', 'linked_group'}:
            raise IdentityError('Unsupported artifact format')

    @property
    def id(self):
        return _id('artifact', asdict(self))


@dataclass(frozen=True)
class ContextIdentity:
    process: ProcessIdentity
    generation: int
    gpu_uuid: str

    def __post_init__(self):
        if not isinstance(self.process, ProcessIdentity):
            raise IdentityError('Expected process identity')
        _uint(self.generation, 'context generation', positive=True)
        _text(self.gpu_uuid, 'GPU UUID')
        if not self.gpu_uuid.startswith('GPU-'):
            raise IdentityError('Expected physical GPU UUID')

    @property
    def id(self):
        return _id('context', asdict(self))


@dataclass(frozen=True)
class ModuleIdentity:
    context: ContextIdentity
    instance: int
    artifact: ArtifactIdentity

    def __post_init__(self):
        if not isinstance(self.context, ContextIdentity) or not isinstance(self.artifact, ArtifactIdentity):
            raise IdentityError('Expected context and artifact identities')
        _uint(self.instance, 'module instance', positive=True)

    @property
    def id(self):
        return _id('module', asdict(self))


@dataclass(frozen=True)
class KernelIdentity:
    module: ModuleIdentity
    entry: str

    def __post_init__(self):
        if not isinstance(self.module, ModuleIdentity):
            raise IdentityError('Expected module identity')
        _text(self.entry, 'entry symbol')

    @property
    def id(self):
        return _id('kernel', asdict(self))


@dataclass(frozen=True)
class LaunchIdentity:
    kernel: KernelIdentity
    ordinal: int
    stream: int

    def __post_init__(self):
        if not isinstance(self.kernel, KernelIdentity):
            raise IdentityError('Expected kernel identity')
        _uint(self.ordinal, 'launch ordinal', positive=True)
        _uint(self.stream, 'stream handle')

    @property
    def id(self):
        # The native collector's process-wide ordinal is the correlation key.
        return _id('launch', {'process': self.kernel.module.context.process.id, 'ordinal': self.ordinal})


class Registry:
    """CPU metadata registry; explicit retirement prevents handle-reuse aliasing.

    Entries remain in history after retirement for asynchronous decoding. This
    registry does not allocate buffers or imply that submitted work completed.
    """
    def __init__(self, process):
        if not isinstance(process, ProcessIdentity):
            raise IdentityError('Expected process identity')
        self.process = process
        self._lock = threading.RLock()
        self._contexts = {}
        self._modules = {}
        self._context_sequence = self._module_sequence = self._launch_sequence = 0
        self.context_history = {}
        self.module_history = {}
        self.kernel_history = {}
        self.launch_history = {}

    def context_created(self, handle, gpu_uuid):
        _uint(handle, 'context handle', positive=True)
        with self._lock:
            if handle in self._contexts:
                raise IdentityError('Context handle is already live')
            generation = self._context_sequence + 1
            context = ContextIdentity(self.process, generation, gpu_uuid)
            self._context_sequence = generation
            self._contexts[handle] = context
            self.context_history[context.id] = context
            return context

    def context_retired(self, handle):
        _uint(handle, 'context handle', positive=True)
        with self._lock:
            context = self._contexts.pop(handle, None)
            if context is None:
                raise IdentityError('Unknown/retired context')
            self._modules = {key: value for key, value in self._modules.items() if key[0] != context.id}
            return context

    def module_loaded(self, context_handle, module_handle, artifact):
        _uint(context_handle, 'context handle', positive=True)
        _uint(module_handle, 'module handle', positive=True)
        with self._lock:
            context = self._contexts.get(context_handle)
            if context is None:
                raise IdentityError('Unknown/retired context')
            key = (context.id, module_handle)
            if key in self._modules:
                raise IdentityError('Module handle is already live')
            sequence = self._module_sequence + 1
            module = ModuleIdentity(context, sequence, artifact)
            self._module_sequence = sequence
            self._modules[key] = module
            self.module_history[module.id] = module
            return module

    def module_retired(self, context_handle, module_handle):
        _uint(context_handle, 'context handle', positive=True)
        _uint(module_handle, 'module handle', positive=True)
        with self._lock:
            context = self._contexts.get(context_handle)
            if context is None or self._modules.pop((context.id, module_handle), None) is None:
                raise IdentityError('Unknown/retired module')

    def kernel(self, context_handle, module_handle, entry):
        _uint(context_handle, 'context handle', positive=True)
        _uint(module_handle, 'module handle', positive=True)
        with self._lock:
            context = self._contexts.get(context_handle)
            module = self._modules.get((context.id, module_handle)) if context else None
            if module is None:
                raise IdentityError('Unknown/retired module')
            kernel = KernelIdentity(module, entry)
            return self.kernel_history.setdefault(kernel.id, kernel)

    def launch(self, kernel, stream):
        with self._lock:
            if not isinstance(kernel, KernelIdentity) or kernel.id not in self.kernel_history:
                raise IdentityError('Unknown kernel')
            if kernel.module not in self._modules.values():
                raise IdentityError('Kernel module is retired')
            ordinal = self._launch_sequence + 1
            launch = LaunchIdentity(kernel, ordinal, stream)
            self._launch_sequence = ordinal
            self.launch_history[launch.id] = launch
            return launch
