# Neutrino

by [Huang Songlin](https://huangs0.github.io) and [Wu Chenshu](https://cswu.me) from the University of Hong Kong.

Neutrino is a Probing-based GPU Kernel Profiler providing eBPF-like user experience for GPU Kernel Profiling, targeting:

1. **Fine-granularity**: Directly works on instructions to offer the finest granularity that can be mapped to particular hardware units.
2. **Programmability**: Extends the programmability of previous tools to probe cooperation with probe
3. **Versatility**: Supports both value profiling (register value like memory address) and value profiling (timestamp from device-side clock).
4. **Hardware-Independence**: Support both NVIDIA/CUDA and AMD/ROCm, more platforms to come!
5. **Ecosystem-Compatibility**: Built-in compatible with PyTorch (and everything on top like Huggingface), JAX, Triton, CUTLASS...

The foundations of this project are described in our OSDI '25 publication: [Neutrino: Fine-grained GPU Kernel Profiling via Programmable Probing](https://www.usenix.org/conference/osdi25/presentation/huang-songlin). Please consider citing this work if you use Neutrino!
The [official documentation](https://open-neutrino.github.io) contains more installation instructions, tutorials, internals and the DMAT galley!

## Latest News
* May 31, 2025: [Neutrino's artifact](https://github.com/open-neutrino/neutrino/tree/artifact) received all [badges](https://sysartifacts.github.io/osdi2024/badges) (Available, Functional, Reproduced) from OSDI 25 Artifact Evaluation!

## Quick Start

### Demos

Following demos are hosted on Colab with simple click `Runtime -> Run All`:

| Demo | Colab Link|
|---|---|
| Unrevealing block scheduling cost of `torch.zeros` | <a href="https://colab.research.google.com/drive/1rhIsfQnV1OwGIud8dSSFrod7AIvsA2cB?usp=sharing" target="_blank" rel="noopener noreferrer">  ![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg) </a> |
| Visualizing FlashAttn-v2 Memory Access | <a href="https://colab.research.google.com/drive/1RTcJfn32Ocx9B_icgpiDiC2WEHTKp1JB?usp=sharing" target="_blank" rel="noopener noreferrer">  ![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg) </a>|
| Warp Scheduling and Tailing Effect | <a href="https://colab.research.google.com/drive/1P50vzbkkgrbF-fVh1rcEe4YdHHUuq1tE?usp=sharing" target="_blank" rel="noopener noreferrer">  ![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg) </a> |

### Installation

Neutrino can be installed as a Python package from source. Building is fast (<30 seconds)!

```bash
# Virtual Environmnt is highly recommended!
conda create conda create -y -n <name> python=3.11 && conda activate <name>
git clone https://github.com/open-neutrino/neutrino
cd neutrino && python setup.py install && cd ..
neutrino --help # test installation
```

Neutrino does not have pre-build wheels, please **DO NOT** `pip instsall neutrino`!

## Using Neutrino

Inspired by [eBPF](https://ebpf.io/what-is-ebpf/), `probe` in Neutrino refers to a tiny sandboxed code snippet that could be attached to the GPU kernel at the assembly level (PTX, GCNAsm, SPIR-V) in the runtime. 
`probe` extends a new programmable interface than traditional programming and provides a convenient way for observability to black-boxed GPU runtime.
Currently Neutrino probes support two programming ways:
1. Pythonic Tracing DSL, suitable for beginners.
2. Direct Assembly probes wrapped in [TOML](https://toml.io/en/), suitable for advanced usage but it is platform-dependent.

<table style="width:50%">
<tr>
<td valign="top">

### Pythonic Tracing DSL
```python
from neutrino import probe, Map
import neutrino.language as nl
CALLBACK = "block_sched.py" # for trace analysis
# declare maps for persistence
@Map(level="warp", type="array", size=16, cap=1)
class block_sched:
  start: nl.u64
  elapsed: nl.u32
  cuid: nl.u32
# declare probe registers shared across probes
start: nl.u64 = 0 # starting clock
elapsed: nl.u64 = 0 # elapsed time, initialized to 0
# define probes with decorator
@probe(pos="kernel", level="warp", before=True)
def thread_start():
  start = nl.clock()
@probe(pos="kernel", level="warp")
def thread_end():
  elapsed = nl.clock() - start
  block_sched.save(start, elapsed, nl.cuid())
```

</td>
<td valign="top">

### Direct Assembly wrapped in TOML
```toml
# CUDA PTX Assembly Example
callback="block_sched.py"
[ map.block_sched ]
type = "array"
level = "warp"
size = "16"
cap = "1"
[ probe.thread_start_thread_end ]
position = "kernel"
level = "warp"
register = {"u32": 2, "u64": 3}
before = """.reg .b64 %PD<3>;
.reg .b32 %P<2>;
mov.u64 %PD0, %clock64;"""
after = """mov.u64 %PD1, %clock64;
sub.u64 %PD1, %PD1, %PD0;
cvt.u32.u64 %P1, %PD1;
mov.u32 %P2, %smid;
SAVE [ block_sched ] {%PD0, %P1, %P2};"""
```

</td>
</tr>
</table>

The interface of `@neutrino.Probe` is inspired by [Triton](https://triton-lang.org/main/index.html) whose contents (left) will be compiled, rather than executed, into platform-specific assemblies (right). 
Probes with the same `level`, `pos`, and optional `funcname` selector are merged.

The formulation (and the name) of `@neutrino.Map` is prompted by [eBPF Map](https://docs.ebpf.io/linux/concepts/maps/). With structured definition, Neutrino can have save (no illegal memory access) and efficient (race-free, no atomics) persistence. 

To simplify the development, Neutrino also provides some helper functions / operands:
* `nl.clock() / nl.time()`: for reading device-side clock and timer.
* `nl.addr/out/in1/in2/in3`: for reading register values
* `Map.save()`: for persisting values for posterior analysis.

### CUDA Inline-Function Probes (Experimental)

Use `pos="func"` and a qualified `funcname` to probe an inline function's entry and exit. Function probes share the existing DSL and maps; `before=True` selects entry, and the default selects exit:

```python
from neutrino import probe, Map
import neutrino.language as nl

@Map(level="warp", type="array", size=24, cap=64)
class function_events:
  timestamp: nl.u64
  site: nl.u64
  event: nl.u64

@probe(pos="func", funcname="sample::adjust", level="warp", before=True)
def enter():
  function_events.save(nl.time(), nl.site_id, nl.event)

@probe(pos="func", funcname="sample::adjust", level="warp")
def leave():
  function_events.save(nl.time(), nl.site_id, nl.event)
```

`nl.site_id` identifies the invocation site within a build; `nl.event` is 0 for entry and 1 for exit. Function names match exactly, including overloads and template specializations of a qualified base name. `-k` and `--filter` still select launch-kernel names.

Function boundaries require source markers embedded in the cubin. `neutrino-nvcc prepare` creates private marked headers from a versioned, source-hash-checked policy without modifying the originals. It accepts a JSON policy or a Python probe with `SOURCE_HINTS`; see the built-in [DeepEP barrier probe](neutrino/tools/deepep_barrier.py) for an example. Unmarked kernels are not instrumented; there is no CFG fallback.

For DeepEP/DeepJIT, set `CUDA_HOME` to a compatible toolkit and replace `/path/to/include` with the directory containing `deep_ep/comm/barrier.cuh`. Prepare before starting the application:

```bash
export NEUTRINO_NVCC_CONFIG="$(neutrino-nvcc prepare \
  --policy deepep_barrier --include-root /path/to/include \
  --real-nvcc "$CUDA_HOME/bin/nvcc" --artifact-root ./cache)"
export EP_JIT_NVCC_COMPILER="$(command -v neutrino-nvcc)"
export EP_JIT_CACHE_DIR="$(python -c 'import json, os; print(json.load(open(os.environ["NEUTRINO_NVCC_CONFIG"]))["jit_cache_dir"])')"
export EP_JIT_WITH_LINEINFO=1
neutrino -p deepep_barrier -k deep_ep --buffer-mb 128 --pool-prewarm 16 \
  --tracedir ./trace python train.py
python -m neutrino.utils.function_trace ./trace --output ./trace/decoded
```

Use the prepared JIT cache, not an existing unmarked cache. For other builds, invoke `neutrino-nvcc` with the application's usual flags and the prepared config. Custom probes run with `neutrino -p custom.py <command>`.

Collection uses private device buffers, CUDA completion events, and a separate nonblocking copy stream. Device buffers are reset before reuse; pinned-host buffers return to the pool after background export. Original launch streams and PDL overlap are preserved.

* `--buffer-mb` limits device-plus-pinned-host pool storage per process (default 128 MiB); `--pool-prewarm` sets slots per capacity class (default 8). Size both for the workload.
* Pool exhaustion skips that launch's trace and runs the original kernel exactly once. No waiting, replay, or pool growth occurs during steady-state acquisition.
* `neutrinoFunctionFlush()` drains accepted work for the current context; application CUDA synchronization does not flush host exports. `neutrinoFunctionGetStats()` and `result/pool-stats.json` report coverage and pool usage.
* The decoder writes `summary.json` and Chrome `timeline.json` for the timestamp/site/event schema. Check coverage, pairing, and saturation before using a trace; map capacity must cover all recorded events.
* Currently supported: CUDA inline functions, lane-0 warp probes, and finite warp array maps. Times include instrumentation and are GPU-local by default. Graph capture, packed `extra` launches, and unsupported boundaries produce coverage gaps; function and opcode probes cannot be mixed. GIN/RDMA and multi-node execution are unverified.

For approximate CPU alignment, the monitor's MoE workload accepts `--clock-align` with `--benchmark-output`: it samples `%globaltimer` against `CLOCK_MONOTONIC` once at startup and saves CPU step spans. Export with the same run's results (omit `--overview` for individual warp tracks):

```bash
python -m neutrino.utils.function_trace ./trace --training-results ./training-results.json --overview
```

The tightest sample's midpoint determines the offset; half its round trip is the startup uncertainty, without drift correction. Alignment requires matching rank/process/device/context identities on one host. Overview spans are cross-warp entry/exit envelopes, not kernel durations.

## Compatibility

More information can be found in our documentation. If you have more platforms or workloads need the support, please raise an issue to let us know!

<table style="width:50%">
<tr>
<td valign="top">

### Hardware


| Hardware Platform	| Support Status |
| --- | --- |
| NVIDIA/CUDA/PTX	| ✅ Supported | 
| AMD/ROCm/GCNAsm |	🛠️ Testing |
| General/OpenCL/SPIR-V	| 🚀 Planning |

</td>
<td valign="top">

### Software

| Software Framework | Status | 
| --- | --- |
| cuBLAS/cuFFT/cuSparse...	| ❌ (no plan for supporting) |
| CUTLASS	| ✅ (with macro in building) |
| PyTorch family (torchvision...) | ✅ (with custom build) |
| JAX	| ✅ (with envariable in runtime) | 
| Triton	| ✅ |

</td>
</tr>
</table>

## Internals

`neutrino` is designed to operate in the following workflow:

<img src="assets/workflow.png" alt="workflow" width="500"/>

The source code are placed in the following structure:

```
neutrino
├── language # DSL and Compiler, Still in Testing
│   ├── __init__.py # DSL Primitive
│   ├── compiler.py # Exported Compiler API
│   ├── frontend.py # Parser and AST Transformer
│   ├── gcn.py     # CUDA PTX Codegen Backend
│   └── ptx.py      # AMD ROCm Codegen Backend
├── probe    # Probe Engine
│   ├── __init__.py # Common Definition and Utilities
│   ├── cuda.py     # CUDA PTX Impl
│   ├── cuda_function.py # CUDA Inline-Function Probes
│   ├── function_hints.py # Source Marker Encoding and Selection
│   └── hip.py      # AMD ROCm Impl
├── src      # Hook Driver
│   ├── common.h    # Platform-agnostic Definition (GNU-only)
│   ├── cuda.c      # CUDA Impl (NVIDIA-related)
│   ├── cuda_function_probe.h # Pooled Asynchronous Function Traces
│   ├── hip.c       # ROCm Impl (AMD-related)
│   ├── preload.c   # Injector via LD_PRELOAD
│   ├── parse.py    # Generate Unhook API (NVIDIA/AMD)
│   ├── sha1.h      # third-parties header-only library
│   └── uthash.h    # third-parties header-only library
├── build.py    # Builder for driver in src/
├── cli.py      # Command Line Interface Entry
├── common.py   # Common Internal API not for User import
├── nvcc_wrapper.py # Immutable Source-Hint Build Wrapper
└── __init__.py # Common Defn for user import like probe, Map
```

The overall structure is clean and approachable, we welcome developers to hack the system for their need. Raise issues if you need help.

## More 

* How to write my probe? Check the [Probe Writing Guide](https://open-neutrino.github.io/docs/write-probes).
* How are probes executed? Check the [Probe Execution Model](https://open-neutrino.github.io/docs/execute-model).
* How to read the neutrino trace? Check the [Trace File Structure](https://open-neutrino.github.io/docs/read-trace).
* How to Neutrino works and how to extend? [Check the Reference and Internals](https://open-neutrino.github.io/docs/system-workflow).
* How good is Neutrino? Check the [Utilities and Extensions](https://open-neutrino.github.io/docs/analysis-code)

## Citation
If you used Neutrino in your research, please cite the paper below. And we welcome you to send us a link to your paper. 
```
@inproceedings{huang2025neutrino,
    author = {Songlin Huang and Chenshu Wu},
    title = {Neutrino: Fine-grained GPU Kernel Profiling via Programmable Probing},
    booktitle = {19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25)},
    year = {2025},
    url = {https://www.usenix.org/conference/osdi25/presentation/huang-songlin},
    publisher = {USENIX Association},
}
```
