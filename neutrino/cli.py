"""CLI Entry for NEUTRINO: Fine-grained GPU Kernel Profiling via Programmable Probing"""

import subprocess
import os
import sys
import toml
import argparse
import uuid


# Main Function, need a func to be referred by setup.py build system
def main(): 
    # NOTE READ CONFIG.TOML FOR DEFAULT SYSTEM CONFIGURATION
    # directory of this python file and other toolkits
    CURDIR = os.path.dirname(os.path.realpath(__file__))
    # directory of the neutrino internals
    NEUTRINO_BUILD_DIR: str = os.path.join(CURDIR, "build")
    NEUTRINO_PROBE_DIR: str = os.path.join(CURDIR, "probe")
    NEUTRINO_TOOLS_DIR: str = os.path.join(CURDIR, "tools")
    # load system configuration, generated in building
    config = toml.load(os.path.join(NEUTRINO_BUILD_DIR, "config.toml"))["system"] 
    # default configurations, can be overwritten by CLI parameters
    NEUTRINO_HOOK_DRIVER_NAME: str = config["NEUTRINO_HOOK_DRIVER_LIB_NAME"]
    NEUTRINO_REAL_DRIVER_DIR : str = config["NEUTRINO_REAL_DRIVER_LIB_DIR"]
    NEUTRINO_REAL_DRIVER_NAME: str = config["NEUTRINO_REAL_DRIVER_LIB_NAME"]
    NEUTRINO_MODE            : str = config["NEUTRINO_MODE"]
    # available built-in tools
    NEUTRINO_TOOLS = {tool[:-3] : tool for tool in os.listdir(NEUTRINO_TOOLS_DIR) if tool.endswith(".py")}
    NEUTRINO_TOOLS['deepep_barrier'] = None
    NEUTRINO_TOOLS['ptx_sync'] = None
    
    parser = argparse.ArgumentParser(
        prog='neutrino', usage='%(prog)s [options] command',
        description=f"""NOTE: Probes must be given via -p (--probe) option. Buit-in tools: {tuple(NEUTRINO_TOOLS.keys())}""", 
        epilog="Examples: `neutrino -t gmem_bytes python test/zero_.py`. Open issue(s) in https://github.com/neutrino-gpu/neutrino if encountered problems", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-p', '--probe', required=True, 
                        help="probe in form of path to the file")
    parser.add_argument('--tracedir', default="./trace", 
                        help="parent folder of traces")
    parser.add_argument('--driver', default=os.path.join(NEUTRINO_REAL_DRIVER_DIR, NEUTRINO_REAL_DRIVER_NAME),
                        help='path to the real cuda/hip driver shared library')
    parser.add_argument("--python", default=sys.executable, 
                        help='path to python executable used')
    parser.add_argument('--filter', 
                        help='filter OUT buggy kernels by (part of) name, split by :')
    parser.add_argument('-k', '--kernel', 
                        help='filter the kernel by (part of) name, split by :')
    parser.add_argument('--callback', 
                        help='attach callback for trace analysis')
    parser.add_argument('--benchmark', action='store_true', 
                        help="enable benchmark mode to evaluate overhead w.r.t. the original kernel")
    parser.add_argument('--memusage', action='store_true', 
                        help="prevent the profiling and only measure the memory usage")
    parser.add_argument('--barrier-kernel-only', choices=('baseline', 'probed'),
                        help='Controlled single-stream kernel timing experiment; disables ordinary trace collection')
    parser.add_argument('--barrier-clock-align', choices=('host', 'none'),
                        help='Host clock calibration; defaults to host for ordinary barrier traces and none for kernel-only timing')
    parser.add_argument('--sync-threads', choices=('warp', 'all', 'block'),
                        help='Record lane 0 per warp, every thread, or thread 0 per block (aware only); defaults: ptx_sync=warp, deepep_barrier=all')
    parser.add_argument('--sync-local-barriers', choices=('exclude', 'include'), default='exclude',
                        help='Exclude bar.sync, barrier.sync, and bar.warp.sync probes; preserve original instructions')
    parser.add_argument('--sync-scope', choices=('all', 'sys'), default='all',
                        help='ptx_sync scope filter; sys excludes GPU/grid/CTA and implicit-scope probes')
    parser.add_argument('--sync-capacity', type=int, default=256,
                        help='ptx_sync intervals per recording unit per launch; overflow invalidates the recording')
    parser.add_argument('--sync-waits', choices=('loop', 'instruction', 'episode'), default='loop',
                        help='Group pure polling loops, bracket instructions, or include cooperative sys polling episodes')
    parser.add_argument('--sync-buffer-mb', type=int, default=1024,
                        help='ptx_sync maximum retained trace storage per process, on both host and GPU')
    parser.add_argument('--sync-register-level', type=int, default=10,
                        help='PTXAS register usage level; no-probe SASS equivalence is required')
    # put command at the end of command
    parser.add_argument("command", nargs=argparse.REMAINDER) 
    # parse the arguments
    args = parser.parse_args()
    
    # same as this executable
    NEUTRINO_PYTHON: str = args.python # default to be this executable
    # directory to put the trace
    NEUTRINO_TRACEDIR: str = args.tracedir
    # filter of kernel
    NEUTRINO_FILTER: str = args.filter if args.filter is not None else ""
    NEUTRINO_KERNEL: str = args.kernel if args.kernel is not None else ""
    # Benchmark mode, will include an additional launch after the trace kernel
    # Used to measure the kernel-level slowdown of Neutrino, disabled by default
    NEUTRINO_BENCHMARK: str = str(int(args.benchmark))
    NEUTRINO_MEMUSAGE: str = str(int(args.memusage))
    # Path to the real driver
    NEUTRINO_REAL_DRIVER: str = args.driver
    # command to be executed
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        parser.error('Command must be specified')
    sync_mode = args.probe == 'ptx_sync'
    if args.sync_scope != 'all' and not sync_mode:
        parser.error('--sync-scope requires ptx_sync')
    if args.sync_waits == 'episode' and (not sync_mode or args.sync_scope != 'sys' or args.sync_local_barriers != 'exclude'):
        parser.error('--sync-waits episode requires ptx_sync, --sync-scope sys, and excluded local barriers')
    args.sync_threads = args.sync_threads or ('warp' if sync_mode else 'all')
    if args.sync_threads == 'block' and args.probe != 'deepep_barrier':
        parser.error('--sync-threads block requires deepep_barrier')
    barrier_mode = args.probe in ('deepep_barrier', 'ptx_sync')
    if not 1 <= args.sync_capacity <= 65536 or not 1 <= args.sync_buffer_mb <= 8192 or not 0 <= args.sync_register_level <= 10:
        parser.error('Invalid synchronization capacity, buffer limit, or register level')
    if args.barrier_kernel_only and not barrier_mode:
        parser.error('--barrier-kernel-only requires deepep_barrier or ptx_sync')
    if args.barrier_clock_align and not barrier_mode:
        parser.error('--barrier-clock-align requires deepep_barrier or ptx_sync')
    if args.barrier_clock_align == 'host' and args.barrier_kernel_only:
        parser.error('Host clock alignment is disabled for kernel-only timing')
    clock_alignment = args.barrier_clock_align or ('host' if barrier_mode and not args.barrier_kernel_only else 'none')
    if barrier_mode and (args.benchmark or args.memusage or args.filter or args.kernel or args.callback or NEUTRINO_MODE != 'CUDA'):
        parser.error('deepep_barrier requires CUDA and forbids replay, filters, and callbacks')
    
    # Parse the PROBE
    NEUTRINO_PROBE_PATH: str = args.probe
    NEUTRINO_READING: str = None
    # NOTE endswith .py triggers the Tracing DSL
    if barrier_mode:
        NEUTRINO_PROBE = {'mode': args.probe, 'dynamic': False}
    elif NEUTRINO_PROBE_PATH.endswith(".py"):
        from neutrino.language.compiler import compile
        source = open(NEUTRINO_PROBE_PATH, "r").read()
        NEUTRINO_PROBE = compile(NEUTRINO_MODE, source)
    elif NEUTRINO_PROBE_PATH.endswith(".toml"):
        NEUTRINO_PROBE = toml.load(NEUTRINO_PROBE_PATH)
    else:
        # No suffix := use built-in tools
        if  NEUTRINO_PROBE_PATH not in NEUTRINO_TOOLS:
            print(f"[error] {NEUTRINO_PROBE_PATH} not in tools: {NEUTRINO_TOOLS}", file=sys.stderr)
            exit(-1)
        else:
            from neutrino.language.compiler import compile
            source = open(os.path.join(NEUTRINO_TOOLS_DIR, NEUTRINO_TOOLS[NEUTRINO_PROBE_PATH]), "r").read()
            NEUTRINO_PROBE = compile(NEUTRINO_MODE, source)
            
    # NOTE generate the trace reading code
    from neutrino.utils.trace_reading import gen_reading_code
    NEUTRINO_READING = None if barrier_mode else gen_reading_code(NEUTRINO_PROBE)
    
    # NOTE check if dynamic is True, shall have a specific keyword in top-level of probe
    NEUTRINO_DYNAMIC = "dynamic" in NEUTRINO_PROBE and NEUTRINO_PROBE["dynamic"] is True
    
    # TODO change the callback to other places
    NEUTRINO_CALLBACK = NEUTRINO_PROBE["CALLBACK"] if "CALLBACK" in NEUTRINO_PROBE else None
    NEUTRINO_CALLBACK = args.callback if args.callback is not None else NEUTRINO_CALLBACK
    if NEUTRINO_CALLBACK:
        # search the path 
        if not os.path.exists(NEUTRINO_CALLBACK):
            searched = os.path.join(os.path.dirname(NEUTRINO_PROBE_PATH), NEUTRINO_CALLBACK)
            if os.path.exists(searched):
                NEUTRINO_CALLBACK = searched
            else:
                print(f"[warn] callback {NEUTRINO_CALLBACK} not found")
                NEUTRINO_CALLBACK = None

    # a copied environment variables
    env = os.environ.copy()
    env['NEUTRINO_BARRIER_MODE'] = str(int(barrier_mode))
    env['NEUTRINO_PTX_SYNC'] = str(int(sync_mode))
    env['NEUTRINO_SYNC_THREADS'] = args.sync_threads
    env['NEUTRINO_SYNC_LOCAL_BARRIERS'] = args.sync_local_barriers
    env['NEUTRINO_SYNC_SCOPE'] = args.sync_scope
    env['NEUTRINO_SYNC_CAPACITY'] = str(args.sync_capacity)
    env['NEUTRINO_SYNC_WAITS'] = args.sync_waits
    env['NEUTRINO_SYNC_BUFFER_MB'] = str(args.sync_buffer_mb)
    env['NEUTRINO_SYNC_REGISTER_LEVEL'] = str(args.sync_register_level)
    env['NEUTRINO_BARRIER_CLOCK_ALIGN'] = clock_alignment
    env['NEUTRINO_TRACE_SESSION'] = str(uuid.uuid4())
    env.pop('NEUTRINO_BARRIER_KERNEL_ONLY', None)
    if args.barrier_kernel_only:
        env['NEUTRINO_BARRIER_KERNEL_ONLY'] = args.barrier_kernel_only
    if barrier_mode:
        env['NEUTRINO_DYNAMIC'] = '0'
    if barrier_mode and not sync_mode:
        env['EP_JIT_WITH_LINEINFO'] = '1'
        env['EP_JIT_DUMP_PTX'] = '1'
        if not env.get('EP_JIT_CACHE_DIR'):
            parser.error('Set a separate EP_JIT_CACHE_DIR before running deepep_barrier')
    NEUTRINO_TRACEDIR = os.path.abspath(NEUTRINO_TRACEDIR)
    os.makedirs(NEUTRINO_TRACEDIR, exist_ok=True)
    # configure Neutrino related environment variables
    env["NEUTRINO_REAL_DRIVER"]  = NEUTRINO_REAL_DRIVER
    env["NEUTRINO_DRIVER_NAME"]  = NEUTRINO_HOOK_DRIVER_NAME
    env["NEUTRINO_HOOK_DRIVER"]  = os.path.join(NEUTRINO_BUILD_DIR, NEUTRINO_HOOK_DRIVER_NAME)
    env["NEUTRINO_PYTHON"]       = NEUTRINO_PYTHON
    env["NEUTRINO_PROBING_PY"]   = os.path.join(NEUTRINO_BUILD_DIR, "process.py")
    env["NEUTRINO_FILTER"]       = NEUTRINO_FILTER
    env["NEUTRINO_KERNEL"]       = NEUTRINO_KERNEL
    env["NEUTRINO_TRACEDIR"]     = NEUTRINO_TRACEDIR
    env["NEUTRINO_PROBES"]       = toml.dumps(NEUTRINO_PROBE) # dump it to string
    # GNU LD_PRELOAD to overwrite dlopen, https://man7.org/linux/man-pages/man8/ld.so.8.html
    env["LD_PRELOAD"]            = os.path.join(NEUTRINO_BUILD_DIR, "preload.so")
    # Add to the LD_LIBRARY_PATH, this would overwrite ldconfig
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"]   = NEUTRINO_BUILD_DIR + ":" + env["LD_LIBRARY_PATH"]
    else:
        env["LD_LIBRARY_PATH"]   = NEUTRINO_BUILD_DIR
    # An Environmental Variable to enable the trace
    # NOTE some bugs here -> still working on
    env["NEUTRINO_ENABLE"] = "1"
    # An Environmental Variable to enable the benchmark mode
    env["NEUTRINO_BENCHMARK"]    = NEUTRINO_BENCHMARK
    env["NEUTRINO_MEMUSAGE"]     = NEUTRINO_MEMUSAGE
    # An Environmental Variables to enable the debug mode -> more messages
    # env["NEUTRINO_VERBOSE"] = "1"
    if NEUTRINO_DYNAMIC:
        env["NEUTRINO_DYNAMIC"]  = "1"
    if NEUTRINO_READING:
        env["NEUTRINO_READING"]  = NEUTRINO_READING
    if NEUTRINO_CALLBACK:
        env["NEUTRINO_CALLBACK"] = NEUTRINO_CALLBACK

    # FIX for Triton
    if NEUTRINO_MODE == "CUDA":
        env["TRITON_LIBCUDA_PATH"] = NEUTRINO_BUILD_DIR
        env["NEUTRINO_PROBING_PY"] = os.path.join(NEUTRINO_PROBE_DIR, "cuda.py")
    elif NEUTRINO_MODE == "HIP":
        # NOTE There's a bug in Triton's impl here, for path we refer to the 
        # directory for ld.so to search, instead of spcific file name ...
        env["TRITON_LIBHIP_PATH"]  = os.path.join(NEUTRINO_BUILD_DIR, "libamdhip64.so") 
        env["NEUTRINO_PROBING_PY"] = os.path.join(NEUTRINO_PROBE_DIR, "hip.py")

    # start the program with new environment
    if len(command) > 0:
        proc = subprocess.Popen(command, env=env)
        status = proc.wait()
        sys.exit(status if status >= 0 else 128 - status)

if __name__ == "__main__":
    main()