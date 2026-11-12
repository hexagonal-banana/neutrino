import os
import sys
import subprocess
from pprint import pprint

try:
    import toml
except:
    import pip
    pip.main(["install", "toml"])
    import toml

CURDIR = os.path.dirname(os.path.realpath(__file__))

def check_command(cmd: str):
    try:
        _ = subprocess.run([cmd], stdout=subprocess.PIPE,  
            stderr=subprocess.PIPE, text=True, check=True)
        return True
    except FileNotFoundError:
        return False

# Use rocm-smi or nvidia-smi to detect if installed
if check_command("rocm-smi"):
    NEUTRINO_MODE = "HIP"
    NEUTRINO_DRIVER_HEADER_NAME = "hip/hip_runtime_api.h"
    NEUTRINO_IMPL_SRC = "hip.c"
    NEUTRINO_HOOK_DRIVER_LIB_NAME = "libamdhip64.so.6"
    NEUTRINO_DRIVER_HEADER_SEARCH_PATH = [
        "/opt/rocm/include/", # AFAIK, add if new path is met
    ]
    extra_flags = ["-D__HIP_PLATFORM_AMD__"]
elif check_command("nvidia-smi"):
    NEUTRINO_MODE = "CUDA"
    NEUTRINO_DRIVER_HEADER_NAME = "cuda.h"
    NEUTRINO_IMPL_SRC = "cuda.c"
    NEUTRINO_HOOK_DRIVER_LIB_NAME = "libcuda.so.1"
    NEUTRINO_DRIVER_HEADER_SEARCH_PATH = [
        os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "include"),
        "/usr/local/cuda/targets/x86_64-linux/include/", # for x86
        "/usr/local/cuda/targets/aarch64-linux/include/", # for ARM
        # add if missed
    ]
    extra_flags = []
else:
    raise RuntimeError("ONLY SUPPORT CUDA and HIP(AMD-ONLY)")

# Internal Configurations
SRC_DIR = os.path.join(CURDIR, "src")
BUILD_DIR = os.path.join(CURDIR, "build")
CC = "cc" # NOTE don't use nvcc or hipcc, need gcc or clang
PY = sys.executable

NEUTRINO_DRIVER_HEADER_DIR = next(
    (directory for directory in NEUTRINO_DRIVER_HEADER_SEARCH_PATH
     if os.path.isfile(os.path.join(directory, NEUTRINO_DRIVER_HEADER_NAME))), None)
if NEUTRINO_DRIVER_HEADER_DIR is None:
    raise RuntimeError(f"Driver header {NEUTRINO_DRIVER_HEADER_NAME} was not found")

# NOTE Locate Driver Shared Library
# inspired by: https://github.com/triton-lang/triton/commit/58c54455ffa691be64f90f4e856501162373572c#diff-3d1f29795218f61553ab953426c15fa1e4162b224405b85529022293054da57aR25
# but we need to further locate the real driver library
libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()
locs = [line.split()[-1] for line in libs.splitlines() if NEUTRINO_HOOK_DRIVER_LIB_NAME in line]
env_ld_library_path = os.getenv("LD_LIBRARY_PATH", "")
if NEUTRINO_MODE == "CUDA":
    # CUDA compatibility drivers take precedence over the system ldconfig entry.
    # Resolve the actual image so preload never redirects this path to our hook.
    locs = [os.path.join(directory, NEUTRINO_HOOK_DRIVER_LIB_NAME)
            for directory in env_ld_library_path.split(":") if directory] + locs
    if os.getenv("NEUTRINO_REAL_DRIVER"):
        locs.insert(0, os.environ["NEUTRINO_REAL_DRIVER"])
    hook = os.path.realpath(os.path.join(BUILD_DIR, NEUTRINO_HOOK_DRIVER_LIB_NAME))
    real_driver = next((os.path.realpath(path) for path in locs
                        if os.path.isfile(path) and os.path.realpath(path) != hook
                        and not any(part in path or part in os.path.realpath(path)
                                    for part in ("lib32", "i386"))), None)
    if real_driver is None:
        raise RuntimeError("A real CUDA driver was not found")
    NEUTRINO_REAL_DRIVER_LIB_DIR, NEUTRINO_REAL_DRIVER_LIB_NAME = os.path.split(real_driver)
else:
    if env_ld_library_path and not locs:
        locs = [os.path.join(dir_, NEUTRINO_HOOK_DRIVER_LIB_NAME) for dir_ in env_ld_library_path.split(":")
                if os.path.exists(os.path.join(dir_, NEUTRINO_HOOK_DRIVER_LIB_NAME))]
    NEUTRINO_REAL_DRIVER_LIB_NAME = ""
    NEUTRINO_REAL_DRIVER_LIB_DIR = ""
    for loc in locs:
        real_lib = os.readlink(loc)
        if "lib32" not in loc and "lib32" not in real_lib and "i386" not in loc and "i386" not in real_lib:
            NEUTRINO_REAL_DRIVER_LIB_DIR = os.path.dirname(loc)
            NEUTRINO_REAL_DRIVER_LIB_NAME = real_lib if not real_lib.startswith("/") else os.path.basename(loc)

print(NEUTRINO_REAL_DRIVER_LIB_DIR, NEUTRINO_REAL_DRIVER_LIB_NAME, file=sys.stderr)

# NOTE call parse.py
cmd = [PY, os.path.join(SRC_DIR, "parse.py"), 
    os.path.join(NEUTRINO_DRIVER_HEADER_DIR, NEUTRINO_DRIVER_HEADER_NAME), 
    os.path.join(NEUTRINO_REAL_DRIVER_LIB_DIR, NEUTRINO_REAL_DRIVER_LIB_NAME)]
print(" ".join(cmd), file=sys.stderr)
subprocess.check_output(cmd)

# NOTE compile cuda.c/hip.c with common.h
cmd = [CC, os.path.join(SRC_DIR, NEUTRINO_IMPL_SRC), "-fPIC", "-shared", "-ldl", "-lpthread", "-O3", *extra_flags,
        "-I", NEUTRINO_DRIVER_HEADER_DIR, "-o", os.path.join(BUILD_DIR, NEUTRINO_HOOK_DRIVER_LIB_NAME)]
print(" ".join(cmd), file=sys.stderr)
subprocess.check_output(cmd)

# NOTE compile preload.c
cmd = [CC, os.path.join(SRC_DIR, "preload.c"), "-fPIC", "-shared", "-O3", 
       "-o", os.path.join(BUILD_DIR, "preload.so")]
print(" ".join(cmd), file=sys.stderr)
subprocess.check_output(cmd)

# NOTE create a symbolic link like libcuda.so -> libcuda.so.1
# TODO verify if this is need 
cmd = ["ln", "-sf", NEUTRINO_HOOK_DRIVER_LIB_NAME, 
       os.path.join(BUILD_DIR, NEUTRINO_HOOK_DRIVER_LIB_NAME[:NEUTRINO_HOOK_DRIVER_LIB_NAME.index("so") + 2])]
print(" ".join(cmd), file=sys.stderr)
subprocess.check_output(cmd)

# NOTE dump system configuration for CLI usage
config = {}
config["system"] = {
    "NEUTRINO_MODE"                 : NEUTRINO_MODE,
    "NEUTRINO_DRIVER_HEADER_NAME"   : NEUTRINO_DRIVER_HEADER_NAME,
    "NEUTRINO_DRIVER_HEADER_DIR"    : NEUTRINO_DRIVER_HEADER_DIR,
    "NEUTRINO_HOOK_DRIVER_LIB_NAME" : NEUTRINO_HOOK_DRIVER_LIB_NAME,
    "NEUTRINO_REAL_DRIVER_LIB_NAME" : NEUTRINO_REAL_DRIVER_LIB_NAME,
    "NEUTRINO_REAL_DRIVER_LIB_DIR"  : NEUTRINO_REAL_DRIVER_LIB_DIR,
}
toml.dump(config, open(os.path.join(BUILD_DIR, "config.toml"), "w"))

print("Build Success, Configuration")
print("============================")
pprint(config)