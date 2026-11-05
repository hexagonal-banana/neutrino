#!/usr/bin/env bash
# Isolated environment for the unified tracing experiment.
export UNIFIED_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export UNIFIED_TOOLCHAIN="$UNIFIED_WORKSPACE/../deepep-monitor/.env"
export UNIFIED_COMPAT="$UNIFIED_WORKSPACE/../deepep-monitor/.local/cuda-compat/usr/local/cuda-13.1/compat"
export VIRTUAL_ENV="$UNIFIED_WORKSPACE/.venv-unified"
export CUDA_HOME="$UNIFIED_TOOLCHAIN" CUDA_PATH="$UNIFIED_TOOLCHAIN"
export PATH="$VIRTUAL_ENV/bin:$UNIFIED_TOOLCHAIN/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH LD_PRELOAD
export CC="$UNIFIED_TOOLCHAIN/bin/x86_64-conda-linux-gnu-cc"
export CXX="$UNIFIED_TOOLCHAIN/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CXX" NVCC_CCBIN="$CXX"
export EP_JIT_NVCC_COMPILER="$UNIFIED_TOOLCHAIN/bin/nvcc"
export EP_NCCL_ROOT_DIR="$UNIFIED_WORKSPACE/.local/unified/nccl"
export NEUTRINO_BUILD_DIR="$UNIFIED_WORKSPACE/.build/unified/neutrino"
export NEUTRINO_REAL_DRIVER="$UNIFIED_COMPAT/libcuda.so.1"
export EP_JIT_CACHE_DIR="$UNIFIED_WORKSPACE/.cache/unified/deepjit"
export EP_JIT_DUMP_PTX=1 EP_JIT_WITH_LINEINFO=1 EP_DISABLE_GIN=1
export NCCL_LSA_TEAM_SIZE=0
export TORCH_EXTENSIONS_DIR="$UNIFIED_WORKSPACE/.cache/unified/torch_extensions"
export TRITON_CACHE_DIR="$UNIFIED_WORKSPACE/.cache/unified/triton"
export CUDA_CACHE_PATH="$UNIFIED_WORKSPACE/.cache/unified/cuda"
export XDG_CACHE_HOME="$UNIFIED_WORKSPACE/.cache/unified"
export PIP_CACHE_DIR="$UNIFIED_WORKSPACE/.cache/unified/pip"
export TMPDIR="$UNIFIED_WORKSPACE/.cache/unified/tmp"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$UNIFIED_COMPAT:$EP_NCCL_ROOT_DIR/lib:$UNIFIED_TOOLCHAIN/lib:$UNIFIED_TOOLCHAIN/targets/x86_64-linux/lib"
# Load the pinned SONAME first; the overlay also redirects PyTorch's absolute-path discovery.
if [[ -f "$EP_NCCL_ROOT_DIR/lib/libnccl.so.2" ]]; then
    export LD_PRELOAD="$EP_NCCL_ROOT_DIR/lib/libnccl.so.2"
fi
for _lib in "$UNIFIED_TOOLCHAIN"/lib/python3.12/site-packages/nvidia/*/lib; do
    [[ "$_lib" == */nvidia/nccl/lib ]] && continue
    [[ -d "$_lib" ]] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$_lib"
done
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/targets/x86_64-linux/include/cccl:$EP_NCCL_ROOT_DIR/include"
export LIBRARY_PATH="$EP_NCCL_ROOT_DIR/lib:$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib"
unset _lib
