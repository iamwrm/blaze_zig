#!/bin/bash
# Set MKL environment variables
export MKLROOT="${CONDA_PREFIX}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export CPATH="${CONDA_PREFIX}/include:${CPATH}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH}"

# Set threading to single thread for benchmarks
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
