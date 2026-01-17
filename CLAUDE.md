# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

Blaze-Zig is a minimal port of the Blaze C++ linear algebra library to Zig. It provides high-performance dense matrix operations backed by Intel MKL (Math Kernel Library) for BLAS computations.

## Key Commands

All commands go through `run.sh`:

```bash
./run.sh setup          # Build Blaze package and install dependencies
./run.sh build          # Build both C++ and Zig benchmarks
./run.sh compare        # Run both benchmarks and compare results
./run.sh zig-bench      # Run only Zig benchmark
./run.sh cpp-bench      # Run only C++ benchmark
./run.sh zig-example    # Run Zig example
./run.sh cpp-example    # Run C++ example
```

Commands can be chained: `./run.sh setup,build,compare`

## Project Structure

```
blaze_zig/
├── zig-blaze/                # Main Zig implementation
│   ├── build.zig             # Zig build configuration
│   └── src/
│       ├── blaze.zig         # Core library (DynamicMatrix, MKL bindings)
│       ├── bench.zig         # Benchmark program
│       └── example.zig       # Usage example
├── cpp-bench/                # C++ reference implementation
│   ├── CMakeLists.txt        # CMake config
│   ├── main.cpp              # C++ benchmark
│   └── example.cpp           # C++ example
├── recipes/                  # Local conda package recipes
│   └── blaze/recipe.yaml     # Blaze C++ library package recipe
├── bootstrap/                # Bootstrap environment for building local packages
│   └── pixi.toml             # rattler-build dependency
├── run.sh                    # Build/run orchestration script
├── pixi.toml                 # Dependency management (CMake, MKL, Zig, Blaze)
└── .github/workflows/ci.yml  # GitHub Actions CI
```

## Architecture

- **DynamicMatrix**: Generic matrix type supporting f32/f64 with row-major or column-major storage
- **MKL Integration**: Uses CBLAS via `@cImport` for matrix multiplication (dgemm/sgemm)
- **Memory**: Uses Zig allocators; call `deinit()` to free matrices

## Build System

Zig build (`zig-blaze/build.zig`):
- Links MKL: `mkl_intel_lp64`, `mkl_sequential`, `mkl_core`
- Links system: `pthread`, `m`, `dl`, `c`
- Build with: `zig build -Doptimize=ReleaseFast`

## Testing

Run Zig tests:
```bash
cd zig-blaze && zig build test
```

## CI/CD

GitHub Actions runs on pushes to `main` and `claude/*` branches. Check status with:
```bash
gh run list --limit 5
gh run view --log     # View latest run logs
```

## Environment

Pixi manages dependencies. MKL environment variables are set automatically in `run.sh`.

### Local Channel

Blaze C++ library is built as a local conda package using rattler-build:
- Recipe: `recipes/blaze/recipe.yaml`
- Output: `local-channel/` (built at setup time, not committed)
- Bootstrap environment: `pixi run -e bootstrap build-blaze`
