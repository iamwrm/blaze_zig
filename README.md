# Blaze C++ to Zig Port

A minimal port of the [Blaze C++ linear algebra library](https://bitbucket.org/blaze-lib/blaze) to Zig, using Intel MKL for high-performance matrix operations.

## Project Structure

```
blaze-zig-port/
├── pixi.toml               # Package manager configuration
├── activate.sh             # MKL environment setup
├── run.sh                  # Run script for benchmarks
├── blaze/                  # Original Blaze C++ library (header-only)
├── cpp-bench/              # C++ benchmark and example
│   ├── CMakeLists.txt
│   ├── main.cpp            # Benchmark
│   └── example.cpp         # Example usage
├── zig-blaze/              # Zig port
│   ├── build.zig
│   └── src/
│       ├── blaze.zig       # Core library
│       ├── bench.zig       # Benchmark
│       └── example.zig     # Example usage
└── build-cpp/              # C++ build directory
```

## Features

The Zig port implements:
- **DynamicMatrix**: Row-major and column-major dense matrices
- **Matrix operations**: Multiplication (MKL CBLAS), addition, subtraction, scalar multiplication
- **Element access**: get/set/ptr methods
- **Utilities**: trace, clone, fillRandom, print

## Requirements

All dependencies are managed via [pixi](https://pixi.sh):
- CMake, Ninja (build tools)
- GCC (C++ compiler)
- Intel MKL (BLAS/LAPACK)
- Zig (downloaded separately due to conda-forge version issues)

## Setup

```bash
# Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Install dependencies
cd blaze-zig-port
pixi install

# Build everything
./run.sh build
```

## Usage

### Run Benchmarks

```bash
# Run both C++ and Zig benchmarks
./run.sh compare

# Run individual benchmarks
./run.sh cpp-bench
./run.sh zig-bench
```

### Run Examples

```bash
./run.sh cpp-example
./run.sh zig-example
```

## Benchmark Results

Single-threaded MKL benchmark on matrix multiplication:

### Double Precision (f64)

| Size | C++ Time (ms) | C++ GFLOPS | Zig Time (ms) | Zig GFLOPS |
|------|---------------|------------|---------------|------------|
| 64   | 0.010         | 53.6       | 0.019         | 27.3       |
| 128  | 0.064         | 65.1       | 0.058         | 72.1       |
| 256  | 0.472         | 71.1       | 0.475         | 70.6       |
| 512  | 3.709         | 72.4       | 3.630         | 73.9       |
| 1024 | 28.501        | 75.3       | 28.587        | 75.1       |
| 2048 | 221.594       | 77.5       | 225.389       | 76.2       |

### Single Precision (f32)

| Size | C++ Time (ms) | C++ GFLOPS | Zig Time (ms) | Zig GFLOPS |
|------|---------------|------------|---------------|------------|
| 64   | 0.011         | 49.5       | 0.010         | 51.5       |
| 128  | 0.042         | 100.8      | 0.031         | 137.3      |
| 256  | 0.232         | 144.7      | 0.235         | 142.9      |
| 512  | 1.750         | 153.4      | 1.753         | 153.1      |
| 1024 | 13.965        | 153.8      | 14.054        | 152.8      |
| 2048 | 109.841       | 156.4      | 111.295       | 154.4      |

Both implementations achieve similar performance since they both use MKL BLAS for the core computation.

## Example Usage (Zig)

```zig
const std = @import("std");
const blaze = @import("blaze");
const DynamicMatrix = blaze.DynamicMatrix;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const Mat = DynamicMatrix(f64, .RowMajor);
    
    // Create 3x3 matrices
    var A = try Mat.initWith(allocator, 3, 3, 0);
    defer A.deinit();
    
    // Set values
    A.set(0, 0, 1.0);
    A.set(1, 1, 2.0);
    A.set(2, 2, 3.0);
    
    var B = try Mat.initWith(allocator, 3, 3, 1.0);
    defer B.deinit();
    
    // Matrix multiplication using MKL
    var C = try Mat.multiply(allocator, &A, &B);
    defer C.deinit();
    
    // Print result
    try C.print(std.io.getStdOut().writer());
}
```

## License

The original Blaze library is licensed under the BSD 3-Clause license.
This port follows the same license.
