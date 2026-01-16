# Blaze-Zig

A minimal port of the [Blaze C++ linear algebra library](https://bitbucket.org/blaze-lib/blaze) to Zig.

## Overview

This project provides:
- C++ examples using the original Blaze library with MKL backend
- A minimal Blaze-like linear algebra library implemented in pure Zig
- Benchmarks comparing both implementations

## Requirements

All dependencies are managed by [Pixi](https://pixi.sh/). No system-level installation is required.

Managed dependencies:
- C++ compiler (GCC/Clang via cxx-compiler)
- CMake and Ninja
- Intel MKL (for BLAS/LAPACK)
- Zig 0.13.x

## Quick Start

### Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc
```

### Build and Run

```bash
# Install dependencies
pixi install

# Clone Blaze library (C++ header-only)
pixi run setup-blaze

# Build and run C++ example
pixi run run-cpp-example

# Build and run Zig example
cd zig && pixi run -- zig build run -Doptimize=ReleaseFast
```

### Run Benchmarks

```bash
# C++ benchmark (MKL-accelerated)
pixi run benchmark-cpp

# Zig benchmark (pure Zig)
cd zig && pixi run -- zig build benchmark -Doptimize=ReleaseFast
```

### Run Tests

```bash
# Zig tests
cd zig && pixi run -- zig build test
```

## Project Structure

```
blaze_zig/
├── pixi.toml              # Pixi configuration with all dependencies
├── cpp/                   # C++ Blaze examples
│   ├── CMakeLists.txt
│   ├── example.cpp        # Basic matrix/vector operations
│   └── benchmark.cpp      # Matrix multiplication benchmark
├── zig/                   # Zig implementation
│   ├── build.zig
│   └── src/
│       ├── blaze.zig      # Main module
│       ├── matrix.zig     # DynamicMatrix implementation
│       ├── vector.zig     # DynamicVector implementation
│       ├── main.zig       # Example application
│       └── benchmark.zig  # Benchmark application
└── external/
    └── blaze/             # Cloned Blaze C++ library (git ignored)
```

## API Reference

### Zig API

```zig
const blaze = @import("blaze");

// Dynamic Matrix
const Matrix = blaze.DynamicMatrix(f64);
var mat = try Matrix.init(allocator, rows, cols);
defer mat.deinit();

mat.set(0, 0, 1.0);
const val = mat.get(0, 0);

// Matrix operations
var c = try Matrix.multiply(allocator, a, b);  // C = A * B
var d = try Matrix.add(allocator, a, b);       // D = A + B
var e = try Matrix.scale(allocator, a, 2.0);   // E = A * 2

// Dynamic Vector
const Vector = blaze.DynamicVector(f64);
var vec = try Vector.fromArray(allocator, 3, .{1.0, 2.0, 3.0});
defer vec.deinit();

const dot = try Vector.inner(v1, v2);  // Dot product
var v3 = try Vector.add(allocator, v1, v2);
```

## Benchmark Results

Example benchmark results on a typical system:

### C++ Blaze with MKL (single-threaded)
| Size | Time (ms) | GFLOPS |
|------|-----------|--------|
| 64   | 0.012     | 45.37  |
| 128  | 0.066     | 63.86  |
| 256  | 0.484     | 69.26  |
| 512  | 3.692     | 72.70  |
| 1024 | 28.401    | 75.61  |
| 2048 | 218.818   | 78.51  |

### Pure Zig (single-threaded, tiled)
| Size | Time (ms) | GFLOPS |
|------|-----------|--------|
| 64   | 0.316     | 1.66   |
| 128  | 1.804     | 2.32   |
| 256  | 13.637    | 2.46   |
| 512  | 104.834   | 2.56   |
| 1024 | 855.149   | 2.51   |
| 2048 | 6693.162  | 2.57   |

Note: The Zig implementation uses a naive tiled algorithm. For production use, consider linking to optimized BLAS libraries.

## License

MIT License
