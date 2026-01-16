#!/bin/bash
set -e

echo "==================================="
echo "Blaze Benchmark Suite"
echo "==================================="
echo

# Check if pixi is available
if ! command -v pixi &> /dev/null; then
    echo "Error: pixi is not installed."
    echo "Install it with: curl -fsSL https://pixi.sh/install.sh | bash"
    exit 1
fi

# Setup Blaze library
echo "Setting up Blaze library..."
pixi run setup-blaze
echo

# Setup Zig compiler
echo "Setting up Zig compiler..."
pixi run setup-zig
echo

# Build and benchmark C++
echo "==================================="
echo "Building C++ (Blaze + MKL)..."
echo "==================================="
pixi run build-cpp
echo

echo "Running C++ example..."
pixi run run-cpp-example
echo

echo "==================================="
echo "C++ Benchmark (Blaze + MKL)"
echo "==================================="
pixi run benchmark-cpp
echo

# Build and benchmark Zig
echo "==================================="
echo "Building Zig (Pure Zig)..."
echo "==================================="
pixi run build-zig
echo

echo "Running Zig tests..."
pixi run test-zig
echo

echo "Running Zig example..."
pixi run run-zig-example
echo

echo "==================================="
echo "Zig Benchmark (Pure Zig)"
echo "==================================="
pixi run benchmark-zig
echo

echo "==================================="
echo "All benchmarks completed!"
echo "==================================="
