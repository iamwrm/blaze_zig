#!/bin/bash
# Run script for Blaze C++ and Zig comparison benchmarks

set -e

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"
PIXI_ENV="$PROJECT_DIR/.pixi/envs/default"
ZIG_DIR="$PROJECT_DIR/zig-0.13.0"

# Set environment
export MKLROOT="$PIXI_ENV"
export LD_LIBRARY_PATH="$PIXI_ENV/lib:$LD_LIBRARY_PATH"
export CPATH="$PIXI_ENV/include:${CPATH}"
export LIBRARY_PATH="$PIXI_ENV/lib:${LIBRARY_PATH}"
export LD_PRELOAD="$PIXI_ENV/lib/libmkl_core.so:$PIXI_ENV/lib/libmkl_sequential.so"
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

case "$1" in
    "setup")
        echo "Installing pixi dependencies..."
        pixi install

        echo "Downloading Blaze C++ library..."
        if [ ! -d "blaze" ]; then
            curl -L -o blaze.tar.gz 'https://github.com/live-clones/blaze/archive/refs/tags/v3.8.2.tar.gz'
            tar xzf blaze.tar.gz && mv blaze-3.8.2 blaze && rm blaze.tar.gz
        fi

        echo "Downloading Zig 0.13.0..."
        if [ ! -d "zig-0.13.0" ]; then
            curl -L -o zig.tar.xz 'https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz'
            tar xf zig.tar.xz && mv zig-linux-x86_64-0.13.0 zig-0.13.0 && rm zig.tar.xz
        fi
        echo "Setup complete!"
        ;;
    "build")
        echo "Building C++ benchmark..."
        pixi run -- bash -c 'mkdir -p build-cpp && cd build-cpp && cmake ../cpp-bench -G Ninja && ninja'
        echo ""
        echo "Building Zig benchmark..."
        cd "$PROJECT_DIR/zig-blaze"
        MKLROOT="$PIXI_ENV" "$ZIG_DIR/zig" build -Doptimize=ReleaseFast
        echo ""
        echo "Build complete!"
        ;;
    "ci")
        # Full CI: setup, build, and run benchmarks
        "$0" setup
        "$0" build
        "$0" zig-bench
        "$0" compare
        ;;
    "cpp-bench")
        echo -e "${BLUE}Running C++ Blaze Benchmark...${NC}"
        "$PROJECT_DIR/build-cpp/blaze_bench"
        ;;
    "zig-bench")
        echo -e "${GREEN}Running Zig Blaze Benchmark...${NC}"
        "$PROJECT_DIR/zig-blaze/zig-out/bin/blaze_zig_bench"
        ;;
    "cpp-example")
        echo -e "${BLUE}Running C++ Blaze Example...${NC}"
        "$PROJECT_DIR/build-cpp/blaze_example"
        ;;
    "zig-example")
        echo -e "${GREEN}Running Zig Blaze Example...${NC}"
        "$PROJECT_DIR/zig-blaze/zig-out/bin/blaze_zig_example"
        ;;
    "compare"|"")
        echo "========================================"
        echo "  Blaze C++ vs Zig Benchmark Comparison"
        echo "========================================"
        echo ""
        echo -e "${BLUE}=== C++ Blaze (with MKL) ===${NC}"
        "$PROJECT_DIR/build-cpp/blaze_bench"
        echo ""
        echo -e "${GREEN}=== Zig Blaze (with MKL) ===${NC}"
        "$PROJECT_DIR/zig-blaze/zig-out/bin/blaze_zig_bench"
        ;;
    *)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  setup       - Download dependencies (Blaze, Zig)"
        echo "  build       - Build all targets"
        echo "  ci          - Full CI: setup + build + benchmarks"
        echo "  compare     - Run both benchmarks (default)"
        echo "  cpp-bench   - Run C++ benchmark only"
        echo "  zig-bench   - Run Zig benchmark only"
        echo "  cpp-example - Run C++ example"
        echo "  zig-example - Run Zig example"
        ;;
esac
