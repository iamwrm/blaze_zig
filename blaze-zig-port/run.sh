#!/bin/bash
# Run script for Blaze C++ and Zig comparison benchmarks

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIXI_ENV="$PROJECT_DIR/.pixi/envs/default"
ZIG_DIR="$PROJECT_DIR/zig-linux-x86_64-0.13.0"

# Set environment
export MKLROOT="$PIXI_ENV"
export LD_LIBRARY_PATH="$PIXI_ENV/lib:$LD_LIBRARY_PATH"
export LD_PRELOAD="$PIXI_ENV/lib/libmkl_core.so:$PIXI_ENV/lib/libmkl_sequential.so"
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

case "$1" in
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
    "build")
        echo "Building C++ benchmark..."
        mkdir -p "$PROJECT_DIR/build-cpp"
        cd "$PROJECT_DIR/build-cpp"
        cmake ../cpp-bench -G Ninja
        ninja
        echo ""
        echo "Building Zig benchmark..."
        cd "$PROJECT_DIR/zig-blaze"
        "$ZIG_DIR/zig" build -Doptimize=ReleaseFast
        echo ""
        echo "Build complete!"
        ;;
    *)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  compare     - Run both benchmarks (default)"
        echo "  cpp-bench   - Run C++ benchmark only"
        echo "  zig-bench   - Run Zig benchmark only"
        echo "  cpp-example - Run C++ example"
        echo "  zig-example - Run Zig example"
        echo "  build       - Build all targets"
        ;;
esac
