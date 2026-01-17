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

# Check if a cell exists in comma-separated list
has_cell() {
    echo ",$1," | grep -q ",$2,"
}

# Show usage
show_help() {
    echo "Usage: $0 <commands>"
    echo ""
    echo "Commands (comma-separated):"
    echo "  setup       - Download dependencies (Blaze, Zig)"
    echo "  build       - Build all targets"
    echo "  cpp-bench   - Run C++ benchmark"
    echo "  zig-bench   - Run Zig benchmark"
    echo "  compare     - Run both benchmarks"
    echo "  cpp-example - Run C++ example"
    echo "  zig-example - Run Zig example"
    echo ""
    echo "Examples:"
    echo "  $0 setup,build,compare"
    echo "  $0 setup,build,zig-bench"
}

COMMANDS="$1"

if [ -z "$COMMANDS" ] || [ "$COMMANDS" = "help" ] || [ "$COMMANDS" = "-h" ] || [ "$COMMANDS" = "--help" ]; then
    show_help
    exit 0
fi

if has_cell "$COMMANDS" "setup"; then
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
fi

if has_cell "$COMMANDS" "build"; then
    echo "Building C++ benchmark..."
    pixi run -- bash -c 'mkdir -p build-cpp && cd build-cpp && cmake ../cpp-bench -G Ninja && ninja'
    echo ""
    echo "Building Zig benchmark..."
    cd "$PROJECT_DIR/zig-blaze"
    MKLROOT="$PIXI_ENV" "$ZIG_DIR/zig" build -Doptimize=ReleaseFast
    cd "$PROJECT_DIR"
    echo ""
    echo "Build complete!"
fi

if has_cell "$COMMANDS" "cpp-bench"; then
    echo -e "${BLUE}Running C++ Blaze Benchmark...${NC}"
    "$PROJECT_DIR/build-cpp/blaze_bench"
fi

if has_cell "$COMMANDS" "zig-bench"; then
    echo -e "${GREEN}Running Zig Blaze Benchmark...${NC}"
    "$PROJECT_DIR/zig-blaze/zig-out/bin/blaze_zig_bench"
fi

if has_cell "$COMMANDS" "cpp-example"; then
    echo -e "${BLUE}Running C++ Blaze Example...${NC}"
    "$PROJECT_DIR/build-cpp/blaze_example"
fi

if has_cell "$COMMANDS" "zig-example"; then
    echo -e "${GREEN}Running Zig Blaze Example...${NC}"
    "$PROJECT_DIR/zig-blaze/zig-out/bin/blaze_zig_example"
fi

if has_cell "$COMMANDS" "compare"; then
    echo "========================================"
    echo "  Blaze C++ vs Zig Benchmark Comparison"
    echo "========================================"
    echo ""
    echo -e "${BLUE}=== C++ Blaze (with MKL) ===${NC}"
    "$PROJECT_DIR/build-cpp/blaze_bench"
    echo ""
    echo -e "${GREEN}=== Zig Blaze (with MKL) ===${NC}"
    "$PROJECT_DIR/zig-blaze/zig-out/bin/blaze_zig_bench"
fi
