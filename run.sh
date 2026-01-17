#!/bin/bash
# Run script for Blaze C++ and Zig comparison benchmarks

set -e

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"
PIXI_ENV="$PROJECT_DIR/.pixi/envs/default"
ZIG="$PIXI_ENV/bin/zig"

# MKL environment configuration
export MKLROOT="$PIXI_ENV"
export LD_LIBRARY_PATH="$PIXI_ENV/lib:$LD_LIBRARY_PATH"
export CPATH="$PIXI_ENV/include:${CPATH}"
export LIBRARY_PATH="$PIXI_ENV/lib:${LIBRARY_PATH}"
export LD_PRELOAD="$PIXI_ENV/lib/libmkl_core.so:$PIXI_ENV/lib/libmkl_sequential.so"
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Output colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RESET='\033[0m'

# Check if command exists in comma-separated list
has_command() {
    [[ ",$COMMAND_LIST," == *",$1,"* ]]
}

show_help() {
    cat <<EOF
Usage: $0 <commands>

Commands (comma-separated):
  setup       - Install pixi dependencies and download Blaze
  build       - Build all targets
  cpp-bench   - Run C++ benchmark
  zig-bench   - Run Zig benchmark
  compare     - Run both benchmarks
  cpp-example - Run C++ example
  zig-example - Run Zig example
  clean       - Remove build artifacts

Examples:
  $0 setup,build,compare
  $0 setup,build,zig-bench
EOF
}

COMMAND_LIST="$1"

if [ -z "$COMMAND_LIST" ] || [ "$COMMAND_LIST" = "help" ] || [ "$COMMAND_LIST" = "-h" ] || [ "$COMMAND_LIST" = "--help" ]; then
    show_help
    exit 0
fi

# Executable paths
CPP_BENCH="$PROJECT_DIR/build-cpp/blaze_bench"
CPP_EXAMPLE="$PROJECT_DIR/build-cpp/blaze_example"
ZIG_BENCH="$PROJECT_DIR/zig-blaze/zig-out/bin/blaze_zig_bench"
ZIG_EXAMPLE="$PROJECT_DIR/zig-blaze/zig-out/bin/blaze_zig_example"

if has_command "setup"; then
    # Build Blaze package if not already built
    if ! ls local-channel/noarch/blaze-*.conda 1>/dev/null 2>&1; then
        echo "Building Blaze package with rattler-build..."

        # Use separate bootstrap project to avoid dependency conflicts
        (cd bootstrap && pixi run build-blaze)
    fi

    echo "Installing pixi dependencies (including Blaze from local channel)..."
    pixi install

    echo "Setup complete!"
fi

if has_command "build"; then
    echo "Building C++ benchmark..."
    pixi run -- bash -c 'mkdir -p build-cpp && cd build-cpp && cmake ../cpp-bench -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && ninja'

    echo ""
    echo "Building Zig benchmark..."
    (cd "$PROJECT_DIR/zig-blaze" && "$ZIG" build -Doptimize=ReleaseFast)

    echo ""
    echo "Build complete!"
fi

if has_command "clean"; then
    echo "Cleaning build artifacts..."
    rm -rf "$PROJECT_DIR/build-cpp"
    rm -rf "$PROJECT_DIR/zig-blaze/zig-out"
    rm -rf "$PROJECT_DIR/zig-blaze/.zig-cache"
    echo "Clean complete!"
fi

if has_command "cpp-bench"; then
    echo -e "${BLUE}Running C++ Blaze Benchmark...${RESET}"
    "$CPP_BENCH"
fi

if has_command "zig-bench"; then
    echo -e "${GREEN}Running Zig Blaze Benchmark...${RESET}"
    "$ZIG_BENCH"
fi

if has_command "cpp-example"; then
    echo -e "${BLUE}Running C++ Blaze Example...${RESET}"
    "$CPP_EXAMPLE"
fi

if has_command "zig-example"; then
    echo -e "${GREEN}Running Zig Blaze Example...${RESET}"
    "$ZIG_EXAMPLE"
fi

if has_command "compare"; then
    echo "========================================"
    echo "  Blaze C++ vs Zig Benchmark Comparison"
    echo "========================================"
    echo ""
    echo -e "${BLUE}=== C++ Blaze (with MKL) ===${RESET}"
    "$CPP_BENCH"
    echo ""
    echo -e "${GREEN}=== Zig Blaze (with MKL) ===${RESET}"
    "$ZIG_BENCH"
fi
