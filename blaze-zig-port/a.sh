cd $(dirname $BASH_SOURCE[0])

# 2. Install dependencies
pixi install

# 3. Download Blaze C++ library (on-demand)
if [ ! -d "blaze" ]; then
    curl -L -o blaze.tar.gz 'https://github.com/live-clones/blaze/archive/refs/tags/v3.8.2.tar.gz'
    tar xzf blaze.tar.gz && mv blaze-3.8.2 blaze && rm blaze.tar.gz
fi

# 4. Download Zig 0.13.0 (on-demand)
if [ ! -d "zig-0.13.0" ]; then
    curl -L -o zig.tar.xz 'https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz'
    tar xf zig.tar.xz && mv zig-linux-x86_64-0.13.0 zig-0.13.0 && rm zig.tar.xz
fi

# 5. Build C++
pixi run -- bash -c 'mkdir -p build-cpp && cd build-cpp && cmake ../cpp-bench -G Ninja && ninja'

# 6. Build Zig
cd zig-blaze && MKLROOT=$(pwd)/../.pixi/envs/default ../zig-0.13.0/zig build -Doptimize=ReleaseFast && cd ..

# 7. Run benchmarks
./run.sh zig-bench
./run.sh compare