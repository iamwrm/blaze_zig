// Blaze Matrix Multiplication Benchmark
// Single-threaded MKL-backed matrix multiplication

#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <blaze/Blaze.h>

// Disable parallelism - single thread only
#define BLAZE_USE_SHARED_MEMORY_PARALLELIZATION 0

using namespace blaze;

// High-resolution timing
template<typename Func>
double benchmark(Func f, int warmup_runs = 3, int timed_runs = 10) {
    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        f();
    }
    
    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timed_runs; ++i) {
        f();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / timed_runs;
}

// Calculate GFLOPS for matrix multiplication
double calculate_gflops(size_t M, size_t N, size_t K, double time_ms) {
    // Matrix multiplication: C(MxN) = A(MxK) * B(KxN)
    // Operations: 2 * M * N * K (multiply + add)
    double ops = 2.0 * M * N * K;
    double seconds = time_ms / 1000.0;
    return (ops / seconds) / 1e9;
}

template<typename T>
void run_benchmark(const std::string& type_name, size_t size) {
    std::cout << "\n=== " << type_name << " Matrix Multiplication ===" << std::endl;
    std::cout << "Matrix size: " << size << "x" << size << std::endl;
    
    // Create matrices
    DynamicMatrix<T, rowMajor> A(size, size);
    DynamicMatrix<T, rowMajor> B(size, size);
    DynamicMatrix<T, rowMajor> C(size, size);
    
    // Initialize with random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist(0.0, 1.0);
    
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            A(i, j) = dist(gen);
            B(i, j) = dist(gen);
        }
    }
    
    // Benchmark matrix multiplication
    auto matmul = [&]() {
        C = A * B;
    };
    
    double time_ms = benchmark(matmul, 3, 10);
    double gflops = calculate_gflops(size, size, size, time_ms);
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Time: " << time_ms << " ms" << std::endl;
    std::cout << "GFLOPS: " << gflops << std::endl;
    
    // Verify result (compute checksum)
    T sum = 0;
    for (size_t i = 0; i < std::min(size, size_t(10)); ++i) {
        sum += C(i, i);
    }
    std::cout << "Checksum (trace of first 10x10): " << sum << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   Blaze C++ Matrix Multiplication" << std::endl;
    std::cout << "   Single-threaded MKL Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Test different matrix sizes
    std::vector<size_t> sizes = {64, 128, 256, 512, 1024, 2048};
    
    for (size_t size : sizes) {
        run_benchmark<double>("Double Precision (f64)", size);
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "   Single Precision Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    for (size_t size : sizes) {
        run_benchmark<float>("Single Precision (f32)", size);
    }
    
    return 0;
}
