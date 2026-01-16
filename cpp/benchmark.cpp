#include <blaze/Blaze.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cstdlib>

// Force single-threaded MKL via environment variable
struct MKLSingleThread {
    MKLSingleThread() {
        // Set MKL to single-threaded mode via environment
        setenv("MKL_NUM_THREADS", "1", 1);
        setenv("OMP_NUM_THREADS", "1", 1);
    }
};

static MKLSingleThread mkl_init;

// Simple timing helper
class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// Fill matrix with random values
void fill_random(blaze::DynamicMatrix<double>& mat) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.columns(); ++j) {
            mat(i, j) = dist(gen);
        }
    }
}

struct BenchmarkResult {
    size_t size;
    double time_ms;
    double gflops;
};

BenchmarkResult benchmark_matmul(size_t N, int warmup_runs = 2, int timed_runs = 5) {
    // Create matrices
    blaze::DynamicMatrix<double> A(N, N);
    blaze::DynamicMatrix<double> B(N, N);
    blaze::DynamicMatrix<double> C(N, N);

    fill_random(A);
    fill_random(B);

    Timer timer;

    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
        C = A * B;
    }

    // Timed runs
    double total_time = 0.0;
    for (int i = 0; i < timed_runs; ++i) {
        timer.start();
        C = A * B;
        total_time += timer.elapsed_ms();
    }

    double avg_time_ms = total_time / timed_runs;

    // Calculate GFLOPS: 2*N^3 operations for matrix multiply
    double flops = 2.0 * N * N * N;
    double gflops = (flops / (avg_time_ms / 1000.0)) / 1e9;

    // Prevent optimization from removing the computation
    volatile double checksum = C(0, 0);
    (void)checksum;

    return {N, avg_time_ms, gflops};
}

int main() {
    std::cout << "Blaze C++ Matrix Multiplication Benchmark (Single-threaded MKL)\n";
    std::cout << "================================================================\n\n";

    // Note: MKL_NUM_THREADS=1 and OMP_NUM_THREADS=1 are set in MKLSingleThread
    std::cout << "Mode: Single-threaded (MKL_NUM_THREADS=1)\n\n";

    std::vector<size_t> sizes = {64, 128, 256, 512, 1024, 2048};
    std::vector<BenchmarkResult> results;

    std::cout << std::setw(10) << "Size"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "GFLOPS"
              << "\n";
    std::cout << std::string(40, '-') << "\n";

    for (size_t size : sizes) {
        auto result = benchmark_matmul(size);
        results.push_back(result);

        std::cout << std::setw(10) << result.size
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.gflops
                  << "\n";
    }

    std::cout << "\nBenchmark completed!\n";
    return 0;
}
