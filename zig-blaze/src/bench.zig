//! Blaze-Zig Matrix Multiplication Benchmark
//! Single-threaded MKL-backed matrix multiplication

const std = @import("std");
const blaze = @import("blaze");
const DynamicMatrix = blaze.DynamicMatrix;

const Allocator = std.mem.Allocator;

/// High-resolution timer for benchmarking
const Timer = struct {
    start_time: i128,

    pub fn start() Timer {
        return .{ .start_time = std.time.nanoTimestamp() };
    }

    pub fn elapsedMs(self: Timer) f64 {
        const end_time = std.time.nanoTimestamp();
        const elapsed_ns = end_time - self.start_time;
        return @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    }
};

/// Benchmark a function with warmup and timed runs
fn benchmark(
    comptime func: anytype,
    args: anytype,
    warmup_runs: usize,
    timed_runs: usize,
) f64 {
    // Warmup
    for (0..warmup_runs) |_| {
        _ = @call(.auto, func, args);
    }

    // Timed runs
    const timer = Timer.start();
    for (0..timed_runs) |_| {
        _ = @call(.auto, func, args);
    }
    const total_ms = timer.elapsedMs();

    return total_ms / @as(f64, @floatFromInt(timed_runs));
}

/// Calculate GFLOPS for matrix multiplication
fn calculateGflops(M: usize, N: usize, K: usize, time_ms: f64) f64 {
    // Matrix multiplication: C(MxN) = A(MxK) * B(KxN)
    // Operations: 2 * M * N * K (multiply + add)
    const ops: f64 = 2.0 * @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(K));
    const seconds = time_ms / 1000.0;
    return (ops / seconds) / 1e9;
}

/// Run matrix multiplication benchmark for a given type and size
fn runBenchmark(
    comptime T: type,
    allocator: Allocator,
    size: usize,
    writer: anytype,
) !void {
    const Matrix = DynamicMatrix(T, .RowMajor);

    // Create matrices
    var A = try Matrix.init(allocator, size, size);
    defer A.deinit();
    var B = try Matrix.init(allocator, size, size);
    defer B.deinit();

    // Initialize with pseudo-random values (same seed as C++)
    A.fillRandom(42);
    B.fillRandom(43);

    // Pre-allocate result matrix for in-place multiplication
    var C = try Matrix.initWith(allocator, size, size, 0);
    defer C.deinit();

    // Warmup runs
    const warmup_runs: usize = 3;
    for (0..warmup_runs) |_| {
        try C.multiplyInto(&A, &B);
    }

    // Timed runs
    const timed_runs: usize = 10;
    const timer = Timer.start();
    for (0..timed_runs) |_| {
        try C.multiplyInto(&A, &B);
    }
    const total_ms = timer.elapsedMs();
    const avg_ms = total_ms / @as(f64, @floatFromInt(timed_runs));

    const gflops = calculateGflops(size, size, size, avg_ms);

    // Calculate memory usage: 3 matrices * size * size * sizeof(T)
    const memory_bytes: f64 = @as(f64, @floatFromInt(3 * size * size * @sizeOf(T)));
    const memory_mb = memory_bytes / (1024.0 * 1024.0);

    // Compute checksum (trace of first 10x10)
    var checksum: T = 0;
    const check_size = @min(size, 10);
    for (0..check_size) |i| {
        checksum += C.get(i, i);
    }

    // Print tabular row
    try writer.print("{d:<8} {d:>10.3} {d:>10.1} {d:>13.1} {d:>12.3}\n", .{ size, avg_ms, gflops, memory_mb, checksum });
}

pub fn main() !void {
    // Use a page allocator for large allocations
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Zig 0.15 I/O: use buffered stdout with explicit flush
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    try stdout.print("========================================\n", .{});
    try stdout.print("   Blaze-Zig Matrix Multiplication\n", .{});
    try stdout.print("   Single-threaded MKL Benchmark\n", .{});
    try stdout.print("========================================\n", .{});

    // Test different matrix sizes
    const sizes = [_]usize{ 256, 1024, 4096 };

    // Double precision tests
    try stdout.print("\n=== Double Precision (f64) ===\n", .{});
    try stdout.print("{s:<8} {s:>10} {s:>10} {s:>13} {s:>12}\n", .{ "Size", "Time(ms)", "GFLOPS", "Memory(MB)", "Checksum" });
    for (sizes) |size| {
        try runBenchmark(f64, allocator, size, stdout);
    }

    // Single precision tests
    try stdout.print("\n=== Single Precision (f32) ===\n", .{});
    try stdout.print("{s:<8} {s:>10} {s:>10} {s:>13} {s:>12}\n", .{ "Size", "Time(ms)", "GFLOPS", "Memory(MB)", "Checksum" });
    for (sizes) |size| {
        try runBenchmark(f32, allocator, size, stdout);
    }
}
