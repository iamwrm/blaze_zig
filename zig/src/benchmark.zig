//! Blaze-Zig Matrix Multiplication Benchmark

const std = @import("std");
const blaze = @import("blaze");

const DynamicMatrix = blaze.DynamicMatrix(f64);

const BenchmarkResult = struct {
    size: usize,
    time_ms: f64,
    gflops: f64,
};

fn benchmarkMatmul(allocator: std.mem.Allocator, n: usize, warmup_runs: u32, timed_runs: u32) !BenchmarkResult {
    // Create matrices
    var a = try DynamicMatrix.init(allocator, n, n);
    defer a.deinit();
    a.fillRandom(42);

    var b = try DynamicMatrix.init(allocator, n, n);
    defer b.deinit();
    b.fillRandom(43);

    // Warmup runs
    for (0..warmup_runs) |_| {
        var c = try DynamicMatrix.multiply(allocator, a, b);
        c.deinit();
    }

    // Timed runs
    var total_time_ns: u64 = 0;
    var timer = try std.time.Timer.start();

    for (0..timed_runs) |_| {
        timer.reset();
        var c = try DynamicMatrix.multiply(allocator, a, b);
        total_time_ns += timer.read();
        c.deinit();
    }

    const avg_time_ms = @as(f64, @floatFromInt(total_time_ns)) / @as(f64, @floatFromInt(timed_runs)) / 1_000_000.0;

    // Calculate GFLOPS: 2*N^3 operations for matrix multiply
    const n_f: f64 = @floatFromInt(n);
    const flops = 2.0 * n_f * n_f * n_f;
    const gflops = (flops / (avg_time_ms / 1000.0)) / 1e9;

    return BenchmarkResult{
        .size = n,
        .time_ms = avg_time_ms,
        .gflops = gflops,
    };
}

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    // Use page allocator for large allocations (better performance)
    const allocator = std.heap.page_allocator;

    try stdout.print("Blaze-Zig Matrix Multiplication Benchmark\n", .{});
    try stdout.print("==========================================\n\n", .{});

    try stdout.print("Mode: Single-threaded (pure Zig)\n\n", .{});

    const sizes = [_]usize{ 64, 128, 256, 512, 1024, 2048 };

    try stdout.print("{s:>10}{s:>15}{s:>15}\n", .{ "Size", "Time (ms)", "GFLOPS" });
    try stdout.print("{s:-<40}\n", .{""});

    for (sizes) |size| {
        const result = try benchmarkMatmul(allocator, size, 2, 5);
        try stdout.print("{d:>10}{d:>15.3}{d:>15.2}\n", .{ result.size, result.time_ms, result.gflops });
    }

    try stdout.print("\nBenchmark completed!\n", .{});
}
