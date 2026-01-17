//! Simple example demonstrating Blaze-Zig matrix operations

const std = @import("std");
const blaze = @import("blaze");
const DynamicMatrix = blaze.DynamicMatrix;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Zig 0.15 I/O: use buffered stdout with explicit flush
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    try stdout.print("=== Blaze-Zig Example ===\n\n", .{});

    // Create a 3x3 matrix
    const Mat = DynamicMatrix(f64, .RowMajor);
    var A = try Mat.init(allocator, 3, 3);
    defer A.deinit();
    @memcpy(A.data, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });

    try stdout.print("Matrix A:\n", .{});
    try A.print(stdout);

    // Create diagonal matrix B
    var B = try Mat.init(allocator, 3, 3);
    defer B.deinit();
    @memcpy(B.data, &[_]f64{ 1, 0, 0, 0, 2, 0, 0, 0, 3 });

    try stdout.print("\nMatrix B (diagonal):\n", .{});
    try B.print(stdout);

    // Matrix multiplication using MKL
    var C = try Mat.multiply(allocator, &A, &B);
    defer C.deinit();

    try stdout.print("\nC = A * B:\n", .{});
    try C.print(stdout);

    // Matrix addition
    var D = try Mat.add(allocator, &A, &C);
    defer D.deinit();

    try stdout.print("\nD = A + C:\n", .{});
    try D.print(stdout);

    // Scalar multiplication
    var E = try Mat.scale(allocator, &A, 0.5);
    defer E.deinit();

    try stdout.print("\nE = 0.5 * A:\n", .{});
    try E.print(stdout);

    // Trace
    try stdout.print("\nTrace of A: {d:.4}\n", .{A.trace()});
    try stdout.print("Trace of C: {d:.4}\n", .{C.trace()});

    try stdout.print("\n=== Example Complete ===\n", .{});
}
