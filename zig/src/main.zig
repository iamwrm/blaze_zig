//! Blaze-Zig Matrix Operations Example

const std = @import("std");
const blaze = @import("blaze");

const DynamicMatrix = blaze.DynamicMatrix(f64);
const DynamicVector = blaze.DynamicVector(f64);

pub fn main() !void {
    const stdout_file = std.fs.File.stdout();
    var stdout_buffer: [4096]u8 = undefined;
    const stdout = stdout_file.writer(&stdout_buffer);
    defer stdout.flush() catch {};

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try stdout.print("Blaze-Zig Matrix Operations Example\n", .{});
    try stdout.print("====================================\n\n", .{});

    // Create matrices
    var a = try DynamicMatrix.fromSlice(allocator, 2, 3, .{
        .{ 1.0, 2.0, 3.0 },
        .{ 4.0, 5.0, 6.0 },
    });
    defer a.deinit();

    var b = try DynamicMatrix.fromSlice(allocator, 3, 2, .{
        .{ 7.0, 8.0 },
        .{ 9.0, 10.0 },
        .{ 11.0, 12.0 },
    });
    defer b.deinit();

    // Matrix multiplication
    var c = try DynamicMatrix.multiply(allocator, a, b);
    defer c.deinit();

    try stdout.print("Matrix A (2x3):\n", .{});
    try a.print(stdout);

    try stdout.print("\nMatrix B (3x2):\n", .{});
    try b.print(stdout);

    try stdout.print("\nC = A * B (2x2):\n", .{});
    try c.print(stdout);

    // Vector operations
    var v1 = try DynamicVector.fromArray(allocator, 3, .{ 1.0, 2.0, 3.0 });
    defer v1.deinit();

    var v2 = try DynamicVector.fromArray(allocator, 3, .{ 4.0, 5.0, 6.0 });
    defer v2.deinit();

    const dot_product = try DynamicVector.inner(v1, v2);
    try stdout.print("\nVector v1: ", .{});
    try v1.print(stdout);
    try stdout.print("Vector v2: ", .{});
    try v2.print(stdout);
    try stdout.print("Dot product v1 . v2 = {d:.2}\n", .{dot_product});

    // Element-wise operations
    var v3 = try DynamicVector.add(allocator, v1, v2);
    defer v3.deinit();
    try stdout.print("v1 + v2 = ", .{});
    try v3.print(stdout);

    var v4 = try DynamicVector.scale(allocator, v1, 2.0);
    defer v4.deinit();
    try stdout.print("v1 * 2 = ", .{});
    try v4.print(stdout);

    try stdout.print("\nBlaze-Zig example completed successfully!\n", .{});
}
