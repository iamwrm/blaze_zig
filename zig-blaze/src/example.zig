//! Simple example demonstrating Blaze-Zig matrix operations

const std = @import("std");
const blaze = @import("blaze");
const DynamicMatrix = blaze.DynamicMatrix;

fn printMatrix(mat: anytype) void {
    const print = std.debug.print;
    print("Matrix({}x{}):\n", .{ mat.rows, mat.cols });
    for (0..mat.rows) |i| {
        print("  [", .{});
        for (0..mat.cols) |j| {
            if (j > 0) print(", ", .{});
            print("{d:.4}", .{mat.get(i, j)});
        }
        print("]\n", .{});
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const print = std.debug.print;

    print("=== Blaze-Zig Example ===\n\n", .{});

    // Create a 3x3 matrix
    const Mat = DynamicMatrix(f64, .RowMajor);
    var A = try Mat.init(allocator, 3, 3);
    defer A.deinit();
    @memcpy(A.data, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });

    print("Matrix A:\n", .{});
    printMatrix(&A);

    // Create diagonal matrix B
    var B = try Mat.init(allocator, 3, 3);
    defer B.deinit();
    @memcpy(B.data, &[_]f64{ 1, 0, 0, 0, 2, 0, 0, 0, 3 });

    print("\nMatrix B (diagonal):\n", .{});
    printMatrix(&B);

    // Matrix multiplication using MKL
    var C = try Mat.multiply(allocator, &A, &B);
    defer C.deinit();

    print("\nC = A * B:\n", .{});
    printMatrix(&C);

    // Matrix addition
    var D = try Mat.add(allocator, &A, &C);
    defer D.deinit();

    print("\nD = A + C:\n", .{});
    printMatrix(&D);

    // Scalar multiplication
    var E = try Mat.scale(allocator, &A, 0.5);
    defer E.deinit();

    print("\nE = 0.5 * A:\n", .{});
    printMatrix(&E);

    // Trace
    print("\nTrace of A: {d:.4}\n", .{A.trace()});
    print("Trace of C: {d:.4}\n", .{C.trace()});

    print("\n=== Example Complete ===\n", .{});
}
