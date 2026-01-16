//! Dynamic matrix implementation for Blaze-Zig

const std = @import("std");
const Allocator = std.mem.Allocator;
const cblas = @import("cblas.zig");

/// Check if BLAS is available at link time
pub const use_blas = @hasDecl(cblas, "dgemm");

/// A dynamically-sized, row-major matrix
pub fn DynamicMatrix(comptime T: type) type {
    return struct {
        const Self = @This();

        data: []T,
        rows_count: usize,
        cols_count: usize,
        allocator: Allocator,

        /// Create a new matrix with the given dimensions
        pub fn init(allocator: Allocator, num_rows: usize, num_cols: usize) !Self {
            const data = try allocator.alloc(T, num_rows * num_cols);
            @memset(data, 0);
            return Self{
                .data = data,
                .rows_count = num_rows,
                .cols_count = num_cols,
                .allocator = allocator,
            };
        }

        /// Create a matrix from a 2D array
        pub fn fromSlice(allocator: Allocator, comptime num_rows: usize, comptime num_cols: usize, values: [num_rows][num_cols]T) !Self {
            var self = try init(allocator, num_rows, num_cols);
            for (0..num_rows) |i| {
                for (0..num_cols) |j| {
                    self.set(i, j, values[i][j]);
                }
            }
            return self;
        }

        /// Free the matrix memory
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.data = &.{};
            self.rows_count = 0;
            self.cols_count = 0;
        }

        /// Get the number of rows
        pub fn rows(self: Self) usize {
            return self.rows_count;
        }

        /// Get the number of columns
        pub fn columns(self: Self) usize {
            return self.cols_count;
        }

        /// Get element at (row, col)
        pub fn get(self: Self, row: usize, col: usize) T {
            return self.data[row * self.cols_count + col];
        }

        /// Set element at (row, col)
        pub fn set(self: *Self, row: usize, col: usize, value: T) void {
            self.data[row * self.cols_count + col] = value;
        }

        /// Get a pointer to element at (row, col) for modification
        pub fn ptr(self: *Self, row: usize, col: usize) *T {
            return &self.data[row * self.cols_count + col];
        }

        /// Fill matrix with random values between 0 and 1
        pub fn fillRandom(self: *Self, seed: u64) void {
            var rng = std.Random.DefaultPrng.init(seed);
            for (self.data) |*val| {
                val.* = rng.random().float(T);
            }
        }

        /// Matrix multiplication: C = A * B
        /// Uses BLAS (MKL/OpenBLAS) when available, otherwise falls back to tiled algorithm
        pub fn multiply(allocator: Allocator, a: Self, b: Self) !Self {
            if (a.cols_count != b.rows_count) {
                return error.DimensionMismatch;
            }

            var result = try init(allocator, a.rows_count, b.cols_count);

            const m = a.rows_count;
            const n = b.cols_count;
            const k = a.cols_count;

            if (T == f64) {
                // Use BLAS dgemm for double precision
                cblas.dgemm(m, n, k, a.data, b.data, result.data);
            } else if (T == f32) {
                // Use BLAS sgemm for single precision
                cblas.sgemm(m, n, k, a.data, b.data, result.data);
            } else {
                // Fallback to pure Zig implementation for other types
                multiplyPureZig(&result, a, b);
            }

            return result;
        }

        /// Pure Zig matrix multiplication (fallback when BLAS not available or for non-float types)
        fn multiplyPureZig(result: *Self, a: Self, b: Self) void {
            const tile_size: usize = 32;
            const m = a.rows_count;
            const n = b.cols_count;
            const k = a.cols_count;

            // Tiled matrix multiplication
            var ii: usize = 0;
            while (ii < m) : (ii += tile_size) {
                const i_end = @min(ii + tile_size, m);
                var kk: usize = 0;
                while (kk < k) : (kk += tile_size) {
                    const k_end = @min(kk + tile_size, k);
                    var jj: usize = 0;
                    while (jj < n) : (jj += tile_size) {
                        const j_end = @min(jj + tile_size, n);

                        // Inner tile computation
                        for (ii..i_end) |i| {
                            for (kk..k_end) |ki| {
                                const a_ik = a.get(i, ki);
                                for (jj..j_end) |j| {
                                    result.ptr(i, j).* += a_ik * b.get(ki, j);
                                }
                            }
                        }
                    }
                }
            }
        }

        /// Element-wise addition: C = A + B
        pub fn add(allocator: Allocator, a: Self, b: Self) !Self {
            if (a.rows_count != b.rows_count or a.cols_count != b.cols_count) {
                return error.DimensionMismatch;
            }

            var result = try init(allocator, a.rows_count, a.cols_count);
            for (0..a.data.len) |i| {
                result.data[i] = a.data[i] + b.data[i];
            }
            return result;
        }

        /// Element-wise subtraction: C = A - B
        pub fn subtract(allocator: Allocator, a: Self, b: Self) !Self {
            if (a.rows_count != b.rows_count or a.cols_count != b.cols_count) {
                return error.DimensionMismatch;
            }

            var result = try init(allocator, a.rows_count, a.cols_count);
            for (0..a.data.len) |i| {
                result.data[i] = a.data[i] - b.data[i];
            }
            return result;
        }

        /// Scalar multiplication: B = A * scalar
        pub fn scale(allocator: Allocator, a: Self, scalar: T) !Self {
            var result = try init(allocator, a.rows_count, a.cols_count);
            for (0..a.data.len) |i| {
                result.data[i] = a.data[i] * scalar;
            }
            return result;
        }

        /// In-place scalar multiplication
        pub fn scaleInPlace(self: *Self, scalar: T) void {
            for (self.data) |*val| {
                val.* *= scalar;
            }
        }

        /// Print matrix to stdout
        pub fn print(self: Self, writer: anytype) !void {
            for (0..self.rows_count) |i| {
                for (0..self.cols_count) |j| {
                    try writer.print("{d:10.2}", .{self.get(i, j)});
                }
                try writer.print("\n", .{});
            }
        }
    };
}

test "matrix creation and access" {
    const allocator = std.testing.allocator;

    var mat = try DynamicMatrix(f64).init(allocator, 3, 3);
    defer mat.deinit();

    mat.set(0, 0, 1.0);
    mat.set(1, 1, 2.0);
    mat.set(2, 2, 3.0);

    try std.testing.expectEqual(@as(f64, 1.0), mat.get(0, 0));
    try std.testing.expectEqual(@as(f64, 2.0), mat.get(1, 1));
    try std.testing.expectEqual(@as(f64, 3.0), mat.get(2, 2));
}

test "matrix from slice" {
    const allocator = std.testing.allocator;

    var mat = try DynamicMatrix(f64).fromSlice(allocator, 2, 3, .{
        .{ 1.0, 2.0, 3.0 },
        .{ 4.0, 5.0, 6.0 },
    });
    defer mat.deinit();

    try std.testing.expectEqual(@as(usize, 2), mat.rows());
    try std.testing.expectEqual(@as(usize, 3), mat.columns());
    try std.testing.expectEqual(@as(f64, 1.0), mat.get(0, 0));
    try std.testing.expectEqual(@as(f64, 6.0), mat.get(1, 2));
}

test "matrix multiplication" {
    const allocator = std.testing.allocator;

    // A = [1 2 3; 4 5 6] (2x3)
    var a = try DynamicMatrix(f64).fromSlice(allocator, 2, 3, .{
        .{ 1.0, 2.0, 3.0 },
        .{ 4.0, 5.0, 6.0 },
    });
    defer a.deinit();

    // B = [7 8; 9 10; 11 12] (3x2)
    var b = try DynamicMatrix(f64).fromSlice(allocator, 3, 2, .{
        .{ 7.0, 8.0 },
        .{ 9.0, 10.0 },
        .{ 11.0, 12.0 },
    });
    defer b.deinit();

    var c = try DynamicMatrix(f64).multiply(allocator, a, b);
    defer c.deinit();

    // C should be [58 64; 139 154]
    try std.testing.expectEqual(@as(usize, 2), c.rows());
    try std.testing.expectEqual(@as(usize, 2), c.columns());
    try std.testing.expectEqual(@as(f64, 58.0), c.get(0, 0));
    try std.testing.expectEqual(@as(f64, 64.0), c.get(0, 1));
    try std.testing.expectEqual(@as(f64, 139.0), c.get(1, 0));
    try std.testing.expectEqual(@as(f64, 154.0), c.get(1, 1));
}

test "matrix addition" {
    const allocator = std.testing.allocator;

    var a = try DynamicMatrix(f64).fromSlice(allocator, 2, 2, .{
        .{ 1.0, 2.0 },
        .{ 3.0, 4.0 },
    });
    defer a.deinit();

    var b = try DynamicMatrix(f64).fromSlice(allocator, 2, 2, .{
        .{ 5.0, 6.0 },
        .{ 7.0, 8.0 },
    });
    defer b.deinit();

    var c = try DynamicMatrix(f64).add(allocator, a, b);
    defer c.deinit();

    try std.testing.expectEqual(@as(f64, 6.0), c.get(0, 0));
    try std.testing.expectEqual(@as(f64, 8.0), c.get(0, 1));
    try std.testing.expectEqual(@as(f64, 10.0), c.get(1, 0));
    try std.testing.expectEqual(@as(f64, 12.0), c.get(1, 1));
}

test "scalar multiplication" {
    const allocator = std.testing.allocator;

    var a = try DynamicMatrix(f64).fromSlice(allocator, 2, 2, .{
        .{ 1.0, 2.0 },
        .{ 3.0, 4.0 },
    });
    defer a.deinit();

    var b = try DynamicMatrix(f64).scale(allocator, a, 2.0);
    defer b.deinit();

    try std.testing.expectEqual(@as(f64, 2.0), b.get(0, 0));
    try std.testing.expectEqual(@as(f64, 4.0), b.get(0, 1));
    try std.testing.expectEqual(@as(f64, 6.0), b.get(1, 0));
    try std.testing.expectEqual(@as(f64, 8.0), b.get(1, 1));
}
