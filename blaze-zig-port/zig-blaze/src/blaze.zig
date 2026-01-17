//! Blaze-Zig: A minimal port of the Blaze C++ library to Zig
//! 
//! This library provides high-performance dense matrix operations
//! backed by Intel MKL BLAS.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// CBLAS bindings for MKL
pub const cblas = @cImport({
    @cInclude("mkl_cblas.h");
});

/// Matrix storage order
pub const StorageOrder = enum {
    RowMajor,
    ColumnMajor,
};

/// Dense dynamic matrix with configurable element type and storage order
pub fn DynamicMatrix(comptime T: type, comptime order: StorageOrder) type {
    return struct {
        const Self = @This();

        /// Underlying data storage
        data: []T,
        /// Number of rows
        rows: usize,
        /// Number of columns
        cols: usize,
        /// Allocator used for memory management
        allocator: Allocator,

        /// Create a new matrix with uninitialized values
        pub fn init(allocator: Allocator, rows: usize, cols: usize) !Self {
            const data = try allocator.alloc(T, rows * cols);
            return Self{
                .data = data,
                .rows = rows,
                .cols = cols,
                .allocator = allocator,
            };
        }

        /// Create a new matrix initialized to a specific value
        pub fn initWith(allocator: Allocator, rows: usize, cols: usize, value: T) !Self {
            const self = try init(allocator, rows, cols);
            @memset(self.data, value);
            return self;
        }

        /// Free the matrix memory
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.* = undefined;
        }

        /// Get element at (row, col)
        pub inline fn get(self: *const Self, row: usize, col: usize) T {
            return self.data[self.index(row, col)];
        }

        /// Set element at (row, col)
        pub inline fn set(self: *Self, row: usize, col: usize, value: T) void {
            self.data[self.index(row, col)] = value;
        }

        /// Get mutable pointer to element at (row, col)
        pub inline fn ptr(self: *Self, row: usize, col: usize) *T {
            return &self.data[self.index(row, col)];
        }

        /// Calculate linear index from (row, col)
        inline fn index(self: *const Self, row: usize, col: usize) usize {
            return switch (order) {
                .RowMajor => row * self.cols + col,
                .ColumnMajor => col * self.rows + row,
            };
        }

        /// Get leading dimension for BLAS calls
        pub inline fn leadingDimension(self: *const Self) usize {
            return switch (order) {
                .RowMajor => self.cols,
                .ColumnMajor => self.rows,
            };
        }

        /// Fill matrix with a value
        pub fn fill(self: *Self, value: T) void {
            @memset(self.data, value);
        }

        /// Fill matrix with random values in [0, 1)
        pub fn fillRandom(self: *Self, seed: u64) void {
            var prng = std.rand.DefaultPrng.init(seed);
            const random = prng.random();
            for (self.data) |*elem| {
                if (T == f32 or T == f64) {
                    elem.* = random.float(T);
                } else {
                    elem.* = @intCast(random.int(u32) % 100);
                }
            }
        }

        /// Compute trace (sum of diagonal elements)
        pub fn trace(self: *const Self) T {
            const n = @min(self.rows, self.cols);
            var sum: T = 0;
            for (0..n) |i| {
                sum += self.get(i, i);
            }
            return sum;
        }

        /// Clone the matrix
        pub fn clone(self: *const Self, allocator: Allocator) !Self {
            const new_matrix = try init(allocator, self.rows, self.cols);
            @memcpy(new_matrix.data, self.data);
            return new_matrix;
        }

        /// Matrix-matrix multiplication: C = A * B
        /// Uses MKL CBLAS for high performance
        pub fn multiply(allocator: Allocator, A: *const Self, B: *const Self) !Self {
            if (A.cols != B.rows) {
                return error.DimensionMismatch;
            }

            var C = try initWith(allocator, A.rows, B.cols, 0);

            const layout = switch (order) {
                .RowMajor => cblas.CblasRowMajor,
                .ColumnMajor => cblas.CblasColMajor,
            };

            if (T == f64) {
                cblas.cblas_dgemm(
                    layout,
                    cblas.CblasNoTrans,
                    cblas.CblasNoTrans,
                    @intCast(A.rows), // M
                    @intCast(B.cols), // N
                    @intCast(A.cols), // K
                    1.0, // alpha
                    A.data.ptr,
                    @intCast(A.leadingDimension()),
                    B.data.ptr,
                    @intCast(B.leadingDimension()),
                    0.0, // beta
                    C.data.ptr,
                    @intCast(C.leadingDimension()),
                );
            } else if (T == f32) {
                cblas.cblas_sgemm(
                    layout,
                    cblas.CblasNoTrans,
                    cblas.CblasNoTrans,
                    @intCast(A.rows), // M
                    @intCast(B.cols), // N
                    @intCast(A.cols), // K
                    1.0, // alpha
                    A.data.ptr,
                    @intCast(A.leadingDimension()),
                    B.data.ptr,
                    @intCast(B.leadingDimension()),
                    0.0, // beta
                    C.data.ptr,
                    @intCast(C.leadingDimension()),
                );
            } else {
                @compileError("MKL BLAS only supports f32 and f64");
            }

            return C;
        }

        /// In-place matrix-matrix multiplication: self = A * B
        pub fn multiplyInto(self: *Self, A: *const Self, B: *const Self) !void {
            if (A.cols != B.rows) {
                return error.DimensionMismatch;
            }
            if (self.rows != A.rows or self.cols != B.cols) {
                return error.DimensionMismatch;
            }

            const layout = switch (order) {
                .RowMajor => cblas.CblasRowMajor,
                .ColumnMajor => cblas.CblasColMajor,
            };

            if (T == f64) {
                cblas.cblas_dgemm(
                    layout,
                    cblas.CblasNoTrans,
                    cblas.CblasNoTrans,
                    @intCast(A.rows),
                    @intCast(B.cols),
                    @intCast(A.cols),
                    1.0,
                    A.data.ptr,
                    @intCast(A.leadingDimension()),
                    B.data.ptr,
                    @intCast(B.leadingDimension()),
                    0.0,
                    self.data.ptr,
                    @intCast(self.leadingDimension()),
                );
            } else if (T == f32) {
                cblas.cblas_sgemm(
                    layout,
                    cblas.CblasNoTrans,
                    cblas.CblasNoTrans,
                    @intCast(A.rows),
                    @intCast(B.cols),
                    @intCast(A.cols),
                    1.0,
                    A.data.ptr,
                    @intCast(A.leadingDimension()),
                    B.data.ptr,
                    @intCast(B.leadingDimension()),
                    0.0,
                    self.data.ptr,
                    @intCast(self.leadingDimension()),
                );
            } else {
                @compileError("MKL BLAS only supports f32 and f64");
            }
        }

        /// Element-wise addition: C = A + B
        pub fn add(allocator: Allocator, A: *const Self, B: *const Self) !Self {
            if (A.rows != B.rows or A.cols != B.cols) {
                return error.DimensionMismatch;
            }

            const C = try init(allocator, A.rows, A.cols);
            for (C.data, A.data, B.data) |*c, a, b| {
                c.* = a + b;
            }
            return C;
        }

        /// Element-wise subtraction: C = A - B
        pub fn subtract(allocator: Allocator, A: *const Self, B: *const Self) !Self {
            if (A.rows != B.rows or A.cols != B.cols) {
                return error.DimensionMismatch;
            }

            const C = try init(allocator, A.rows, A.cols);
            for (C.data, A.data, B.data) |*c, a, b| {
                c.* = a - b;
            }
            return C;
        }

        /// Scalar multiplication: B = alpha * A
        pub fn scale(allocator: Allocator, A: *const Self, alpha: T) !Self {
            const B = try init(allocator, A.rows, A.cols);
            for (B.data, A.data) |*b_elem, a| {
                b_elem.* = alpha * a;
            }
            return B;
        }

        /// Print matrix (for debugging)
        pub fn print(self: *const Self, writer: anytype) !void {
            try writer.print("Matrix({}x{}):\n", .{ self.rows, self.cols });
            for (0..self.rows) |i| {
                try writer.print("  [", .{});
                for (0..self.cols) |j| {
                    if (j > 0) try writer.print(", ", .{});
                    try writer.print("{d:.4}", .{self.get(i, j)});
                }
                try writer.print("]\n", .{});
            }
        }
    };
}

/// Type aliases for common matrix types (mimicking Blaze's naming)
pub const DynamicMatrixF64 = DynamicMatrix(f64, .RowMajor);
pub const DynamicMatrixF32 = DynamicMatrix(f32, .RowMajor);

/// Column-major variants
pub const DynamicMatrixF64ColMajor = DynamicMatrix(f64, .ColumnMajor);
pub const DynamicMatrixF32ColMajor = DynamicMatrix(f32, .ColumnMajor);

// =============================================================================
// Unit Tests
// =============================================================================

test "DynamicMatrix basic operations" {
    const allocator = std.testing.allocator;

    var A = try DynamicMatrixF64.initWith(allocator, 2, 2, 0);
    defer A.deinit();

    A.set(0, 0, 1.0);
    A.set(0, 1, 2.0);
    A.set(1, 0, 3.0);
    A.set(1, 1, 4.0);

    try std.testing.expectEqual(@as(f64, 1.0), A.get(0, 0));
    try std.testing.expectEqual(@as(f64, 2.0), A.get(0, 1));
    try std.testing.expectEqual(@as(f64, 3.0), A.get(1, 0));
    try std.testing.expectEqual(@as(f64, 4.0), A.get(1, 1));
}

test "DynamicMatrix trace" {
    const allocator = std.testing.allocator;

    var A = try DynamicMatrixF64.initWith(allocator, 3, 3, 0);
    defer A.deinit();

    A.set(0, 0, 1.0);
    A.set(1, 1, 2.0);
    A.set(2, 2, 3.0);

    try std.testing.expectEqual(@as(f64, 6.0), A.trace());
}
