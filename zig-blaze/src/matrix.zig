//! Fixed-size Matrix Type with Expression Support
//!
//! This module provides a compile-time sized matrix type that integrates
//! with the expression system for lazy evaluation and BLAS/SIMD optimization.

const std = @import("std");
const cblas = @import("blas/cblas.zig");
const simd = @import("kernels/simd.zig");
const vec = @import("vector.zig");

/// Size threshold for using BLAS over SIMD
const BLAS_THRESHOLD: usize = 64;

/// Fixed-size matrix with compile-time known dimensions
pub fn Matrix(comptime T: type, comptime Rows: usize, comptime Cols: usize) type {
    return struct {
        const Self = @This();
        pub const ElementType = T;
        pub const rows = Rows;
        pub const cols = Cols;
        pub const is_vector = false;
        pub const is_matrix = true;

        // SIMD configuration
        const VecLen = simd.vecLen(T);
        pub const SimdVec = @Vector(VecLen, T);

        data: *align(64) [Rows * Cols]T,
        allocator: ?std.mem.Allocator = null,

        // ─────────────────────────────────────────────────────────────────
        // Initialization
        // ─────────────────────────────────────────────────────────────────

        /// Create a new matrix with uninitialized data
        pub fn init(allocator: std.mem.Allocator) !Self {
            const data = try allocator.alignedAlloc(T, 64, Rows * Cols);
            return .{
                .data = @ptrCast(data.ptr),
                .allocator = allocator,
            };
        }

        /// Create a new matrix initialized to a specific value
        pub fn initWithValue(allocator: std.mem.Allocator, value: T) !Self {
            const result = try init(allocator);
            @memset(result.data, value);
            return result;
        }

        /// Create an identity matrix
        pub fn initIdentity(allocator: std.mem.Allocator) !Self {
            const result = try init(allocator);
            @memset(result.data, 0);
            const min_dim = @min(Rows, Cols);
            for (0..min_dim) |i| {
                result.data[i * Cols + i] = 1;
            }
            return result;
        }

        /// Create a matrix from an existing array (no allocation, does not own memory)
        pub fn fromArray(data: *align(64) [Rows * Cols]T) Self {
            return .{
                .data = data,
                .allocator = null,
            };
        }

        /// Free the matrix memory
        pub fn deinit(self: *Self) void {
            if (self.allocator) |alloc| {
                const mem_slice: []align(64) T = @as([*]align(64) T, @ptrCast(self.data))[0 .. Rows * Cols];
                alloc.free(mem_slice);
            }
            self.* = undefined;
        }

        // ─────────────────────────────────────────────────────────────────
        // Data Access
        // ─────────────────────────────────────────────────────────────────

        /// Get raw pointer for BLAS calls
        pub fn ptr(self: Self) [*]T {
            return @ptrCast(self.data);
        }

        /// Get const pointer for BLAS calls
        pub fn constPtr(self: Self) [*]const T {
            return @ptrCast(self.data);
        }

        /// Get element at (r, c)
        pub fn at(self: Self, r: usize, c: usize) T {
            return self.data[r * Cols + c];
        }

        /// Set element at (r, c)
        pub fn set(self: *Self, r: usize, c: usize, value: T) void {
            self.data[r * Cols + c] = value;
        }

        /// Get mutable pointer to element at (r, c)
        pub fn ptrAt(self: *Self, r: usize, c: usize) *T {
            return &self.data[r * Cols + c];
        }

        /// Get a slice of the underlying data
        pub fn slice(self: Self) []T {
            return self.data[0 .. Rows * Cols];
        }

        /// Get a row as a vector
        pub fn row(self: Self, r: usize) vec.Vector(T, Cols) {
            const row_data: *align(64) [Cols]T = @ptrCast(@alignCast(self.data[r * Cols ..].ptr));
            return vec.Vector(T, Cols).fromArray(row_data);
        }

        // ─────────────────────────────────────────────────────────────────
        // BLAS Level 2 Operations (Matrix-Vector)
        // ─────────────────────────────────────────────────────────────────

        /// Matrix-vector multiply: result = A*x
        pub fn mulVecInto(self: Self, x: vec.Vector(T, Cols), result: *vec.Vector(T, Rows)) void {
            if (comptime shouldUseBlas()) {
                cblas.gemv(
                    T,
                    .RowMajor,
                    .NoTrans,
                    @intCast(Rows),
                    @intCast(Cols),
                    1,
                    self.constPtr(),
                    @intCast(Cols),
                    x.constPtr(),
                    1,
                    0,
                    result.ptr(),
                    1,
                );
            } else {
                simd.gemv(T, Rows, Cols, 1, self.data, x.data, 0, result.data);
            }
        }

        /// Triangular matrix-vector multiply: result = A*x (upper triangular)
        pub fn trmvInto(self: Self, x: vec.Vector(T, Cols), result: *vec.Vector(T, Rows), uplo: cblas.UpLo, diag: cblas.Diag) void {
            @memcpy(result.data, x.data);
            cblas.trmv(
                T,
                .RowMajor,
                uplo,
                .NoTrans,
                diag,
                @intCast(Rows),
                self.constPtr(),
                @intCast(Cols),
                result.ptr(),
                1,
            );
        }

        /// Symmetric matrix-vector multiply: result = A*x
        pub fn symvInto(self: Self, x: vec.Vector(T, Cols), result: *vec.Vector(T, Rows), uplo: cblas.UpLo) void {
            cblas.symv(
                T,
                .RowMajor,
                uplo,
                @intCast(Rows),
                1,
                self.constPtr(),
                @intCast(Cols),
                x.constPtr(),
                1,
                0,
                result.ptr(),
                1,
            );
        }

        // ─────────────────────────────────────────────────────────────────
        // BLAS Level 3 Operations (Matrix-Matrix)
        // ─────────────────────────────────────────────────────────────────

        /// Matrix-matrix multiply: result = self * other
        pub fn mulInto(self: Self, other: Matrix(T, Cols, Cols), result: *Self) void {
            self.gemmInto(other, 1, 0, result);
        }

        /// General matrix multiply: result = α*self*other + β*result
        pub fn gemmInto(self: Self, other: anytype, alpha: T, beta: T, result: anytype) void {
            const OtherType = @TypeOf(other);
            const ResultType = @TypeOf(result.*);

            comptime {
                if (!OtherType.is_matrix) @compileError("other must be a matrix");
                if (!ResultType.is_matrix) @compileError("result must be a matrix");
                if (Cols != OtherType.rows) @compileError("Matrix dimension mismatch for multiplication");
                if (Rows != ResultType.rows or OtherType.cols != ResultType.cols) {
                    @compileError("Result matrix has wrong dimensions");
                }
            }

            const K = Cols;
            const N = OtherType.cols;

            if (comptime shouldUseBlas()) {
                cblas.gemm(
                    T,
                    .RowMajor,
                    .NoTrans,
                    .NoTrans,
                    @intCast(Rows),
                    @intCast(N),
                    @intCast(K),
                    alpha,
                    self.constPtr(),
                    @intCast(K),
                    other.constPtr(),
                    @intCast(N),
                    beta,
                    result.ptr(),
                    @intCast(N),
                );
            } else {
                simd.gemm(T, Rows, N, K, alpha, self.data, other.data, beta, result.data);
            }
        }

        /// Triangular matrix solve: result = A⁻¹*B
        pub fn trsmInto(self: Self, B: *Self, side: cblas.Side, uplo: cblas.UpLo, diag: cblas.Diag) void {
            cblas.trsm(
                T,
                .RowMajor,
                side,
                uplo,
                .NoTrans,
                diag,
                @intCast(Rows),
                @intCast(Cols),
                1,
                self.constPtr(),
                @intCast(Cols),
                B.ptr(),
                @intCast(Cols),
            );
        }

        // ─────────────────────────────────────────────────────────────────
        // Element-wise Operations
        // ─────────────────────────────────────────────────────────────────

        /// Element-wise addition: result = self + other
        pub fn addInto(self: Self, other: Self, result: *Self) void {
            simd.add(T, Rows * Cols, self.data, other.data, result.data);
        }

        /// Element-wise subtraction: result = self - other
        pub fn subInto(self: Self, other: Self, result: *Self) void {
            simd.sub(T, Rows * Cols, self.data, other.data, result.data);
        }

        /// Scale in-place: self = α*self
        pub fn scaleInPlace(self: *Self, alpha: T) void {
            if (comptime shouldUseBlas()) {
                cblas.scal(T, @intCast(Rows * Cols), alpha, self.ptr(), 1);
            } else {
                simd.scal(T, Rows * Cols, alpha, self.data);
            }
        }

        /// Scale into result: result = α*self
        pub fn scaleInto(self: Self, alpha: T, result: *Self) void {
            @memcpy(result.data, self.data);
            result.scaleInPlace(alpha);
        }

        // ─────────────────────────────────────────────────────────────────
        // Transpose
        // ─────────────────────────────────────────────────────────────────

        /// Transpose: result = selfᵀ
        pub fn transposeInto(self: Self, result: *Matrix(T, Cols, Rows)) void {
            simd.transpose(T, Rows, Cols, self.data, result.data);
        }

        // ─────────────────────────────────────────────────────────────────
        // Utility Functions
        // ─────────────────────────────────────────────────────────────────

        /// Fill matrix with a value
        pub fn fill(self: *Self, value: T) void {
            @memset(self.data, value);
        }

        /// Fill matrix with random values
        pub fn fillRandom(self: *Self, seed: u64) void {
            var prng = std.Random.DefaultPrng.init(seed);
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
        pub fn trace(self: Self) T {
            const n = @min(Rows, Cols);
            var sum: T = 0;
            for (0..n) |i| {
                sum += self.at(i, i);
            }
            return sum;
        }

        /// Frobenius norm: ||A||_F = sqrt(Σaᵢⱼ²)
        pub fn frobeniusNorm(self: Self) T {
            if (comptime shouldUseBlas()) {
                return cblas.nrm2(T, @intCast(Rows * Cols), self.constPtr(), 1);
            } else {
                return simd.nrm2(T, Rows * Cols, self.data);
            }
        }

        /// Clone the matrix
        pub fn clone(self: Self, allocator: std.mem.Allocator) !Self {
            const new_matrix = try init(allocator);
            @memcpy(new_matrix.data, self.data);
            return new_matrix;
        }

        /// Print matrix (for debugging)
        pub fn print(self: Self, writer: anytype) !void {
            try writer.print("Matrix({}x{}):\n", .{ Rows, Cols });
            for (0..Rows) |i| {
                try writer.print("  [", .{});
                for (0..Cols) |j| {
                    if (j > 0) try writer.print(", ", .{});
                    try writer.print("{d:.4}", .{self.at(i, j)});
                }
                try writer.print("]\n", .{});
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // Helper Functions
        // ─────────────────────────────────────────────────────────────────

        fn shouldUseBlas() bool {
            return cblas.isBLASCompatible(T) and Rows * Cols >= BLAS_THRESHOLD;
        }
    };
}

// =============================================================================
// Type Aliases
// =============================================================================

/// Common f64 matrix types
pub fn MatrixF64(comptime Rows: usize, comptime Cols: usize) type {
    return Matrix(f64, Rows, Cols);
}

/// Common f32 matrix types
pub fn MatrixF32(comptime Rows: usize, comptime Cols: usize) type {
    return Matrix(f32, Rows, Cols);
}

// =============================================================================
// Unit Tests
// =============================================================================

test "Matrix initialization and access" {
    const allocator = std.testing.allocator;

    var A = try Matrix(f64, 2, 2).initWithValue(allocator, 1.0);
    defer A.deinit();

    try std.testing.expectEqual(@as(f64, 1.0), A.at(0, 0));
    try std.testing.expectEqual(@as(f64, 1.0), A.at(1, 1));

    A.set(0, 1, 5.0);
    try std.testing.expectEqual(@as(f64, 5.0), A.at(0, 1));
}

test "Matrix identity" {
    const allocator = std.testing.allocator;

    var I = try Matrix(f64, 3, 3).initIdentity(allocator);
    defer I.deinit();

    try std.testing.expectEqual(@as(f64, 1.0), I.at(0, 0));
    try std.testing.expectEqual(@as(f64, 0.0), I.at(0, 1));
    try std.testing.expectEqual(@as(f64, 1.0), I.at(1, 1));
    try std.testing.expectEqual(@as(f64, 0.0), I.at(1, 0));
    try std.testing.expectEqual(@as(f64, 1.0), I.at(2, 2));
}

test "Matrix trace" {
    const allocator = std.testing.allocator;

    var A = try Matrix(f64, 3, 3).initWithValue(allocator, 0.0);
    defer A.deinit();

    A.set(0, 0, 1.0);
    A.set(1, 1, 2.0);
    A.set(2, 2, 3.0);

    try std.testing.expectEqual(@as(f64, 6.0), A.trace());
}

test "Matrix multiply" {
    const allocator = std.testing.allocator;

    // A = [[1, 2], [3, 4]]
    var A = try Matrix(f64, 2, 2).init(allocator);
    defer A.deinit();
    A.set(0, 0, 1.0);
    A.set(0, 1, 2.0);
    A.set(1, 0, 3.0);
    A.set(1, 1, 4.0);

    // I = identity
    var I = try Matrix(f64, 2, 2).initIdentity(allocator);
    defer I.deinit();

    var C = try Matrix(f64, 2, 2).init(allocator);
    defer C.deinit();

    // C = A * I = A
    A.mulInto(I, &C);

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), C.at(0, 0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), C.at(0, 1), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), C.at(1, 0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), C.at(1, 1), 1e-10);
}

test "Matrix-vector multiply" {
    const allocator = std.testing.allocator;

    // A = [[1, 2], [3, 4]]
    var A = try Matrix(f64, 2, 2).init(allocator);
    defer A.deinit();
    A.set(0, 0, 1.0);
    A.set(0, 1, 2.0);
    A.set(1, 0, 3.0);
    A.set(1, 1, 4.0);

    // x = [1, 2]
    var x = try vec.Vector(f64, 2).init(allocator);
    defer x.deinit();
    x.set(0, 1.0);
    x.set(1, 2.0);

    var y = try vec.Vector(f64, 2).init(allocator);
    defer y.deinit();

    // y = A*x = [1*1 + 2*2, 3*1 + 4*2] = [5, 11]
    A.mulVecInto(x, &y);

    try std.testing.expectApproxEqAbs(@as(f64, 5.0), y.at(0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 11.0), y.at(1), 1e-10);
}

test "Matrix add and sub" {
    const allocator = std.testing.allocator;

    var A = try Matrix(f64, 2, 2).initWithValue(allocator, 3.0);
    defer A.deinit();
    var B = try Matrix(f64, 2, 2).initWithValue(allocator, 1.0);
    defer B.deinit();
    var C = try Matrix(f64, 2, 2).init(allocator);
    defer C.deinit();

    A.addInto(B, &C);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), C.at(0, 0), 1e-10);

    A.subInto(B, &C);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), C.at(0, 0), 1e-10);
}

test "Matrix scale" {
    const allocator = std.testing.allocator;

    var A = try Matrix(f64, 2, 2).initWithValue(allocator, 2.0);
    defer A.deinit();

    A.scaleInPlace(3.0);

    try std.testing.expectApproxEqAbs(@as(f64, 6.0), A.at(0, 0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), A.at(1, 1), 1e-10);
}

test "Matrix transpose" {
    const allocator = std.testing.allocator;

    // A = [[1, 2, 3], [4, 5, 6]]
    var A = try Matrix(f64, 2, 3).init(allocator);
    defer A.deinit();
    A.set(0, 0, 1.0);
    A.set(0, 1, 2.0);
    A.set(0, 2, 3.0);
    A.set(1, 0, 4.0);
    A.set(1, 1, 5.0);
    A.set(1, 2, 6.0);

    var B = try Matrix(f64, 3, 2).init(allocator);
    defer B.deinit();

    A.transposeInto(&B);

    // B = [[1, 4], [2, 5], [3, 6]]
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), B.at(0, 0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), B.at(0, 1), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), B.at(1, 0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), B.at(1, 1), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), B.at(2, 0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), B.at(2, 1), 1e-10);
}

test "Matrix Frobenius norm" {
    const allocator = std.testing.allocator;

    // A = [[3, 0], [4, 0]]
    var A = try Matrix(f64, 2, 2).initWithValue(allocator, 0.0);
    defer A.deinit();
    A.set(0, 0, 3.0);
    A.set(1, 0, 4.0);

    // ||A||_F = sqrt(9 + 16) = 5
    const norm = A.frobeniusNorm();
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), norm, 1e-10);
}

test "Matrix clone" {
    const allocator = std.testing.allocator;

    var A = try Matrix(f64, 2, 2).initWithValue(allocator, 7.0);
    defer A.deinit();

    var B = try A.clone(allocator);
    defer B.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, 7.0), B.at(0, 0), 1e-10);

    // Ensure they're independent
    A.set(0, 0, 0.0);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), B.at(0, 0), 1e-10);
}
