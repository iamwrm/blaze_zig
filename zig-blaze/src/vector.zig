//! Fixed-size Vector Type with Expression Support
//!
//! This module provides a compile-time sized vector type that integrates
//! with the expression system for lazy evaluation and BLAS/SIMD optimization.

const std = @import("std");
const cblas = @import("blas/cblas.zig");
const simd = @import("kernels/simd.zig");

/// Size threshold for using BLAS over SIMD
const BLAS_THRESHOLD: usize = 64;

/// Fixed-size vector with compile-time known dimensions
pub fn Vector(comptime T: type, comptime Size: usize) type {
    return struct {
        const Self = @This();
        pub const ElementType = T;
        pub const size = Size;
        pub const is_vector = true;
        pub const is_matrix = false;

        // SIMD configuration
        const VecLen = simd.vecLen(T);
        pub const SimdVec = @Vector(VecLen, T);

        data: *align(64) [Size]T,
        allocator: ?std.mem.Allocator = null,

        // ─────────────────────────────────────────────────────────────────
        // Initialization
        // ─────────────────────────────────────────────────────────────────

        /// Create a new vector with uninitialized data
        pub fn init(allocator: std.mem.Allocator) !Self {
            const data = try allocator.alignedAlloc(T, @enumFromInt(6), Size); // 64-byte alignment (2^6 = 64)
            return .{
                .data = @ptrCast(data.ptr),
                .allocator = allocator,
            };
        }

        /// Create a new vector initialized to a specific value
        pub fn initWithValue(allocator: std.mem.Allocator, value: T) !Self {
            const result = try init(allocator);
            @memset(result.data, value);
            return result;
        }

        /// Create a vector from an existing array (no allocation, does not own memory)
        pub fn fromArray(data: *align(64) [Size]T) Self {
            return .{
                .data = data,
                .allocator = null,
            };
        }

        /// Free the vector memory
        pub fn deinit(self: *Self) void {
            if (self.allocator) |alloc| {
                const mem_slice: []align(64) T = @as([*]align(64) T, @ptrCast(self.data))[0..Size];
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

        /// Get element at index
        pub fn at(self: Self, i: usize) T {
            return self.data[i];
        }

        /// Set element at index
        pub fn set(self: *Self, i: usize, value: T) void {
            self.data[i] = value;
        }

        /// Get a slice of the underlying data
        pub fn slice(self: Self) []T {
            return self.data[0..Size];
        }

        // ─────────────────────────────────────────────────────────────────
        // BLAS Level 1 Operations
        // ─────────────────────────────────────────────────────────────────

        /// Dot product: x · y
        pub fn dot(self: Self, other: Self) T {
            if (comptime shouldUseBlas()) {
                return cblas.dot(T, @intCast(Size), self.constPtr(), 1, other.constPtr(), 1);
            } else {
                return simd.dot(T, Size, self.data, other.data);
            }
        }

        /// AXPY: result = self + α*x
        pub fn axpyInto(self: Self, alpha: T, x: Self, result: *Self) void {
            // Copy self to result first
            @memcpy(result.data, self.data);

            if (comptime shouldUseBlas()) {
                cblas.axpy(T, @intCast(Size), alpha, x.constPtr(), 1, result.ptr(), 1);
            } else {
                simd.axpy(T, Size, alpha, x.data, result.data);
            }
        }

        /// Scale in-place: self = α*self
        pub fn scaleInPlace(self: *Self, alpha: T) void {
            if (comptime shouldUseBlas()) {
                cblas.scal(T, @intCast(Size), alpha, self.ptr(), 1);
            } else {
                simd.scal(T, Size, alpha, self.data);
            }
        }

        /// Scale into result: result = α*self
        pub fn scaleInto(self: Self, alpha: T, result: *Self) void {
            @memcpy(result.data, self.data);
            result.scaleInPlace(alpha);
        }

        /// Euclidean norm: ||x||₂
        pub fn norm2(self: Self) T {
            if (comptime shouldUseBlas()) {
                return cblas.nrm2(T, @intCast(Size), self.constPtr(), 1);
            } else {
                return simd.nrm2(T, Size, self.data);
            }
        }

        /// Sum of absolute values: Σ|xᵢ|
        pub fn asum(self: Self) T {
            if (comptime shouldUseBlas()) {
                return cblas.asum(T, @intCast(Size), self.constPtr(), 1);
            } else {
                return simd.asum(T, Size, self.data);
            }
        }

        /// Index of max absolute value
        pub fn iamax(self: Self) usize {
            return cblas.iamax(T, @intCast(Size), self.constPtr(), 1);
        }

        /// Copy to another vector
        pub fn copyTo(self: Self, dest: *Self) void {
            if (comptime shouldUseBlas()) {
                cblas.copy(T, @intCast(Size), self.constPtr(), 1, dest.ptr(), 1);
            } else {
                simd.copyVec(T, Size, self.data, dest.data);
            }
        }

        /// Swap with another vector
        pub fn swap(self: *Self, other: *Self) void {
            cblas.swap(T, @intCast(Size), self.ptr(), 1, other.ptr(), 1);
        }

        /// Element-wise addition: result = self + other
        pub fn addInto(self: Self, other: Self, result: *Self) void {
            simd.add(T, Size, self.data, other.data, result.data);
        }

        /// Element-wise subtraction: result = self - other
        pub fn subInto(self: Self, other: Self, result: *Self) void {
            simd.sub(T, Size, self.data, other.data, result.data);
        }

        // ─────────────────────────────────────────────────────────────────
        // Utility Functions
        // ─────────────────────────────────────────────────────────────────

        /// Fill vector with a value
        pub fn fill(self: *Self, value: T) void {
            @memset(self.data, value);
        }

        /// Fill vector with random values
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

        /// Sum all elements
        pub fn sum(self: Self) T {
            var total: T = 0;
            for (self.data) |elem| {
                total += elem;
            }
            return total;
        }

        /// Clone the vector
        pub fn clone(self: Self, allocator: std.mem.Allocator) !Self {
            const new_vec = try init(allocator);
            @memcpy(new_vec.data, self.data);
            return new_vec;
        }

        /// Print vector (for debugging)
        pub fn print(self: Self, writer: anytype) !void {
            try writer.print("Vector({}):\n  [", .{Size});
            for (self.data, 0..) |elem, i| {
                if (i > 0) try writer.print(", ", .{});
                try writer.print("{d:.4}", .{elem});
            }
            try writer.print("]\n", .{});
        }

        // ─────────────────────────────────────────────────────────────────
        // Helper Functions
        // ─────────────────────────────────────────────────────────────────

        fn shouldUseBlas() bool {
            return cblas.isBLASCompatible(T) and Size >= BLAS_THRESHOLD;
        }
    };
}

// =============================================================================
// Type Aliases
// =============================================================================

/// Common f64 vector types
pub fn VectorF64(comptime Size: usize) type {
    return Vector(f64, Size);
}

/// Common f32 vector types
pub fn VectorF32(comptime Size: usize) type {
    return Vector(f32, Size);
}

// =============================================================================
// Unit Tests
// =============================================================================

test "Vector initialization and access" {
    const allocator = std.testing.allocator;

    var v = try Vector(f64, 4).initWithValue(allocator, 1.0);
    defer v.deinit();

    try std.testing.expectEqual(@as(f64, 1.0), v.at(0));
    try std.testing.expectEqual(@as(f64, 1.0), v.at(3));

    v.set(2, 5.0);
    try std.testing.expectEqual(@as(f64, 5.0), v.at(2));
}

test "Vector dot product" {
    const allocator = std.testing.allocator;

    var x = try Vector(f64, 4).initWithValue(allocator, 1.0);
    defer x.deinit();
    var y = try Vector(f64, 4).initWithValue(allocator, 2.0);
    defer y.deinit();

    const result = x.dot(y);
    try std.testing.expectApproxEqAbs(@as(f64, 8.0), result, 1e-10);
}

test "Vector norm2" {
    const allocator = std.testing.allocator;

    var v = try Vector(f64, 2).init(allocator);
    defer v.deinit();
    v.set(0, 3.0);
    v.set(1, 4.0);

    const result = v.norm2();
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result, 1e-10);
}

test "Vector axpy" {
    const allocator = std.testing.allocator;

    var x = try Vector(f64, 4).initWithValue(allocator, 2.0);
    defer x.deinit();
    var y = try Vector(f64, 4).initWithValue(allocator, 1.0);
    defer y.deinit();
    var z = try Vector(f64, 4).init(allocator);
    defer z.deinit();

    // z = y + 2*x = 1 + 2*2 = 5
    y.axpyInto(2.0, x, &z);

    try std.testing.expectApproxEqAbs(@as(f64, 5.0), z.at(0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), z.at(3), 1e-10);
}

test "Vector scale" {
    const allocator = std.testing.allocator;

    var v = try Vector(f64, 4).initWithValue(allocator, 2.0);
    defer v.deinit();

    v.scaleInPlace(3.0);

    try std.testing.expectApproxEqAbs(@as(f64, 6.0), v.at(0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), v.at(3), 1e-10);
}

test "Vector asum" {
    const allocator = std.testing.allocator;

    var v = try Vector(f64, 4).init(allocator);
    defer v.deinit();
    v.set(0, -1.0);
    v.set(1, 2.0);
    v.set(2, -3.0);
    v.set(3, 4.0);

    const result = v.asum();
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), result, 1e-10);
}

test "Vector add and sub" {
    const allocator = std.testing.allocator;

    var x = try Vector(f64, 4).initWithValue(allocator, 3.0);
    defer x.deinit();
    var y = try Vector(f64, 4).initWithValue(allocator, 1.0);
    defer y.deinit();
    var z = try Vector(f64, 4).init(allocator);
    defer z.deinit();

    x.addInto(y, &z);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), z.at(0), 1e-10);

    x.subInto(y, &z);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), z.at(0), 1e-10);
}

test "Vector sum" {
    const allocator = std.testing.allocator;

    var v = try Vector(f64, 4).init(allocator);
    defer v.deinit();
    v.set(0, 1.0);
    v.set(1, 2.0);
    v.set(2, 3.0);
    v.set(3, 4.0);

    const result = v.sum();
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), result, 1e-10);
}

test "Vector clone" {
    const allocator = std.testing.allocator;

    var v = try Vector(f64, 4).initWithValue(allocator, 7.0);
    defer v.deinit();

    var v2 = try v.clone(allocator);
    defer v2.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, 7.0), v2.at(0), 1e-10);

    // Ensure they're independent
    v.set(0, 0.0);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), v2.at(0), 1e-10);
}
