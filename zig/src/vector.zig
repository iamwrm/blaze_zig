//! Dynamic vector implementation for Blaze-Zig

const std = @import("std");
const Allocator = std.mem.Allocator;

/// A dynamically-sized vector
pub fn DynamicVector(comptime T: type) type {
    return struct {
        const Self = @This();

        data: []T,
        allocator: Allocator,

        /// Create a new vector with the given size
        pub fn init(allocator: Allocator, vec_size: usize) !Self {
            const data = try allocator.alloc(T, vec_size);
            @memset(data, 0);
            return Self{
                .data = data,
                .allocator = allocator,
            };
        }

        /// Create a vector from a slice of values
        pub fn fromSlice(allocator: Allocator, values: []const T) !Self {
            const data = try allocator.alloc(T, values.len);
            @memcpy(data, values);
            return Self{
                .data = data,
                .allocator = allocator,
            };
        }

        /// Create a vector from a comptime-known array
        pub fn fromArray(allocator: Allocator, comptime N: usize, values: [N]T) !Self {
            return fromSlice(allocator, &values);
        }

        /// Free the vector memory
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.data = &.{};
        }

        /// Get the size of the vector
        pub fn size(self: Self) usize {
            return self.data.len;
        }

        /// Get element at index
        pub fn get(self: Self, index: usize) T {
            return self.data[index];
        }

        /// Set element at index
        pub fn set(self: *Self, index: usize, value: T) void {
            self.data[index] = value;
        }

        /// Get a pointer to element at index
        pub fn ptr(self: *Self, index: usize) *T {
            return &self.data[index];
        }

        /// Dot product (inner product) of two vectors
        pub fn inner(a: Self, b: Self) !T {
            if (a.data.len != b.data.len) {
                return error.DimensionMismatch;
            }

            var result: T = 0;
            for (0..a.data.len) |i| {
                result += a.data[i] * b.data[i];
            }
            return result;
        }

        /// Element-wise addition: c = a + b
        pub fn add(allocator: Allocator, a: Self, b: Self) !Self {
            if (a.data.len != b.data.len) {
                return error.DimensionMismatch;
            }

            var result = try init(allocator, a.data.len);
            for (0..a.data.len) |i| {
                result.data[i] = a.data[i] + b.data[i];
            }
            return result;
        }

        /// Element-wise subtraction: c = a - b
        pub fn subtract(allocator: Allocator, a: Self, b: Self) !Self {
            if (a.data.len != b.data.len) {
                return error.DimensionMismatch;
            }

            var result = try init(allocator, a.data.len);
            for (0..a.data.len) |i| {
                result.data[i] = a.data[i] - b.data[i];
            }
            return result;
        }

        /// Scalar multiplication: b = a * scalar
        pub fn scale(allocator: Allocator, a: Self, scalar: T) !Self {
            var result = try init(allocator, a.data.len);
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

        /// Euclidean norm (L2 norm)
        pub fn norm(self: Self) T {
            var sum: T = 0;
            for (self.data) |val| {
                sum += val * val;
            }
            return @sqrt(sum);
        }

        /// Print vector to stdout
        pub fn print(self: Self, writer: anytype) !void {
            try writer.print("[", .{});
            for (0..self.data.len) |i| {
                if (i > 0) try writer.print(", ", .{});
                try writer.print("{d:.2}", .{self.get(i)});
            }
            try writer.print("]\n", .{});
        }
    };
}

test "vector creation and access" {
    const allocator = std.testing.allocator;

    var vec = try DynamicVector(f64).init(allocator, 3);
    defer vec.deinit();

    vec.set(0, 1.0);
    vec.set(1, 2.0);
    vec.set(2, 3.0);

    try std.testing.expectEqual(@as(f64, 1.0), vec.get(0));
    try std.testing.expectEqual(@as(f64, 2.0), vec.get(1));
    try std.testing.expectEqual(@as(f64, 3.0), vec.get(2));
}

test "vector from array" {
    const allocator = std.testing.allocator;

    var vec = try DynamicVector(f64).fromArray(allocator, 3, .{ 1.0, 2.0, 3.0 });
    defer vec.deinit();

    try std.testing.expectEqual(@as(usize, 3), vec.size());
    try std.testing.expectEqual(@as(f64, 1.0), vec.get(0));
    try std.testing.expectEqual(@as(f64, 2.0), vec.get(1));
    try std.testing.expectEqual(@as(f64, 3.0), vec.get(2));
}

test "dot product" {
    const allocator = std.testing.allocator;

    var v1 = try DynamicVector(f64).fromArray(allocator, 3, .{ 1.0, 2.0, 3.0 });
    defer v1.deinit();

    var v2 = try DynamicVector(f64).fromArray(allocator, 3, .{ 4.0, 5.0, 6.0 });
    defer v2.deinit();

    const dot = try DynamicVector(f64).inner(v1, v2);
    try std.testing.expectEqual(@as(f64, 32.0), dot);
}

test "vector addition" {
    const allocator = std.testing.allocator;

    var v1 = try DynamicVector(f64).fromArray(allocator, 3, .{ 1.0, 2.0, 3.0 });
    defer v1.deinit();

    var v2 = try DynamicVector(f64).fromArray(allocator, 3, .{ 4.0, 5.0, 6.0 });
    defer v2.deinit();

    var v3 = try DynamicVector(f64).add(allocator, v1, v2);
    defer v3.deinit();

    try std.testing.expectEqual(@as(f64, 5.0), v3.get(0));
    try std.testing.expectEqual(@as(f64, 7.0), v3.get(1));
    try std.testing.expectEqual(@as(f64, 9.0), v3.get(2));
}

test "scalar multiplication" {
    const allocator = std.testing.allocator;

    var v1 = try DynamicVector(f64).fromArray(allocator, 3, .{ 1.0, 2.0, 3.0 });
    defer v1.deinit();

    var v2 = try DynamicVector(f64).scale(allocator, v1, 2.0);
    defer v2.deinit();

    try std.testing.expectEqual(@as(f64, 2.0), v2.get(0));
    try std.testing.expectEqual(@as(f64, 4.0), v2.get(1));
    try std.testing.expectEqual(@as(f64, 6.0), v2.get(2));
}

test "norm" {
    const allocator = std.testing.allocator;

    var vec = try DynamicVector(f64).fromArray(allocator, 3, .{ 3.0, 4.0, 0.0 });
    defer vec.deinit();

    try std.testing.expectEqual(@as(f64, 5.0), vec.norm());
}
