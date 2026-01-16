//! Blaze-Zig: A minimal port of Blaze C++ linear algebra library to Zig
//!
//! This library provides efficient matrix and vector operations with
//! an API similar to the original Blaze C++ library.

pub const matrix = @import("matrix.zig");
pub const vector = @import("vector.zig");

pub const DynamicMatrix = matrix.DynamicMatrix;
pub const DynamicVector = vector.DynamicVector;

test {
    _ = @import("matrix.zig");
    _ = @import("vector.zig");
}
