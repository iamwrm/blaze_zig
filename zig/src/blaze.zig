//! Blaze-Zig: A minimal port of Blaze C++ linear algebra library to Zig
//!
//! This library provides efficient matrix and vector operations with
//! an API similar to the original Blaze C++ library.

pub const matrix = @import("matrix.zig");
pub const vector = @import("vector.zig");
pub const build_options = @import("build_options");

pub const DynamicMatrix = matrix.DynamicMatrix;
pub const DynamicVector = vector.DynamicVector;

/// Check if any BLAS backend is enabled
pub const use_blas = build_options.use_mkl;

/// Check if Intel MKL is the backend
pub const use_intel_mkl = build_options.use_intel_mkl;

/// Check if OpenBLAS is the backend
pub const use_openblas = build_options.use_openblas;

test {
    _ = @import("matrix.zig");
    _ = @import("vector.zig");
}
