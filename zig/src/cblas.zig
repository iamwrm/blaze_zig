//! CBLAS bindings for Zig - compatible with Intel MKL and OpenBLAS

const std = @import("std");

/// CBLAS row/column major order
pub const CBLAS_LAYOUT = enum(c_int) {
    CblasRowMajor = 101,
    CblasColMajor = 102,
};

/// CBLAS transpose operation
pub const CBLAS_TRANSPOSE = enum(c_int) {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113,
};

/// External CBLAS dgemm function from MKL/OpenBLAS
/// C = alpha * op(A) * op(B) + beta * C
extern fn cblas_dgemm(
    layout: CBLAS_LAYOUT,
    transA: CBLAS_TRANSPOSE,
    transB: CBLAS_TRANSPOSE,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: f64,
    a: [*]const f64,
    lda: c_int,
    b: [*]const f64,
    ldb: c_int,
    beta: f64,
    c: [*]f64,
    ldc: c_int,
) void;

/// External CBLAS sgemm function for single precision
extern fn cblas_sgemm(
    layout: CBLAS_LAYOUT,
    transA: CBLAS_TRANSPOSE,
    transB: CBLAS_TRANSPOSE,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: f32,
    a: [*]const f32,
    lda: c_int,
    b: [*]const f32,
    ldb: c_int,
    beta: f32,
    c: [*]f32,
    ldc: c_int,
) void;

/// High-level wrapper for double-precision matrix multiplication
/// C = A * B where A is m x k, B is k x n, C is m x n
pub fn dgemm(
    m: usize,
    n: usize,
    k: usize,
    a: []const f64,
    b: []const f64,
    c: []f64,
) void {
    cblas_dgemm(
        .CblasRowMajor,
        .CblasNoTrans,
        .CblasNoTrans,
        @intCast(m),
        @intCast(n),
        @intCast(k),
        1.0, // alpha
        a.ptr,
        @intCast(k), // lda
        b.ptr,
        @intCast(n), // ldb
        0.0, // beta
        c.ptr,
        @intCast(n), // ldc
    );
}

/// High-level wrapper for single-precision matrix multiplication
/// C = A * B where A is m x k, B is k x n, C is m x n
pub fn sgemm(
    m: usize,
    n: usize,
    k: usize,
    a: []const f32,
    b: []const f32,
    c: []f32,
) void {
    cblas_sgemm(
        .CblasRowMajor,
        .CblasNoTrans,
        .CblasNoTrans,
        @intCast(m),
        @intCast(n),
        @intCast(k),
        1.0, // alpha
        a.ptr,
        @intCast(k), // lda
        b.ptr,
        @intCast(n), // ldb
        0.0, // beta
        c.ptr,
        @intCast(n), // ldc
    );
}
