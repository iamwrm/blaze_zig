//! CBLAS bindings for MKL
//!
//! This module provides Zig-idiomatic wrappers around MKL CBLAS functions
//! for all BLAS Level 1, 2, and 3 operations.

const std = @import("std");

pub const c = @cImport({
    @cInclude("mkl_cblas.h");
});

// =============================================================================
// Type Aliases
// =============================================================================

pub const Order = enum(c_int) {
    RowMajor = c.CblasRowMajor,
    ColMajor = c.CblasColMajor,
};

pub const Transpose = enum(c_int) {
    NoTrans = c.CblasNoTrans,
    Trans = c.CblasTrans,
    ConjTrans = c.CblasConjTrans,
};

pub const UpLo = enum(c_int) {
    Upper = c.CblasUpper,
    Lower = c.CblasLower,
};

pub const Diag = enum(c_int) {
    NonUnit = c.CblasNonUnit,
    Unit = c.CblasUnit,
};

pub const Side = enum(c_int) {
    Left = c.CblasLeft,
    Right = c.CblasRight,
};

// =============================================================================
// BLAS LEVEL 1: Vector-Vector Operations
// =============================================================================

/// DOT: x · y (unconjugated dot product)
pub fn dot(comptime T: type, n: c_int, x: [*]const T, incX: c_int, y: [*]const T, incY: c_int) T {
    return switch (T) {
        f32 => c.cblas_sdot(n, x, incX, y, incY),
        f64 => c.cblas_ddot(n, x, incX, y, incY),
        else => @compileError("dot: unsupported type, must be f32 or f64"),
    };
}

/// AXPY: y = α*x + y
pub fn axpy(comptime T: type, n: c_int, alpha: T, x: [*]const T, incX: c_int, y: [*]T, incY: c_int) void {
    switch (T) {
        f32 => c.cblas_saxpy(n, alpha, x, incX, y, incY),
        f64 => c.cblas_daxpy(n, alpha, x, incX, y, incY),
        else => @compileError("axpy: unsupported type, must be f32 or f64"),
    }
}

/// SCAL: x = α*x
pub fn scal(comptime T: type, n: c_int, alpha: T, x: [*]T, incX: c_int) void {
    switch (T) {
        f32 => c.cblas_sscal(n, alpha, x, incX),
        f64 => c.cblas_dscal(n, alpha, x, incX),
        else => @compileError("scal: unsupported type, must be f32 or f64"),
    }
}

/// NRM2: ||x||₂
pub fn nrm2(comptime T: type, n: c_int, x: [*]const T, incX: c_int) T {
    return switch (T) {
        f32 => c.cblas_snrm2(n, x, incX),
        f64 => c.cblas_dnrm2(n, x, incX),
        else => @compileError("nrm2: unsupported type, must be f32 or f64"),
    };
}

/// ASUM: Σ|xᵢ|
pub fn asum(comptime T: type, n: c_int, x: [*]const T, incX: c_int) T {
    return switch (T) {
        f32 => c.cblas_sasum(n, x, incX),
        f64 => c.cblas_dasum(n, x, incX),
        else => @compileError("asum: unsupported type, must be f32 or f64"),
    };
}

/// IAMAX: index of max |xᵢ|
pub fn iamax(comptime T: type, n: c_int, x: [*]const T, incX: c_int) usize {
    return switch (T) {
        f32 => @intCast(c.cblas_isamax(n, x, incX)),
        f64 => @intCast(c.cblas_idamax(n, x, incX)),
        else => @compileError("iamax: unsupported type, must be f32 or f64"),
    };
}

/// COPY: y = x
pub fn copy(comptime T: type, n: c_int, x: [*]const T, incX: c_int, y: [*]T, incY: c_int) void {
    switch (T) {
        f32 => c.cblas_scopy(n, x, incX, y, incY),
        f64 => c.cblas_dcopy(n, x, incX, y, incY),
        else => @compileError("copy: unsupported type, must be f32 or f64"),
    }
}

/// SWAP: x ↔ y
pub fn swap(comptime T: type, n: c_int, x: [*]T, incX: c_int, y: [*]T, incY: c_int) void {
    switch (T) {
        f32 => c.cblas_sswap(n, x, incX, y, incY),
        f64 => c.cblas_dswap(n, x, incX, y, incY),
        else => @compileError("swap: unsupported type, must be f32 or f64"),
    }
}

// =============================================================================
// BLAS LEVEL 2: Matrix-Vector Operations
// =============================================================================

/// GEMV: y = α*A*x + β*y
pub fn gemv(
    comptime T: type,
    order: Order,
    transA: Transpose,
    m: c_int,
    n: c_int,
    alpha: T,
    A: [*]const T,
    lda: c_int,
    x: [*]const T,
    incX: c_int,
    beta: T,
    y: [*]T,
    incY: c_int,
) void {
    switch (T) {
        f32 => c.cblas_sgemv(@intFromEnum(order), @intFromEnum(transA), m, n, alpha, A, lda, x, incX, beta, y, incY),
        f64 => c.cblas_dgemv(@intFromEnum(order), @intFromEnum(transA), m, n, alpha, A, lda, x, incX, beta, y, incY),
        else => @compileError("gemv: unsupported type, must be f32 or f64"),
    }
}

/// TRMV: x = A*x (triangular)
pub fn trmv(
    comptime T: type,
    order: Order,
    uplo: UpLo,
    transA: Transpose,
    diag: Diag,
    n: c_int,
    A: [*]const T,
    lda: c_int,
    x: [*]T,
    incX: c_int,
) void {
    switch (T) {
        f32 => c.cblas_strmv(@intFromEnum(order), @intFromEnum(uplo), @intFromEnum(transA), @intFromEnum(diag), n, A, lda, x, incX),
        f64 => c.cblas_dtrmv(@intFromEnum(order), @intFromEnum(uplo), @intFromEnum(transA), @intFromEnum(diag), n, A, lda, x, incX),
        else => @compileError("trmv: unsupported type, must be f32 or f64"),
    }
}

/// TRSV: x = A⁻¹*x (triangular solve)
pub fn trsv(
    comptime T: type,
    order: Order,
    uplo: UpLo,
    transA: Transpose,
    diag: Diag,
    n: c_int,
    A: [*]const T,
    lda: c_int,
    x: [*]T,
    incX: c_int,
) void {
    switch (T) {
        f32 => c.cblas_strsv(@intFromEnum(order), @intFromEnum(uplo), @intFromEnum(transA), @intFromEnum(diag), n, A, lda, x, incX),
        f64 => c.cblas_dtrsv(@intFromEnum(order), @intFromEnum(uplo), @intFromEnum(transA), @intFromEnum(diag), n, A, lda, x, incX),
        else => @compileError("trsv: unsupported type, must be f32 or f64"),
    }
}

/// SYMV: y = α*A*x + β*y (symmetric)
pub fn symv(
    comptime T: type,
    order: Order,
    uplo: UpLo,
    n: c_int,
    alpha: T,
    A: [*]const T,
    lda: c_int,
    x: [*]const T,
    incX: c_int,
    beta: T,
    y: [*]T,
    incY: c_int,
) void {
    switch (T) {
        f32 => c.cblas_ssymv(@intFromEnum(order), @intFromEnum(uplo), n, alpha, A, lda, x, incX, beta, y, incY),
        f64 => c.cblas_dsymv(@intFromEnum(order), @intFromEnum(uplo), n, alpha, A, lda, x, incX, beta, y, incY),
        else => @compileError("symv: unsupported type, must be f32 or f64"),
    }
}

/// GER: A = α*x*yᵀ + A (rank-1 update)
pub fn ger(
    comptime T: type,
    order: Order,
    m: c_int,
    n: c_int,
    alpha: T,
    x: [*]const T,
    incX: c_int,
    y: [*]const T,
    incY: c_int,
    A: [*]T,
    lda: c_int,
) void {
    switch (T) {
        f32 => c.cblas_sger(@intFromEnum(order), m, n, alpha, x, incX, y, incY, A, lda),
        f64 => c.cblas_dger(@intFromEnum(order), m, n, alpha, x, incX, y, incY, A, lda),
        else => @compileError("ger: unsupported type, must be f32 or f64"),
    }
}

/// SYR: A = α*x*xᵀ + A (symmetric rank-1 update)
pub fn syr(
    comptime T: type,
    order: Order,
    uplo: UpLo,
    n: c_int,
    alpha: T,
    x: [*]const T,
    incX: c_int,
    A: [*]T,
    lda: c_int,
) void {
    switch (T) {
        f32 => c.cblas_ssyr(@intFromEnum(order), @intFromEnum(uplo), n, alpha, x, incX, A, lda),
        f64 => c.cblas_dsyr(@intFromEnum(order), @intFromEnum(uplo), n, alpha, x, incX, A, lda),
        else => @compileError("syr: unsupported type, must be f32 or f64"),
    }
}

// =============================================================================
// BLAS LEVEL 3: Matrix-Matrix Operations
// =============================================================================

/// GEMM: C = α*A*B + β*C
pub fn gemm(
    comptime T: type,
    order: Order,
    transA: Transpose,
    transB: Transpose,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: T,
    A: [*]const T,
    lda: c_int,
    B: [*]const T,
    ldb: c_int,
    beta: T,
    C: [*]T,
    ldc: c_int,
) void {
    switch (T) {
        f32 => c.cblas_sgemm(@intFromEnum(order), @intFromEnum(transA), @intFromEnum(transB), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc),
        f64 => c.cblas_dgemm(@intFromEnum(order), @intFromEnum(transA), @intFromEnum(transB), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc),
        else => @compileError("gemm: unsupported type, must be f32 or f64"),
    }
}

/// TRMM: B = α*A*B or B = α*B*A (triangular matrix-matrix)
pub fn trmm(
    comptime T: type,
    order: Order,
    side: Side,
    uplo: UpLo,
    transA: Transpose,
    diag: Diag,
    m: c_int,
    n: c_int,
    alpha: T,
    A: [*]const T,
    lda: c_int,
    B: [*]T,
    ldb: c_int,
) void {
    switch (T) {
        f32 => c.cblas_strmm(@intFromEnum(order), @intFromEnum(side), @intFromEnum(uplo), @intFromEnum(transA), @intFromEnum(diag), m, n, alpha, A, lda, B, ldb),
        f64 => c.cblas_dtrmm(@intFromEnum(order), @intFromEnum(side), @intFromEnum(uplo), @intFromEnum(transA), @intFromEnum(diag), m, n, alpha, A, lda, B, ldb),
        else => @compileError("trmm: unsupported type, must be f32 or f64"),
    }
}

/// TRSM: B = α*A⁻¹*B or B = α*B*A⁻¹ (triangular solve)
pub fn trsm(
    comptime T: type,
    order: Order,
    side: Side,
    uplo: UpLo,
    transA: Transpose,
    diag: Diag,
    m: c_int,
    n: c_int,
    alpha: T,
    A: [*]const T,
    lda: c_int,
    B: [*]T,
    ldb: c_int,
) void {
    switch (T) {
        f32 => c.cblas_strsm(@intFromEnum(order), @intFromEnum(side), @intFromEnum(uplo), @intFromEnum(transA), @intFromEnum(diag), m, n, alpha, A, lda, B, ldb),
        f64 => c.cblas_dtrsm(@intFromEnum(order), @intFromEnum(side), @intFromEnum(uplo), @intFromEnum(transA), @intFromEnum(diag), m, n, alpha, A, lda, B, ldb),
        else => @compileError("trsm: unsupported type, must be f32 or f64"),
    }
}

/// SYMM: C = α*A*B + β*C or C = α*B*A + β*C (symmetric matrix-matrix)
pub fn symm(
    comptime T: type,
    order: Order,
    side: Side,
    uplo: UpLo,
    m: c_int,
    n: c_int,
    alpha: T,
    A: [*]const T,
    lda: c_int,
    B: [*]const T,
    ldb: c_int,
    beta: T,
    C: [*]T,
    ldc: c_int,
) void {
    switch (T) {
        f32 => c.cblas_ssymm(@intFromEnum(order), @intFromEnum(side), @intFromEnum(uplo), m, n, alpha, A, lda, B, ldb, beta, C, ldc),
        f64 => c.cblas_dsymm(@intFromEnum(order), @intFromEnum(side), @intFromEnum(uplo), m, n, alpha, A, lda, B, ldb, beta, C, ldc),
        else => @compileError("symm: unsupported type, must be f32 or f64"),
    }
}

/// SYRK: C = α*A*Aᵀ + β*C or C = α*Aᵀ*A + β*C (symmetric rank-k update)
pub fn syrk(
    comptime T: type,
    order: Order,
    uplo: UpLo,
    trans: Transpose,
    n: c_int,
    k: c_int,
    alpha: T,
    A: [*]const T,
    lda: c_int,
    beta: T,
    C: [*]T,
    ldc: c_int,
) void {
    switch (T) {
        f32 => c.cblas_ssyrk(@intFromEnum(order), @intFromEnum(uplo), @intFromEnum(trans), n, k, alpha, A, lda, beta, C, ldc),
        f64 => c.cblas_dsyrk(@intFromEnum(order), @intFromEnum(uplo), @intFromEnum(trans), n, k, alpha, A, lda, beta, C, ldc),
        else => @compileError("syrk: unsupported type, must be f32 or f64"),
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Check if a type is BLAS-compatible (f32 or f64)
pub fn isBLASCompatible(comptime T: type) bool {
    return T == f32 or T == f64;
}
