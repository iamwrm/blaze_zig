//! SIMD Kernels for Small/Medium Matrix Operations
//!
//! This module provides SIMD-optimized implementations using Zig's @Vector
//! for operations on small to medium-sized matrices where BLAS overhead
//! would be excessive.

const std = @import("std");

/// Get the recommended SIMD vector length for a type
pub fn vecLen(comptime T: type) comptime_int {
    return std.simd.suggestVectorLength(T) orelse 4;
}

// =============================================================================
// BLAS Level 1: Vector-Vector Operations
// =============================================================================

/// SIMD dot product: x · y
pub fn dot(comptime T: type, comptime n: usize, x: *const [n]T, y: *const [n]T) T {
    const VecLen = vecLen(T);
    const Vec = @Vector(VecLen, T);

    var sum: Vec = @splat(0);
    var i: usize = 0;

    // Vectorized loop
    while (i + VecLen <= n) : (i += VecLen) {
        const xv: Vec = x[i..][0..VecLen].*;
        const yv: Vec = y[i..][0..VecLen].*;
        sum += xv * yv;
    }

    // Horizontal sum
    var result = @reduce(.Add, sum);

    // Scalar remainder
    while (i < n) : (i += 1) {
        result += x[i] * y[i];
    }

    return result;
}

/// SIMD axpy: y = α*x + y
pub fn axpy(comptime T: type, comptime n: usize, alpha: T, x: *const [n]T, y: *[n]T) void {
    const VecLen = vecLen(T);
    const Vec = @Vector(VecLen, T);
    const alpha_vec: Vec = @splat(alpha);

    var i: usize = 0;
    while (i + VecLen <= n) : (i += VecLen) {
        const xv: Vec = x[i..][0..VecLen].*;
        var yv: Vec = y[i..][0..VecLen].*;
        yv += alpha_vec * xv;
        y[i..][0..VecLen].* = yv;
    }

    // Scalar remainder
    while (i < n) : (i += 1) {
        y[i] += alpha * x[i];
    }
}

/// SIMD scal: x = α*x
pub fn scal(comptime T: type, comptime n: usize, alpha: T, x: *[n]T) void {
    const VecLen = vecLen(T);
    const Vec = @Vector(VecLen, T);
    const alpha_vec: Vec = @splat(alpha);

    var i: usize = 0;
    while (i + VecLen <= n) : (i += VecLen) {
        var xv: Vec = x[i..][0..VecLen].*;
        xv *= alpha_vec;
        x[i..][0..VecLen].* = xv;
    }

    while (i < n) : (i += 1) {
        x[i] *= alpha;
    }
}

/// SIMD nrm2: ||x||₂
pub fn nrm2(comptime T: type, comptime n: usize, x: *const [n]T) T {
    return @sqrt(dot(T, n, x, x));
}

/// SIMD asum: Σ|xᵢ|
pub fn asum(comptime T: type, comptime n: usize, x: *const [n]T) T {
    const VecLen = vecLen(T);
    const Vec = @Vector(VecLen, T);

    var sum: Vec = @splat(0);
    var i: usize = 0;

    while (i + VecLen <= n) : (i += VecLen) {
        const xv: Vec = x[i..][0..VecLen].*;
        sum += @abs(xv);
    }

    var result = @reduce(.Add, sum);

    while (i < n) : (i += 1) {
        result += @abs(x[i]);
    }

    return result;
}

/// SIMD copy: y = x
pub fn copyVec(comptime T: type, comptime n: usize, x: *const [n]T, y: *[n]T) void {
    const VecLen = vecLen(T);
    const Vec = @Vector(VecLen, T);

    var i: usize = 0;
    while (i + VecLen <= n) : (i += VecLen) {
        const xv: Vec = x[i..][0..VecLen].*;
        y[i..][0..VecLen].* = xv;
    }

    while (i < n) : (i += 1) {
        y[i] = x[i];
    }
}

/// SIMD add: z = x + y
pub fn add(comptime T: type, comptime n: usize, x: *const [n]T, y: *const [n]T, z: *[n]T) void {
    const VecLen = vecLen(T);
    const Vec = @Vector(VecLen, T);

    var i: usize = 0;
    while (i + VecLen <= n) : (i += VecLen) {
        const xv: Vec = x[i..][0..VecLen].*;
        const yv: Vec = y[i..][0..VecLen].*;
        z[i..][0..VecLen].* = xv + yv;
    }

    while (i < n) : (i += 1) {
        z[i] = x[i] + y[i];
    }
}

/// SIMD sub: z = x - y
pub fn sub(comptime T: type, comptime n: usize, x: *const [n]T, y: *const [n]T, z: *[n]T) void {
    const VecLen = vecLen(T);
    const Vec = @Vector(VecLen, T);

    var i: usize = 0;
    while (i + VecLen <= n) : (i += VecLen) {
        const xv: Vec = x[i..][0..VecLen].*;
        const yv: Vec = y[i..][0..VecLen].*;
        z[i..][0..VecLen].* = xv - yv;
    }

    while (i < n) : (i += 1) {
        z[i] = x[i] - y[i];
    }
}

// =============================================================================
// BLAS Level 2: Matrix-Vector Operations
// =============================================================================

/// SIMD gemv: y = α*A*x + β*y (row-major)
pub fn gemv(
    comptime T: type,
    comptime m: usize,
    comptime n: usize,
    alpha: T,
    A: *const [m * n]T,
    x: *const [n]T,
    beta: T,
    y: *[m]T,
) void {
    const VecLen = vecLen(T);
    const Vec = @Vector(VecLen, T);

    for (0..m) |i| {
        var sum: Vec = @splat(0);
        var j: usize = 0;

        while (j + VecLen <= n) : (j += VecLen) {
            const av: Vec = A[i * n + j ..][0..VecLen].*;
            const xv: Vec = x[j..][0..VecLen].*;
            sum += av * xv;
        }

        var row_sum = @reduce(.Add, sum);

        while (j < n) : (j += 1) {
            row_sum += A[i * n + j] * x[j];
        }

        y[i] = alpha * row_sum + beta * y[i];
    }
}

// =============================================================================
// BLAS Level 3: Matrix-Matrix Operations
// =============================================================================

/// SIMD gemm: C = α*A*B + β*C (blocked for cache efficiency)
pub fn gemm(
    comptime T: type,
    comptime m: usize,
    comptime n: usize,
    comptime k: usize,
    alpha: T,
    A: *const [m * k]T,
    B: *const [k * n]T,
    beta: T,
    C: *[m * n]T,
) void {
    const BLOCK = 32;
    const VecLen = vecLen(T);
    const Vec = @Vector(VecLen, T);

    // Scale C by beta
    if (beta == 0) {
        @memset(C, 0);
    } else if (beta != 1) {
        scal(T, m * n, beta, C);
    }

    // Blocked matrix multiply
    var ii: usize = 0;
    while (ii < m) : (ii += BLOCK) {
        const i_end = @min(ii + BLOCK, m);

        var kk: usize = 0;
        while (kk < k) : (kk += BLOCK) {
            const k_end = @min(kk + BLOCK, k);

            var jj: usize = 0;
            while (jj < n) : (jj += BLOCK) {
                const j_end = @min(jj + BLOCK, n);

                // Inner block multiply
                for (ii..i_end) |i| {
                    for (kk..k_end) |ki| {
                        const a_val = alpha * A[i * k + ki];
                        const a_vec: Vec = @splat(a_val);

                        var j: usize = jj;
                        while (j + VecLen <= j_end) : (j += VecLen) {
                            const bv: Vec = B[ki * n + j ..][0..VecLen].*;
                            var cv: Vec = C[i * n + j ..][0..VecLen].*;
                            cv += a_vec * bv;
                            C[i * n + j ..][0..VecLen].* = cv;
                        }

                        while (j < j_end) : (j += 1) {
                            C[i * n + j] += a_val * B[ki * n + j];
                        }
                    }
                }
            }
        }
    }
}

/// Transpose a matrix: B = Aᵀ
pub fn transpose(
    comptime T: type,
    comptime rows: usize,
    comptime cols: usize,
    A: *const [rows * cols]T,
    B: *[cols * rows]T,
) void {
    for (0..rows) |i| {
        for (0..cols) |j| {
            B[j * rows + i] = A[i * cols + j];
        }
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

test "simd dot product" {
    const x = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const y = [_]f64{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    const result = dot(f64, 8, &x, &y);
    try std.testing.expectApproxEqAbs(@as(f64, 36.0), result, 1e-10);
}

test "simd axpy" {
    const x = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var y = [_]f64{ 1.0, 1.0, 1.0, 1.0 };
    axpy(f64, 4, 2.0, &x, &y);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), y[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), y[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), y[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), y[3], 1e-10);
}

test "simd scal" {
    var x = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    scal(f64, 4, 2.0, &x);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), x[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), x[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), x[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 8.0), x[3], 1e-10);
}

test "simd nrm2" {
    const x = [_]f64{ 3.0, 4.0 };
    const result = nrm2(f64, 2, &x);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result, 1e-10);
}

test "simd asum" {
    const x = [_]f64{ -1.0, 2.0, -3.0, 4.0 };
    const result = asum(f64, 4, &x);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), result, 1e-10);
}

test "simd gemv" {
    // A = [[1, 2], [3, 4]]
    const A = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const x = [_]f64{ 1.0, 2.0 };
    var y = [_]f64{ 0.0, 0.0 };

    gemv(f64, 2, 2, 1.0, &A, &x, 0.0, &y);

    // y = A*x = [1*1 + 2*2, 3*1 + 4*2] = [5, 11]
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), y[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 11.0), y[1], 1e-10);
}

test "simd gemm" {
    // A = [[1, 2], [3, 4]], B = [[1, 0], [0, 1]] (identity)
    const A = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const B = [_]f64{ 1.0, 0.0, 0.0, 1.0 };
    var C = [_]f64{ 0.0, 0.0, 0.0, 0.0 };

    gemm(f64, 2, 2, 2, 1.0, &A, &B, 0.0, &C);

    // C = A * I = A
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), C[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), C[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), C[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), C[3], 1e-10);
}

test "simd transpose" {
    const A = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }; // 2x3
    var B: [6]f64 = undefined; // 3x2

    transpose(f64, 2, 3, &A, &B);

    // B should be [[1, 4], [2, 5], [3, 6]]
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), B[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), B[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), B[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), B[3], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), B[4], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), B[5], 1e-10);
}
