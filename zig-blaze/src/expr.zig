//! Expression System for Lazy Evaluation
//!
//! This module provides expression types that enable lazy evaluation and
//! automatic kernel selection (BLAS vs SIMD) based on operation type and size.
//!
//! Example usage:
//! ```zig
//! // Create expression (lazy)
//! const expr = A.expr().mul(B.expr()).scale(2.0).add(C.expr());
//!
//! // Evaluate into result
//! expr.evalInto(&result);
//! ```

const std = @import("std");
const cblas = @import("blas/cblas.zig");
const simd = @import("kernels/simd.zig");
const vec = @import("vector.zig");
const mat = @import("matrix.zig");

/// Expression kind for pattern matching
pub const ExprKind = enum {
    // Leaves
    vector,
    matrix,
    scalar,

    // Level 1 (vector-vector)
    dot,
    axpy,
    add,
    sub,
    scale,
    norm2,
    asum,

    // Level 2 (matrix-vector)
    gemv,

    // Level 3 (matrix-matrix)
    gemm,

    // Modifiers
    transpose,
    negate,
};

// =============================================================================
// KERNEL SELECTION
// =============================================================================

/// Size threshold for using BLAS over SIMD
const BLAS_THRESHOLD: usize = 64;

/// Determine if BLAS should be used for an operation
pub fn shouldUseBlas(comptime T: type, comptime size: usize, comptime kind: ExprKind) bool {
    if (!cblas.isBLASCompatible(T)) return false;

    return switch (kind) {
        .dot, .axpy, .norm2, .asum => size >= BLAS_THRESHOLD,
        .scale => size >= BLAS_THRESHOLD * 2,
        .gemv => size >= BLAS_THRESHOLD / 2,
        .gemm => size >= BLAS_THRESHOLD,
        else => false,
    };
}

// =============================================================================
// VECTOR EXPRESSION WRAPPER
// =============================================================================

/// Expression wrapper for vectors that enables fluent chaining
pub fn VectorExpr(comptime T: type, comptime Size: usize) type {
    return struct {
        const Self = @This();
        pub const ElementType = T;
        pub const size = Size;
        pub const kind = ExprKind.vector;
        pub const VectorType = vec.Vector(T, Size);

        inner: VectorType,

        /// Create from a vector
        pub fn init(v: VectorType) Self {
            return .{ .inner = v };
        }

        // ─────────────────────────────────────────────────────────────────
        // BLAS Level 1 Operations
        // ─────────────────────────────────────────────────────────────────

        /// Dot product: self · other
        pub fn dot(self: Self, other: Self) DotExpr(T, Size) {
            return DotExpr(T, Size).init(self.inner, other.inner);
        }

        /// Scaled add: self + α*other (AXPY pattern)
        pub fn axpy(self: Self, alpha: T, other: Self) AxpyExpr(T, Size) {
            return AxpyExpr(T, Size).init(self.inner, alpha, other.inner);
        }

        /// Scale: α*self
        pub fn scale(self: Self, alpha: T) ScaleVectorExpr(T, Size) {
            return ScaleVectorExpr(T, Size).init(self.inner, alpha);
        }

        /// Add: self + other
        pub fn add(self: Self, other: Self) AddVectorExpr(T, Size) {
            return AddVectorExpr(T, Size).init(self.inner, other.inner);
        }

        /// Subtract: self - other
        pub fn sub(self: Self, other: Self) SubVectorExpr(T, Size) {
            return SubVectorExpr(T, Size).init(self.inner, other.inner);
        }

        /// Euclidean norm: ||self||₂
        pub fn norm2(self: Self) Norm2Expr(T, Size) {
            return Norm2Expr(T, Size).init(self.inner);
        }

        /// Sum of absolute values: Σ|xᵢ|
        pub fn asum(self: Self) AsumExpr(T, Size) {
            return AsumExpr(T, Size).init(self.inner);
        }

        /// Evaluate and store in result
        pub fn evalInto(self: Self, result: *VectorType) void {
            @memcpy(result.data, self.inner.data);
        }
    };
}

// =============================================================================
// MATRIX EXPRESSION WRAPPER
// =============================================================================

/// Expression wrapper for matrices that enables fluent chaining
pub fn MatrixExpr(comptime T: type, comptime Rows: usize, comptime Cols: usize) type {
    return struct {
        const Self = @This();
        pub const ElementType = T;
        pub const rows = Rows;
        pub const cols = Cols;
        pub const kind = ExprKind.matrix;
        pub const MatrixType = mat.Matrix(T, Rows, Cols);

        inner: MatrixType,

        /// Create from a matrix
        pub fn init(m: MatrixType) Self {
            return .{ .inner = m };
        }

        // ─────────────────────────────────────────────────────────────────
        // BLAS Level 2 Operations (Matrix-Vector)
        // ─────────────────────────────────────────────────────────────────

        /// Matrix-vector multiply: self*x
        pub fn mulVec(self: Self, x: VectorExpr(T, Cols)) GemvExpr(T, Rows, Cols) {
            return GemvExpr(T, Rows, Cols).init(self.inner, x.inner);
        }

        // ─────────────────────────────────────────────────────────────────
        // BLAS Level 3 Operations (Matrix-Matrix)
        // ─────────────────────────────────────────────────────────────────

        /// Matrix-matrix multiply: self*other
        pub fn mul(self: Self, comptime OtherCols: usize, other: MatrixExpr(T, Cols, OtherCols)) GemmExpr(T, Rows, Cols, OtherCols) {
            return GemmExpr(T, Rows, Cols, OtherCols).init(self.inner, other.inner);
        }

        /// Scale: α*self
        pub fn scale(self: Self, alpha: T) ScaleMatrixExpr(T, Rows, Cols) {
            return ScaleMatrixExpr(T, Rows, Cols).init(self.inner, alpha);
        }

        /// Add: self + other
        pub fn add(self: Self, other: Self) AddMatrixExpr(T, Rows, Cols) {
            return AddMatrixExpr(T, Rows, Cols).init(self.inner, other.inner);
        }

        /// Subtract: self - other
        pub fn sub(self: Self, other: Self) SubMatrixExpr(T, Rows, Cols) {
            return SubMatrixExpr(T, Rows, Cols).init(self.inner, other.inner);
        }

        /// Transpose: selfᵀ
        pub fn transpose(self: Self) TransposeExpr(T, Rows, Cols) {
            return TransposeExpr(T, Rows, Cols).init(self.inner);
        }

        /// Evaluate and store in result
        pub fn evalInto(self: Self, result: *MatrixType) void {
            @memcpy(result.data, self.inner.data);
        }
    };
}

// =============================================================================
// BLAS LEVEL 1 EXPRESSION TYPES
// =============================================================================

/// DOT: x · y → scalar
pub fn DotExpr(comptime T: type, comptime Size: usize) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.dot;
        pub const ResultType = T;

        x: vec.Vector(T, Size),
        y: vec.Vector(T, Size),

        pub fn init(x: vec.Vector(T, Size), y: vec.Vector(T, Size)) Self {
            return .{ .x = x, .y = y };
        }

        /// Evaluate and return the scalar result
        pub fn eval(self: Self) T {
            return self.x.dot(self.y);
        }
    };
}

/// AXPY: z = y + α*x
pub fn AxpyExpr(comptime T: type, comptime Size: usize) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.axpy;
        pub const ResultType = vec.Vector(T, Size);

        y: vec.Vector(T, Size),
        alpha: T,
        x: vec.Vector(T, Size),

        pub fn init(y: vec.Vector(T, Size), alpha: T, x: vec.Vector(T, Size)) Self {
            return .{ .y = y, .alpha = alpha, .x = x };
        }

        /// Chain: add another scaled vector
        pub fn axpy(self: Self, alpha2: T, x2: vec.Vector(T, Size)) AxpyChainExpr(T, Size) {
            return AxpyChainExpr(T, Size).init(self, alpha2, x2);
        }

        /// Evaluate into result
        pub fn evalInto(self: Self, result: *vec.Vector(T, Size)) void {
            self.y.axpyInto(self.alpha, self.x, result);
        }
    };
}

/// Chained AXPY for multiple additions
pub fn AxpyChainExpr(comptime T: type, comptime Size: usize) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.axpy;
        pub const ResultType = vec.Vector(T, Size);

        base: AxpyExpr(T, Size),
        alpha: T,
        x: vec.Vector(T, Size),

        pub fn init(base: AxpyExpr(T, Size), alpha: T, x: vec.Vector(T, Size)) Self {
            return .{ .base = base, .alpha = alpha, .x = x };
        }

        /// Evaluate into result
        pub fn evalInto(self: Self, result: *vec.Vector(T, Size)) void {
            // First evaluate base
            self.base.evalInto(result);
            // Then add the new term
            if (comptime shouldUseBlas(T, Size, .axpy)) {
                cblas.axpy(T, @intCast(Size), self.alpha, self.x.constPtr(), 1, result.ptr(), 1);
            } else {
                simd.axpy(T, Size, self.alpha, self.x.data, result.data);
            }
        }
    };
}

/// SCAL: z = α*x (vector)
pub fn ScaleVectorExpr(comptime T: type, comptime Size: usize) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.scale;
        pub const ResultType = vec.Vector(T, Size);

        x: vec.Vector(T, Size),
        alpha: T,

        pub fn init(x: vec.Vector(T, Size), alpha: T) Self {
            return .{ .x = x, .alpha = alpha };
        }

        /// Chain: add another vector
        pub fn add(self: Self, other: vec.Vector(T, Size)) AxpyExpr(T, Size) {
            return AxpyExpr(T, Size).init(other, self.alpha, self.x);
        }

        /// Evaluate into result
        pub fn evalInto(self: Self, result: *vec.Vector(T, Size)) void {
            self.x.scaleInto(self.alpha, result);
        }
    };
}

/// ADD: z = x + y (vector)
pub fn AddVectorExpr(comptime T: type, comptime Size: usize) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.add;
        pub const ResultType = vec.Vector(T, Size);

        x: vec.Vector(T, Size),
        y: vec.Vector(T, Size),

        pub fn init(x: vec.Vector(T, Size), y: vec.Vector(T, Size)) Self {
            return .{ .x = x, .y = y };
        }

        /// Evaluate into result
        pub fn evalInto(self: Self, result: *vec.Vector(T, Size)) void {
            self.x.addInto(self.y, result);
        }
    };
}

/// SUB: z = x - y (vector)
pub fn SubVectorExpr(comptime T: type, comptime Size: usize) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.sub;
        pub const ResultType = vec.Vector(T, Size);

        x: vec.Vector(T, Size),
        y: vec.Vector(T, Size),

        pub fn init(x: vec.Vector(T, Size), y: vec.Vector(T, Size)) Self {
            return .{ .x = x, .y = y };
        }

        /// Evaluate into result
        pub fn evalInto(self: Self, result: *vec.Vector(T, Size)) void {
            self.x.subInto(self.y, result);
        }
    };
}

/// NRM2: ||x||₂
pub fn Norm2Expr(comptime T: type, comptime Size: usize) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.norm2;
        pub const ResultType = T;

        x: vec.Vector(T, Size),

        pub fn init(x: vec.Vector(T, Size)) Self {
            return .{ .x = x };
        }

        /// Evaluate and return the scalar result
        pub fn eval(self: Self) T {
            return self.x.norm2();
        }
    };
}

/// ASUM: Σ|xᵢ|
pub fn AsumExpr(comptime T: type, comptime Size: usize) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.asum;
        pub const ResultType = T;

        x: vec.Vector(T, Size),

        pub fn init(x: vec.Vector(T, Size)) Self {
            return .{ .x = x };
        }

        /// Evaluate and return the scalar result
        pub fn eval(self: Self) T {
            return self.x.asum();
        }
    };
}

// =============================================================================
// BLAS LEVEL 2 EXPRESSION TYPES
// =============================================================================

/// GEMV: y = α*A*x + β*y
pub fn GemvExpr(comptime T: type, comptime M: usize, comptime N: usize) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.gemv;
        pub const ResultType = vec.Vector(T, M);

        A: mat.Matrix(T, M, N),
        x: vec.Vector(T, N),
        alpha: T = 1,
        beta: T = 0,
        y: ?vec.Vector(T, M) = null,

        pub fn init(A: mat.Matrix(T, M, N), x: vec.Vector(T, N)) Self {
            return .{ .A = A, .x = x };
        }

        /// Set alpha coefficient
        pub fn setAlpha(self: Self, alpha: T) Self {
            var result = self;
            result.alpha = alpha;
            return result;
        }

        /// Add y vector: y = α*A*x + y
        pub fn addY(self: Self, y: vec.Vector(T, M)) Self {
            var result = self;
            result.beta = 1;
            result.y = y;
            return result;
        }

        /// Scale the result
        pub fn scale(self: Self, s: T) Self {
            var result = self;
            result.alpha *= s;
            return result;
        }

        /// Evaluate into result
        pub fn evalInto(self: Self, result: *vec.Vector(T, M)) void {
            // Initialize result with y if present
            if (self.y) |y_vec| {
                @memcpy(result.data, y_vec.data);
            } else {
                @memset(result.data, 0);
            }

            if (comptime shouldUseBlas(T, M * N, .gemv)) {
                cblas.gemv(
                    T,
                    .RowMajor,
                    .NoTrans,
                    @intCast(M),
                    @intCast(N),
                    self.alpha,
                    self.A.constPtr(),
                    @intCast(N),
                    self.x.constPtr(),
                    1,
                    self.beta,
                    result.ptr(),
                    1,
                );
            } else {
                simd.gemv(T, M, N, self.alpha, self.A.data, self.x.data, self.beta, result.data);
            }
        }
    };
}

// =============================================================================
// BLAS LEVEL 3 EXPRESSION TYPES
// =============================================================================

/// GEMM: C = α*A*B + β*C
pub fn GemmExpr(comptime T: type, comptime M: usize, comptime K: usize, comptime N: usize) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.gemm;
        pub const ResultType = mat.Matrix(T, M, N);

        A: mat.Matrix(T, M, K),
        B: mat.Matrix(T, K, N),
        alpha: T = 1,
        beta: T = 0,
        C: ?mat.Matrix(T, M, N) = null,

        pub fn init(A: mat.Matrix(T, M, K), B: mat.Matrix(T, K, N)) Self {
            return .{ .A = A, .B = B };
        }

        /// Set alpha coefficient
        pub fn setAlpha(self: Self, alpha: T) Self {
            var result = self;
            result.alpha = alpha;
            return result;
        }

        /// Scale the result: α*(A*B)
        pub fn scale(self: Self, s: T) Self {
            var result = self;
            result.alpha *= s;
            return result;
        }

        /// Add C matrix: C = α*A*B + C
        pub fn addC(self: Self, C_mat: mat.Matrix(T, M, N)) Self {
            var result = self;
            result.beta = 1;
            result.C = C_mat;
            return result;
        }

        /// Chain: (A*B)*D
        pub fn mul(self: Self, comptime P: usize, D: mat.Matrix(T, N, P)) GemmChainExpr(T, M, K, N, P) {
            return GemmChainExpr(T, M, K, N, P).init(self, D);
        }

        /// Evaluate into result
        pub fn evalInto(self: Self, result: *mat.Matrix(T, M, N)) void {
            // Initialize result with C if present
            if (self.C) |c_mat| {
                @memcpy(result.data, c_mat.data);
            } else {
                @memset(result.data, 0);
            }

            if (comptime shouldUseBlas(T, M * N, .gemm)) {
                cblas.gemm(
                    T,
                    .RowMajor,
                    .NoTrans,
                    .NoTrans,
                    @intCast(M),
                    @intCast(N),
                    @intCast(K),
                    self.alpha,
                    self.A.constPtr(),
                    @intCast(K),
                    self.B.constPtr(),
                    @intCast(N),
                    self.beta,
                    result.ptr(),
                    @intCast(N),
                );
            } else {
                simd.gemm(T, M, N, K, self.alpha, self.A.data, self.B.data, self.beta, result.data);
            }
        }
    };
}

/// Chained GEMM for (A*B)*C
pub fn GemmChainExpr(comptime T: type, comptime M: usize, comptime K: usize, comptime N: usize, comptime P: usize) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.gemm;
        pub const ResultType = mat.Matrix(T, M, P);

        base: GemmExpr(T, M, K, N),
        D: mat.Matrix(T, N, P),

        pub fn init(base: GemmExpr(T, M, K, N), D: mat.Matrix(T, N, P)) Self {
            return .{ .base = base, .D = D };
        }

        /// Evaluate into result (requires temp allocation)
        pub fn evalIntoWithAlloc(self: Self, allocator: std.mem.Allocator, result: *mat.Matrix(T, M, P)) !void {
            // Allocate temporary for intermediate result
            var temp = try mat.Matrix(T, M, N).init(allocator);
            defer temp.deinit();

            // Evaluate A*B into temp
            self.base.evalInto(&temp);

            // Then temp*D into result
            temp.gemmInto(self.D, 1, 0, result);
        }
    };
}

/// Scale matrix expression
pub fn ScaleMatrixExpr(comptime T: type, comptime Rows: usize, comptime Cols: usize) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.scale;
        pub const ResultType = mat.Matrix(T, Rows, Cols);

        A: mat.Matrix(T, Rows, Cols),
        alpha: T,

        pub fn init(A: mat.Matrix(T, Rows, Cols), alpha: T) Self {
            return .{ .A = A, .alpha = alpha };
        }

        /// Evaluate into result
        pub fn evalInto(self: Self, result: *mat.Matrix(T, Rows, Cols)) void {
            self.A.scaleInto(self.alpha, result);
        }
    };
}

/// Add matrix expression
pub fn AddMatrixExpr(comptime T: type, comptime Rows: usize, comptime Cols: usize) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.add;
        pub const ResultType = mat.Matrix(T, Rows, Cols);

        A: mat.Matrix(T, Rows, Cols),
        B: mat.Matrix(T, Rows, Cols),

        pub fn init(A: mat.Matrix(T, Rows, Cols), B: mat.Matrix(T, Rows, Cols)) Self {
            return .{ .A = A, .B = B };
        }

        /// Evaluate into result
        pub fn evalInto(self: Self, result: *mat.Matrix(T, Rows, Cols)) void {
            self.A.addInto(self.B, result);
        }
    };
}

/// Subtract matrix expression
pub fn SubMatrixExpr(comptime T: type, comptime Rows: usize, comptime Cols: usize) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.sub;
        pub const ResultType = mat.Matrix(T, Rows, Cols);

        A: mat.Matrix(T, Rows, Cols),
        B: mat.Matrix(T, Rows, Cols),

        pub fn init(A: mat.Matrix(T, Rows, Cols), B: mat.Matrix(T, Rows, Cols)) Self {
            return .{ .A = A, .B = B };
        }

        /// Evaluate into result
        pub fn evalInto(self: Self, result: *mat.Matrix(T, Rows, Cols)) void {
            self.A.subInto(self.B, result);
        }
    };
}

/// Transpose expression
pub fn TransposeExpr(comptime T: type, comptime Rows: usize, comptime Cols: usize) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.transpose;
        pub const ResultType = mat.Matrix(T, Cols, Rows);

        A: mat.Matrix(T, Rows, Cols),

        pub fn init(A: mat.Matrix(T, Rows, Cols)) Self {
            return .{ .A = A };
        }

        /// Evaluate into result
        pub fn evalInto(self: Self, result: *mat.Matrix(T, Cols, Rows)) void {
            self.A.transposeInto(result);
        }
    };
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Create a vector expression from a vector
pub fn vectorExpr(comptime T: type, comptime Size: usize, v: vec.Vector(T, Size)) VectorExpr(T, Size) {
    return VectorExpr(T, Size).init(v);
}

/// Create a matrix expression from a matrix
pub fn matrixExpr(comptime T: type, comptime Rows: usize, comptime Cols: usize, m: mat.Matrix(T, Rows, Cols)) MatrixExpr(T, Rows, Cols) {
    return MatrixExpr(T, Rows, Cols).init(m);
}

// =============================================================================
// Unit Tests
// =============================================================================

test "VectorExpr dot product" {
    const allocator = std.testing.allocator;

    var x = try vec.Vector(f64, 4).initWithValue(allocator, 1.0);
    defer x.deinit();
    var y = try vec.Vector(f64, 4).initWithValue(allocator, 2.0);
    defer y.deinit();

    const x_expr = VectorExpr(f64, 4).init(x);
    const y_expr = VectorExpr(f64, 4).init(y);

    const result = x_expr.dot(y_expr).eval();
    try std.testing.expectApproxEqAbs(@as(f64, 8.0), result, 1e-10);
}

test "VectorExpr axpy" {
    const allocator = std.testing.allocator;

    var x = try vec.Vector(f64, 4).initWithValue(allocator, 2.0);
    defer x.deinit();
    var y = try vec.Vector(f64, 4).initWithValue(allocator, 1.0);
    defer y.deinit();
    var z = try vec.Vector(f64, 4).init(allocator);
    defer z.deinit();

    const x_expr = VectorExpr(f64, 4).init(x);
    const y_expr = VectorExpr(f64, 4).init(y);

    // z = y + 2*x = 1 + 2*2 = 5
    y_expr.axpy(2.0, x_expr).evalInto(&z);

    try std.testing.expectApproxEqAbs(@as(f64, 5.0), z.at(0), 1e-10);
}

test "VectorExpr norm2" {
    const allocator = std.testing.allocator;

    var v = try vec.Vector(f64, 2).init(allocator);
    defer v.deinit();
    v.set(0, 3.0);
    v.set(1, 4.0);

    const v_expr = VectorExpr(f64, 2).init(v);
    const result = v_expr.norm2().eval();

    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result, 1e-10);
}

test "MatrixExpr multiply" {
    const allocator = std.testing.allocator;

    // A = [[1, 2], [3, 4]]
    var A = try mat.Matrix(f64, 2, 2).init(allocator);
    defer A.deinit();
    A.set(0, 0, 1.0);
    A.set(0, 1, 2.0);
    A.set(1, 0, 3.0);
    A.set(1, 1, 4.0);

    // I = identity
    var I = try mat.Matrix(f64, 2, 2).initIdentity(allocator);
    defer I.deinit();

    var C = try mat.Matrix(f64, 2, 2).init(allocator);
    defer C.deinit();

    const A_expr = MatrixExpr(f64, 2, 2).init(A);
    const I_expr = MatrixExpr(f64, 2, 2).init(I);

    // C = A * I = A
    A_expr.mul(2, I_expr).evalInto(&C);

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), C.at(0, 0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), C.at(0, 1), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), C.at(1, 0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), C.at(1, 1), 1e-10);
}

test "MatrixExpr scaled multiply" {
    const allocator = std.testing.allocator;

    var A = try mat.Matrix(f64, 2, 2).initIdentity(allocator);
    defer A.deinit();
    var B = try mat.Matrix(f64, 2, 2).initIdentity(allocator);
    defer B.deinit();
    var C = try mat.Matrix(f64, 2, 2).init(allocator);
    defer C.deinit();

    const A_expr = MatrixExpr(f64, 2, 2).init(A);
    const B_expr = MatrixExpr(f64, 2, 2).init(B);

    // C = 2.0 * A * B = 2.0 * I * I = 2.0 * I
    A_expr.mul(2, B_expr).scale(2.0).evalInto(&C);

    try std.testing.expectApproxEqAbs(@as(f64, 2.0), C.at(0, 0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), C.at(0, 1), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), C.at(1, 1), 1e-10);
}

test "GemvExpr matrix-vector multiply" {
    const allocator = std.testing.allocator;

    // A = [[1, 2], [3, 4]]
    var A = try mat.Matrix(f64, 2, 2).init(allocator);
    defer A.deinit();
    A.set(0, 0, 1.0);
    A.set(0, 1, 2.0);
    A.set(1, 0, 3.0);
    A.set(1, 1, 4.0);

    var x = try vec.Vector(f64, 2).init(allocator);
    defer x.deinit();
    x.set(0, 1.0);
    x.set(1, 2.0);

    var y = try vec.Vector(f64, 2).init(allocator);
    defer y.deinit();

    const A_expr = MatrixExpr(f64, 2, 2).init(A);
    const x_expr = VectorExpr(f64, 2).init(x);

    // y = A*x = [5, 11]
    A_expr.mulVec(x_expr).evalInto(&y);

    try std.testing.expectApproxEqAbs(@as(f64, 5.0), y.at(0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 11.0), y.at(1), 1e-10);
}

test "MatrixExpr transpose" {
    const allocator = std.testing.allocator;

    // A = [[1, 2, 3], [4, 5, 6]]
    var A = try mat.Matrix(f64, 2, 3).init(allocator);
    defer A.deinit();
    A.set(0, 0, 1.0);
    A.set(0, 1, 2.0);
    A.set(0, 2, 3.0);
    A.set(1, 0, 4.0);
    A.set(1, 1, 5.0);
    A.set(1, 2, 6.0);

    var B = try mat.Matrix(f64, 3, 2).init(allocator);
    defer B.deinit();

    const A_expr = MatrixExpr(f64, 2, 3).init(A);
    A_expr.transpose().evalInto(&B);

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), B.at(0, 0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), B.at(0, 1), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), B.at(1, 0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), B.at(1, 1), 1e-10);
}
