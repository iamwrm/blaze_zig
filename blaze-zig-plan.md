# Zig-Blaze: High-Performance Linear Algebra Library

A Zig implementation of the Blaze C++ library's core concepts, leveraging comptime expression building for zero-overhead abstractions and seamless BLAS/MKL integration.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Zig Language Feature Assessment](#zig-language-feature-assessment)
4. [CBLAS Bindings](#cblas-bindings)
5. [Core Types](#core-types)
6. [Expression System](#expression-system)
7. [BLAS Level 1 Operations](#blas-level-1-operations)
8. [BLAS Level 2 Operations](#blas-level-2-operations)
9. [BLAS Level 3 Operations](#blas-level-3-operations)
10. [SIMD Kernels](#simd-kernels)
11. [Usage Examples](#usage-examples)
12. [Blaze C++ Reference](#blaze-c-reference)
13. [Project Structure](#project-structure)

---

## Overview

### Goals

- **Zero-overhead abstractions**: Comptime expression building eliminates runtime overhead
- **BLAS integration**: Seamless calls to MKL/OpenBLAS for large operations
- **SIMD optimization**: Native `@Vector` operations for small/medium sizes
- **Type safety**: Compile-time dimension checking
- **Ergonomic API**: Fluent expression chaining despite no operator overloading

### Design Philosophy

Blaze C++ uses expression templates with operator overloading:
```cpp
// C++ Blaze
Matrix C = A * B + D;  // Lazy expression, optimized evaluation
```

Zig lacks operator overloading, so we use **Comptime Expression Building**:
```zig
// Zig-Blaze
const expr = comptime A.mul(B).add(D);
evaluate(expr, &C);  // Optimized evaluation
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Zig-Blaze Architecture                           │
├─────────────────────────────────────────────────────────────────────────┤
│  User API Layer                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────┐ │
│  │   Vector    │  │   Matrix    │  │  Comptime Expression Builder    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│  Expression Types                                                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │  L1Expr │ │  L2Expr │ │  L3Expr │ │ ScaleEx │ │ TransEx │          │
│  │ (vec·vec)│ │(mat·vec)│ │(mat·mat)│ │  (αX)   │ │  (X^T)  │          │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
├─────────────────────────────────────────────────────────────────────────┤
│  Comptime Evaluator (Pattern Matching & Kernel Selection)               │
├─────────────────────────────────────────────────────────────────────────┤
│  Kernel Dispatch                                                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │   BLAS Kernels   │  │   SIMD Kernels   │  │  Scalar Fallback │      │
│  │  (MKL/OpenBLAS)  │  │  (@Vector ops)   │  │                  │      │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘      │
├─────────────────────────────────────────────────────────────────────────┤
│  CBLAS Bindings (@cImport)                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │  sdot   │ │  saxpy  │ │  sgemv  │ │  sgemm  │ │  strmv  │  ...     │
│  │  ddot   │ │  daxpy  │ │  dgemv  │ │  dgemm  │ │  dtrmv  │          │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Call Flow

```
User Code: const expr = comptime A.mul(B).add(C);
                ↓
    Expression object created at comptime
                ↓
    evaluate(expr, &result)
                ↓
    Comptime pattern analysis → selectKernel()
                ↓
    ├── Large matrix? → BLAS kernel (cblas_dgemm, etc.)
    │
    └── Small matrix? → SIMD kernel (@Vector operations)
                ↓
    Result written to output
```

---

## Zig Language Feature Assessment

### Features Zig HAS (Ready)

| Feature | Zig Capability | Blaze Requirement |
|---------|---------------|-------------------|
| **SIMD/Vectors** | `@Vector(N, T)` - first-class vector type | Blaze's `SIMDType` |
| **Comptime Generics** | `comptime T: type` - compile-time duck typing | C++ templates |
| **C Interop** | `@cImport` - seamless C header importing | MKL/BLAS calls |
| **Memory Alignment** | `align(N)` on types, variables, functions | SIMD-aligned storage |
| **Inline Functions** | `inline fn` | Aggressive inlining |
| **Complex Numbers** | `std.math.complex.Complex(T)` | Complex type support |
| **Low-level Control** | Direct pointer manipulation, `@ptrCast` | BLAS data access |

### Critical Difference: No Operator Overloading

Zig explicitly states: *"There is no operator overloading."*

**Solution: Comptime Expression Building**

```zig
// Instead of: C = A * B + D
// We write:
const expr = comptime A.mul(B).add(D);
evaluate(expr, &C);
```

Benefits:
- Zero runtime overhead (expression tree resolved at comptime)
- Pattern matching for kernel optimization
- Type safety with compile-time dimension checking

---

## CBLAS Bindings

### File: `src/blas/cblas.zig`

```zig
const std = @import("std");

pub const c = @cImport({
    @cDefine("HAVE_CBLAS", "1");
    @cInclude("cblas.h");
});

// Type aliases
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
        else => @compileError("dot: unsupported type"),
    };
}

/// AXPY: y = α*x + y
pub fn axpy(comptime T: type, n: c_int, alpha: T, x: [*]const T, incX: c_int, y: [*]T, incY: c_int) void {
    switch (T) {
        f32 => c.cblas_saxpy(n, alpha, x, incX, y, incY),
        f64 => c.cblas_daxpy(n, alpha, x, incX, y, incY),
        else => @compileError("axpy: unsupported type"),
    }
}

/// SCAL: x = α*x
pub fn scal(comptime T: type, n: c_int, alpha: T, x: [*]T, incX: c_int) void {
    switch (T) {
        f32 => c.cblas_sscal(n, alpha, x, incX),
        f64 => c.cblas_dscal(n, alpha, x, incX),
        else => @compileError("scal: unsupported type"),
    }
}

/// NRM2: ||x||₂
pub fn nrm2(comptime T: type, n: c_int, x: [*]const T, incX: c_int) T {
    return switch (T) {
        f32 => c.cblas_snrm2(n, x, incX),
        f64 => c.cblas_dnrm2(n, x, incX),
        else => @compileError("nrm2: unsupported type"),
    };
}

/// ASUM: Σ|xᵢ|
pub fn asum(comptime T: type, n: c_int, x: [*]const T, incX: c_int) T {
    return switch (T) {
        f32 => c.cblas_sasum(n, x, incX),
        f64 => c.cblas_dasum(n, x, incX),
        else => @compileError("asum: unsupported type"),
    };
}

/// IAMAX: index of max |xᵢ|
pub fn iamax(comptime T: type, n: c_int, x: [*]const T, incX: c_int) c_int {
    return switch (T) {
        f32 => @intCast(c.cblas_isamax(n, x, incX)),
        f64 => @intCast(c.cblas_idamax(n, x, incX)),
        else => @compileError("iamax: unsupported type"),
    };
}

/// COPY: y = x
pub fn copy(comptime T: type, n: c_int, x: [*]const T, incX: c_int, y: [*]T, incY: c_int) void {
    switch (T) {
        f32 => c.cblas_scopy(n, x, incX, y, incY),
        f64 => c.cblas_dcopy(n, x, incX, y, incY),
        else => @compileError("copy: unsupported type"),
    }
}

/// SWAP: x ↔ y
pub fn swap(comptime T: type, n: c_int, x: [*]T, incX: c_int, y: [*]T, incY: c_int) void {
    switch (T) {
        f32 => c.cblas_sswap(n, x, incX, y, incY),
        f64 => c.cblas_dswap(n, x, incX, y, incY),
        else => @compileError("swap: unsupported type"),
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
        else => @compileError("gemv: unsupported type"),
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
        else => @compileError("trmv: unsupported type"),
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
        else => @compileError("trsv: unsupported type"),
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
        else => @compileError("symv: unsupported type"),
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
        else => @compileError("ger: unsupported type"),
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
        else => @compileError("syr: unsupported type"),
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
        else => @compileError("gemm: unsupported type"),
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
        else => @compileError("trmm: unsupported type"),
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
        else => @compileError("trsm: unsupported type"),
    }
}
```

---

## Core Types

### Vector Type: `src/vector.zig`

```zig
const std = @import("std");
const cblas = @import("blas/cblas.zig");
const expr = @import("expr.zig");

pub fn Vector(comptime T: type, comptime Size: usize) type {
    return struct {
        const Self = @This();
        pub const ElementType = T;
        pub const size = Size;
        pub const is_vector = true;
        pub const is_matrix = false;

        // SIMD configuration
        const VecLen = std.simd.suggestVectorLength(T) orelse 4;
        pub const SimdVec = @Vector(VecLen, T);

        data: *align(64) [Size]T,
        allocator: ?std.mem.Allocator = null,

        // ─────────────────────────────────────────────────────────────────
        // Initialization
        // ─────────────────────────────────────────────────────────────────

        pub fn init(allocator: std.mem.Allocator) !Self {
            const data = try allocator.alignedAlloc(T, 64, Size);
            return .{
                .data = @ptrCast(data.ptr),
                .allocator = allocator,
            };
        }

        pub fn initWithValue(allocator: std.mem.Allocator, value: T) !Self {
            var self = try init(allocator);
            @memset(self.data, value);
            return self;
        }

        pub fn deinit(self: *Self) void {
            if (self.allocator) |alloc| {
                const slice: []align(64) T = @as([*]align(64) T, @ptrCast(self.data))[0..Size];
                alloc.free(slice);
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // BLAS Level 1 Expression Builders
        // ─────────────────────────────────────────────────────────────────

        /// Dot product: x · y
        pub fn dot(self: Self, other: Self) expr.DotExpr(Self, Self) {
            return .{ .lhs = self, .rhs = other };
        }

        /// Scaled addition: y + α*x (AXPY pattern)
        pub fn axpy(self: Self, alpha: T, x: Self) expr.AxpyExpr(Self, T) {
            return .{ .y = self, .alpha = alpha, .x = x };
        }

        /// Scale: α*x
        pub fn scale(self: Self, alpha: T) expr.ScaleExpr(Self, T) {
            return .{ .operand = self, .alpha = alpha };
        }

        /// Add: x + y
        pub fn add(self: Self, other: anytype) expr.Expr(.add, Self, @TypeOf(other)) {
            return .{ .lhs = self, .rhs = other };
        }

        /// Subtract: x - y
        pub fn sub(self: Self, other: anytype) expr.Expr(.sub, Self, @TypeOf(other)) {
            return .{ .lhs = self, .rhs = other };
        }

        /// Euclidean norm: ||x||₂
        pub fn norm2(self: Self) expr.Norm2Expr(Self) {
            return .{ .operand = self };
        }

        /// Sum of absolute values: Σ|xᵢ|
        pub fn asum(self: Self) expr.AsumExpr(Self) {
            return .{ .operand = self };
        }

        /// Index of max absolute value
        pub fn iamax(self: Self) expr.IamaxExpr(Self) {
            return .{ .operand = self };
        }

        // ─────────────────────────────────────────────────────────────────
        // Data Access
        // ─────────────────────────────────────────────────────────────────

        pub fn ptr(self: Self) [*]T {
            return @ptrCast(self.data);
        }

        pub fn constPtr(self: Self) [*]const T {
            return @ptrCast(self.data);
        }

        pub fn at(self: Self, i: usize) T {
            return self.data[i];
        }

        pub fn set(self: Self, i: usize, value: T) void {
            self.data[i] = value;
        }
    };
}
```

### Matrix Type: `src/matrix.zig`

```zig
const std = @import("std");
const cblas = @import("blas/cblas.zig");
const expr = @import("expr.zig");
const vec = @import("vector.zig");

pub fn Matrix(comptime T: type, comptime Rows: usize, comptime Cols: usize) type {
    return struct {
        const Self = @This();
        pub const ElementType = T;
        pub const rows = Rows;
        pub const cols = Cols;
        pub const is_vector = false;
        pub const is_matrix = true;

        // SIMD configuration
        const VecLen = std.simd.suggestVectorLength(T) orelse 4;
        pub const SimdVec = @Vector(VecLen, T);

        data: *align(64) [Rows * Cols]T,
        allocator: ?std.mem.Allocator = null,

        // ─────────────────────────────────────────────────────────────────
        // Initialization
        // ─────────────────────────────────────────────────────────────────

        pub fn init(allocator: std.mem.Allocator) !Self {
            const data = try allocator.alignedAlloc(T, 64, Rows * Cols);
            return .{
                .data = @ptrCast(data.ptr),
                .allocator = allocator,
            };
        }

        pub fn initIdentity(allocator: std.mem.Allocator) !Self {
            var self = try init(allocator);
            @memset(self.data, 0);
            const min_dim = @min(Rows, Cols);
            for (0..min_dim) |i| {
                self.data[i * Cols + i] = 1;
            }
            return self;
        }

        pub fn deinit(self: *Self) void {
            if (self.allocator) |alloc| {
                const slice: []align(64) T = @as([*]align(64) T, @ptrCast(self.data))[0 .. Rows * Cols];
                alloc.free(slice);
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // BLAS Level 2 Expression Builders (Matrix-Vector)
        // ─────────────────────────────────────────────────────────────────

        /// Matrix-vector multiply: A*x → GEMV
        pub fn mulVec(self: Self, x: vec.Vector(T, Cols)) expr.GemvExpr(Self, vec.Vector(T, Cols)) {
            return .{ .A = self, .x = x };
        }

        /// Triangular matrix-vector: A*x → TRMV
        pub fn trmv(self: Self, x: vec.Vector(T, Cols)) expr.TrmvExpr(Self, vec.Vector(T, Cols)) {
            return .{ .A = self, .x = x };
        }

        /// Symmetric matrix-vector: A*x → SYMV
        pub fn symv(self: Self, x: vec.Vector(T, Cols)) expr.SymvExpr(Self, vec.Vector(T, Cols)) {
            return .{ .A = self, .x = x };
        }

        /// Rank-1 update: A + α*x*yᵀ → GER
        pub fn ger(self: Self, x: vec.Vector(T, Rows), y: vec.Vector(T, Cols)) expr.GerExpr(vec.Vector(T, Rows), vec.Vector(T, Cols), Self) {
            return .{ .x = x, .y = y, .A = self };
        }

        /// Symmetric rank-1 update: A + α*x*xᵀ → SYR
        pub fn syr(self: Self, x: vec.Vector(T, Rows)) expr.SyrExpr(vec.Vector(T, Rows), Self) {
            return .{ .x = x, .A = self };
        }

        // ─────────────────────────────────────────────────────────────────
        // BLAS Level 3 Expression Builders (Matrix-Matrix)
        // ─────────────────────────────────────────────────────────────────

        /// Matrix-matrix multiply: A*B → GEMM
        pub fn mul(self: Self, other: anytype) expr.GemmExpr(Self, @TypeOf(other)) {
            return .{ .A = self, .B = other };
        }

        /// Transpose: Aᵀ
        pub fn T(self: Self) expr.TransposeExpr(Self) {
            return .{ .operand = self };
        }

        /// Scale: α*A
        pub fn scale(self: Self, alpha: T) expr.ScaleExpr(Self, T) {
            return .{ .operand = self, .alpha = alpha };
        }

        /// Add: A + B
        pub fn add(self: Self, other: anytype) expr.Expr(.add, Self, @TypeOf(other)) {
            return .{ .lhs = self, .rhs = other };
        }

        /// Subtract: A - B
        pub fn sub(self: Self, other: anytype) expr.Expr(.sub, Self, @TypeOf(other)) {
            return .{ .lhs = self, .rhs = other };
        }

        // ─────────────────────────────────────────────────────────────────
        // Data Access
        // ─────────────────────────────────────────────────────────────────

        pub fn ptr(self: Self) [*]T {
            return @ptrCast(self.data);
        }

        pub fn constPtr(self: Self) [*]const T {
            return @ptrCast(self.data);
        }

        pub fn at(self: Self, row: usize, col: usize) T {
            return self.data[row * Cols + col];
        }

        pub fn set(self: Self, row: usize, col: usize, value: T) void {
            self.data[row * Cols + col] = value;
        }
    };
}
```

---

## Expression System

### File: `src/expr.zig`

```zig
const std = @import("std");
const cblas = @import("blas/cblas.zig");
const simd = @import("kernels/simd.zig");

pub const ExprKind = enum {
    // Level 1 (vector-vector)
    dot,
    axpy,
    scale,
    add,
    sub,
    norm2,
    asum,
    iamax,

    // Level 2 (matrix-vector)
    gemv,
    trmv,
    symv,
    trsv,
    ger,
    syr,

    // Level 3 (matrix-matrix)
    gemm,
    trmm,
    trsm,

    // Modifiers
    transpose,
};

// =============================================================================
// COMPTIME KERNEL SELECTION
// =============================================================================

pub const EvalStrategy = enum {
    blas_l1,
    blas_l2,
    blas_l3,
    simd_vectorized,
    scalar_fallback,
};

pub fn shouldUseBlas(comptime T: type, comptime size: usize, comptime op: ExprKind) bool {
    // Type must be BLAS compatible
    if (T != f32 and T != f64) return false;

    // Size thresholds for different operations
    return switch (op) {
        .dot => size >= 64,
        .axpy => size >= 64,
        .scale => size >= 128,
        .norm2 => size >= 64,
        .asum => size >= 64,
        .gemv => size >= 32,
        .trmv => size >= 32,
        .symv => size >= 32,
        .ger => size >= 64,
        .gemm => size >= 64,
        else => false,
    };
}

fn isBLASCompatible(comptime T: type) bool {
    return T == f32 or T == f64;
}

// =============================================================================
// BLAS LEVEL 1 EXPRESSION TYPES
// =============================================================================

/// DOT: x · y → scalar
pub fn DotExpr(comptime L: type, comptime R: type) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.dot;
        pub const ResultType = L.ElementType;
        pub const is_scalar_result = true;

        lhs: L,
        rhs: R,

        /// Evaluate dot product
        pub fn eval(self: Self) ResultType {
            const T = L.ElementType;
            const n = L.size;

            if (comptime shouldUseBlas(T, n, .dot)) {
                return cblas.dot(T, @intCast(n), self.lhs.constPtr(), 1, self.rhs.constPtr(), 1);
            } else {
                return simd.dot(T, n, self.lhs.data, self.rhs.data);
            }
        }
    };
}

/// AXPY: y = α*x + y
pub fn AxpyExpr(comptime V: type, comptime S: type) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.axpy;
        pub const ResultType = V;
        pub const ElementType = V.ElementType;

        y: V,
        alpha: S,
        x: V,

        /// Chain: (y + α₁*x₁) + α₂*x₂
        pub fn axpy(self: Self, alpha2: S, x2: V) AxpyChainExpr(Self, V, S) {
            return .{ .base = self, .alpha = alpha2, .x = x2 };
        }

        /// Chain: (y + α*x) + z
        pub fn add(self: Self, other: anytype) Expr(.add, Self, @TypeOf(other)) {
            return .{ .lhs = self, .rhs = other };
        }

        pub fn eval(self: Self, result: *V) void {
            const T = V.ElementType;
            const n = V.size;

            // Copy y to result first
            @memcpy(result.data, self.y.data);

            if (comptime shouldUseBlas(T, n, .axpy)) {
                cblas.axpy(T, @intCast(n), self.alpha, self.x.constPtr(), 1, result.ptr(), 1);
            } else {
                simd.axpy(T, n, self.alpha, self.x.data, result.data);
            }
        }
    };
}

/// SCAL: α*x
pub fn ScaleExpr(comptime V: type, comptime S: type) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.scale;
        pub const ResultType = V;
        pub const ElementType = V.ElementType;
        pub const is_vector = V.is_vector;
        pub const is_matrix = V.is_matrix;

        // Forward dimension info
        pub const size = if (@hasDecl(V, "size")) V.size else V.rows * V.cols;
        pub const rows = if (@hasDecl(V, "rows")) V.rows else V.size;
        pub const cols = if (@hasDecl(V, "cols")) V.cols else 1;

        operand: V,
        alpha: S,

        /// Chain: (α*x) + y
        pub fn add(self: Self, other: anytype) Expr(.add, Self, @TypeOf(other)) {
            return .{ .lhs = self, .rhs = other };
        }

        /// Chain: (α*x) - y
        pub fn sub(self: Self, other: anytype) Expr(.sub, Self, @TypeOf(other)) {
            return .{ .lhs = self, .rhs = other };
        }

        /// Chain: (α*x) · y → α*(x·y)
        pub fn dot(self: Self, other: V) ScaledDotExpr(V, S) {
            return .{ .alpha = self.alpha, .x = self.operand, .y = other };
        }

        pub fn eval(self: Self, result: *V) void {
            const T = V.ElementType;
            const n = size;

            @memcpy(result.data, self.operand.data);

            if (comptime shouldUseBlas(T, n, .scale)) {
                cblas.scal(T, @intCast(n), self.alpha, result.ptr(), 1);
            } else {
                simd.scal(T, n, self.alpha, result.data);
            }
        }
    };
}

/// NRM2: ||x||₂
pub fn Norm2Expr(comptime V: type) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.norm2;
        pub const ResultType = V.ElementType;
        pub const is_scalar_result = true;

        operand: V,

        pub fn eval(self: Self) ResultType {
            const T = V.ElementType;
            const n = V.size;

            if (comptime shouldUseBlas(T, n, .norm2)) {
                return cblas.nrm2(T, @intCast(n), self.operand.constPtr(), 1);
            } else {
                return simd.nrm2(T, n, self.operand.data);
            }
        }
    };
}

/// ASUM: Σ|xᵢ|
pub fn AsumExpr(comptime V: type) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.asum;
        pub const ResultType = V.ElementType;
        pub const is_scalar_result = true;

        operand: V,

        pub fn eval(self: Self) ResultType {
            const T = V.ElementType;
            const n = V.size;

            if (comptime shouldUseBlas(T, n, .asum)) {
                return cblas.asum(T, @intCast(n), self.operand.constPtr(), 1);
            } else {
                return simd.asum(T, n, self.operand.data);
            }
        }
    };
}

/// IAMAX: index of max |xᵢ|
pub fn IamaxExpr(comptime V: type) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.iamax;
        pub const ResultType = usize;
        pub const is_scalar_result = true;

        operand: V,

        pub fn eval(self: Self) ResultType {
            const T = V.ElementType;
            const n = V.size;

            return @intCast(cblas.iamax(T, @intCast(n), self.operand.constPtr(), 1));
        }
    };
}

// =============================================================================
// BLAS LEVEL 2 EXPRESSION TYPES
// =============================================================================

/// GEMV: y = α*A*x + β*y
pub fn GemvExpr(comptime M: type, comptime V: type) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.gemv;
        pub const ElementType = M.ElementType;
        pub const ResultRows = M.rows;
        pub const ResultType = @import("vector.zig").Vector(ElementType, ResultRows);

        A: M,
        x: V,
        alpha: ElementType = 1,
        beta: ElementType = 0,
        y: ?ResultType = null,
        transA: bool = false,

        /// Set alpha: α*(A*x)
        pub fn setAlpha(self: Self, alpha_new: ElementType) Self {
            var result = self;
            result.alpha = alpha_new;
            return result;
        }

        /// Chain: (A*x) + y → GEMV with β=1
        pub fn add(self: Self, y_vec: ResultType) Self {
            var result = self;
            result.beta = 1;
            result.y = y_vec;
            return result;
        }

        /// Chain: α*(A*x)
        pub fn scale(self: Self, alpha_new: ElementType) Self {
            var result = self;
            result.alpha *= alpha_new;
            return result;
        }

        /// Transpose A
        pub fn T(self: Self) Self {
            var result = self;
            result.transA = !result.transA;
            return result;
        }

        pub fn eval(self: Self, result: *ResultType) void {
            const T = ElementType;
            const m: c_int = @intCast(M.rows);
            const n: c_int = @intCast(M.cols);

            // Initialize result with y if β ≠ 0
            if (self.y) |y_vec| {
                @memcpy(result.data, y_vec.data);
            } else if (self.beta != 0) {
                @memset(result.data, 0);
            }

            const transA: cblas.Transpose = if (self.transA) .Trans else .NoTrans;

            if (comptime shouldUseBlas(T, M.rows * M.cols, .gemv)) {
                cblas.gemv(
                    T,
                    .RowMajor,
                    transA,
                    m,
                    n,
                    self.alpha,
                    self.A.constPtr(),
                    n,
                    self.x.constPtr(),
                    1,
                    self.beta,
                    result.ptr(),
                    1,
                );
            } else {
                simd.gemv(T, M.rows, M.cols, self.alpha, self.A.data, self.x.data, self.beta, result.data);
            }
        }
    };
}

/// TRMV: x = A*x (triangular)
pub fn TrmvExpr(comptime M: type, comptime V: type) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.trmv;
        pub const ElementType = M.ElementType;
        pub const ResultType = V;

        A: M,
        x: V,
        uplo: cblas.UpLo = .Upper,
        diag: cblas.Diag = .NonUnit,
        transA: bool = false,

        /// Set to lower triangular
        pub fn lower(self: Self) Self {
            var result = self;
            result.uplo = .Lower;
            return result;
        }

        /// Set to upper triangular
        pub fn upper(self: Self) Self {
            var result = self;
            result.uplo = .Upper;
            return result;
        }

        /// Set unit diagonal
        pub fn unitDiag(self: Self) Self {
            var result = self;
            result.diag = .Unit;
            return result;
        }

        /// Transpose A
        pub fn T(self: Self) Self {
            var result = self;
            result.transA = !result.transA;
            return result;
        }

        pub fn eval(self: Self, result: *ResultType) void {
            const T = ElementType;
            const n: c_int = @intCast(M.rows);

            @memcpy(result.data, self.x.data);

            const transA: cblas.Transpose = if (self.transA) .Trans else .NoTrans;

            cblas.trmv(T, .RowMajor, self.uplo, transA, self.diag, n, self.A.constPtr(), n, result.ptr(), 1);
        }
    };
}

/// SYMV: y = α*A*x + β*y (symmetric)
pub fn SymvExpr(comptime M: type, comptime V: type) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.symv;
        pub const ElementType = M.ElementType;
        pub const ResultType = V;

        A: M,
        x: V,
        alpha: ElementType = 1,
        beta: ElementType = 0,
        y: ?V = null,
        uplo: cblas.UpLo = .Upper,

        pub fn setAlpha(self: Self, alpha_new: ElementType) Self {
            var result = self;
            result.alpha = alpha_new;
            return result;
        }

        pub fn add(self: Self, y_vec: V) Self {
            var result = self;
            result.beta = 1;
            result.y = y_vec;
            return result;
        }

        pub fn lower(self: Self) Self {
            var result = self;
            result.uplo = .Lower;
            return result;
        }

        pub fn eval(self: Self, result: *ResultType) void {
            const T = ElementType;
            const n: c_int = @intCast(M.rows);

            if (self.y) |y_vec| {
                @memcpy(result.data, y_vec.data);
            }

            cblas.symv(T, .RowMajor, self.uplo, n, self.alpha, self.A.constPtr(), n, self.x.constPtr(), 1, self.beta, result.ptr(), 1);
        }
    };
}

/// GER: A = α*x*yᵀ + A (rank-1 update)
pub fn GerExpr(comptime VX: type, comptime VY: type, comptime M: type) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.ger;
        pub const ElementType = VX.ElementType;
        pub const ResultType = M;

        x: VX,
        y: VY,
        A: M,
        alpha: ElementType = 1,

        pub fn scale(self: Self, alpha_new: ElementType) Self {
            var result = self;
            result.alpha *= alpha_new;
            return result;
        }

        pub fn eval(self: Self, result: *ResultType) void {
            const T = ElementType;
            const m: c_int = @intCast(VX.size);
            const n: c_int = @intCast(VY.size);

            @memcpy(result.data, self.A.data);

            cblas.ger(T, .RowMajor, m, n, self.alpha, self.x.constPtr(), 1, self.y.constPtr(), 1, result.ptr(), n);
        }
    };
}

/// SYR: A = α*x*xᵀ + A (symmetric rank-1 update)
pub fn SyrExpr(comptime V: type, comptime M: type) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.syr;
        pub const ElementType = V.ElementType;
        pub const ResultType = M;

        x: V,
        A: M,
        alpha: ElementType = 1,
        uplo: cblas.UpLo = .Upper,

        pub fn scale(self: Self, alpha_new: ElementType) Self {
            var result = self;
            result.alpha *= alpha_new;
            return result;
        }

        pub fn lower(self: Self) Self {
            var result = self;
            result.uplo = .Lower;
            return result;
        }

        pub fn eval(self: Self, result: *ResultType) void {
            const T = ElementType;
            const n: c_int = @intCast(V.size);

            @memcpy(result.data, self.A.data);

            cblas.syr(T, .RowMajor, self.uplo, n, self.alpha, self.x.constPtr(), 1, result.ptr(), n);
        }
    };
}

// =============================================================================
// BLAS LEVEL 3 EXPRESSION TYPES
// =============================================================================

/// GEMM: C = α*A*B + β*C
pub fn GemmExpr(comptime MA: type, comptime MB: type) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.gemm;
        pub const ElementType = MA.ElementType;
        pub const ResultRows = MA.rows;
        pub const ResultCols = MB.cols;
        pub const ResultType = @import("matrix.zig").Matrix(ElementType, ResultRows, ResultCols);

        A: MA,
        B: MB,
        alpha: ElementType = 1,
        beta: ElementType = 0,
        C: ?ResultType = null,
        transA: bool = false,
        transB: bool = false,

        /// Chain: (A*B) + C → GEMM with β=1
        pub fn add(self: Self, c_mat: ResultType) Self {
            var result = self;
            result.beta = 1;
            result.C = c_mat;
            return result;
        }

        /// Chain: α*(A*B)
        pub fn scale(self: Self, alpha_new: ElementType) Self {
            var result = self;
            result.alpha *= alpha_new;
            return result;
        }

        /// Chain: (A*B)*D
        pub fn mul(self: Self, other: anytype) GemmExpr(Self, @TypeOf(other)) {
            return .{ .A = self, .B = other };
        }

        pub fn eval(self: Self, result: *ResultType) void {
            const T = ElementType;
            const m: c_int = @intCast(MA.rows);
            const n: c_int = @intCast(MB.cols);
            const k: c_int = @intCast(MA.cols);

            if (self.C) |c_mat| {
                @memcpy(result.data, c_mat.data);
            }

            const transA: cblas.Transpose = if (self.transA) .Trans else .NoTrans;
            const transB: cblas.Transpose = if (self.transB) .Trans else .NoTrans;

            if (comptime shouldUseBlas(T, MA.rows * MB.cols, .gemm)) {
                cblas.gemm(
                    T,
                    .RowMajor,
                    transA,
                    transB,
                    m,
                    n,
                    k,
                    self.alpha,
                    self.A.constPtr(),
                    k,
                    self.B.constPtr(),
                    n,
                    self.beta,
                    result.ptr(),
                    n,
                );
            } else {
                simd.gemm(T, MA.rows, MB.cols, MA.cols, self.alpha, self.A.data, self.B.data, self.beta, result.data);
            }
        }
    };
}

/// Transpose expression
pub fn TransposeExpr(comptime M: type) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.transpose;
        pub const ElementType = M.ElementType;
        pub const rows = M.cols;
        pub const cols = M.rows;
        pub const ResultType = @import("matrix.zig").Matrix(ElementType, rows, cols);

        operand: M,

        pub fn eval(self: Self, result: *ResultType) void {
            for (0..M.rows) |i| {
                for (0..M.cols) |j| {
                    result.data[j * M.rows + i] = self.operand.data[i * M.cols + j];
                }
            }
        }
    };
}

// =============================================================================
// GENERIC EXPRESSION TYPE (for add/sub chaining)
// =============================================================================

pub fn Expr(comptime Kind: ExprKind, comptime L: type, comptime R: type) type {
    return struct {
        const Self = @This();
        pub const kind = Kind;
        pub const LHS = L;
        pub const RHS = R;
        pub const ElementType = L.ElementType;

        // Infer result dimensions
        pub const is_vector = if (@hasDecl(L, "is_vector")) L.is_vector else false;
        pub const is_matrix = if (@hasDecl(L, "is_matrix")) L.is_matrix else false;
        pub const size = if (@hasDecl(L, "size")) L.size else L.rows * L.cols;
        pub const rows = if (@hasDecl(L, "rows")) L.rows else L.size;
        pub const cols = if (@hasDecl(L, "cols")) L.cols else 1;

        pub const ResultType = if (is_vector)
            @import("vector.zig").Vector(ElementType, size)
        else
            @import("matrix.zig").Matrix(ElementType, rows, cols);

        lhs: L,
        rhs: R,

        /// Continue chaining: expr + other
        pub fn add(self: Self, other: anytype) Expr(.add, Self, @TypeOf(other)) {
            return .{ .lhs = self, .rhs = other };
        }

        /// Continue chaining: expr - other
        pub fn sub(self: Self, other: anytype) Expr(.sub, Self, @TypeOf(other)) {
            return .{ .lhs = self, .rhs = other };
        }

        /// Continue chaining: α*expr
        pub fn scale(self: Self, alpha: ElementType) ScaleExpr(Self, ElementType) {
            return .{ .operand = self, .alpha = alpha };
        }
    };
}

// =============================================================================
// SCALED DOT (for optimization: α*(x·y))
// =============================================================================

pub fn ScaledDotExpr(comptime V: type, comptime S: type) type {
    return struct {
        const Self = @This();
        pub const ResultType = S;
        pub const is_scalar_result = true;

        alpha: S,
        x: V,
        y: V,

        pub fn eval(self: Self) ResultType {
            const T = V.ElementType;
            const n = V.size;

            const dot_result = if (comptime shouldUseBlas(T, n, .dot))
                cblas.dot(T, @intCast(n), self.x.constPtr(), 1, self.y.constPtr(), 1)
            else
                simd.dot(T, n, self.x.data, self.y.data);

            return self.alpha * dot_result;
        }
    };
}

// =============================================================================
// AXPY CHAIN (for multiple AXPY in sequence)
// =============================================================================

pub fn AxpyChainExpr(comptime Base: type, comptime V: type, comptime S: type) type {
    return struct {
        const Self = @This();
        pub const kind = ExprKind.axpy;
        pub const ResultType = V;
        pub const ElementType = V.ElementType;

        base: Base,
        alpha: S,
        x: V,

        /// Continue chaining
        pub fn axpy(self: Self, alpha2: S, x2: V) AxpyChainExpr(Self, V, S) {
            return .{ .base = self, .alpha = alpha2, .x = x2 };
        }

        pub fn eval(self: Self, result: *V) void {
            const T = V.ElementType;
            const n = V.size;

            // First evaluate base expression
            self.base.eval(result);

            // Then apply this AXPY
            if (comptime shouldUseBlas(T, n, .axpy)) {
                cblas.axpy(T, @intCast(n), self.alpha, self.x.constPtr(), 1, result.ptr(), 1);
            } else {
                simd.axpy(T, n, self.alpha, self.x.data, result.data);
            }
        }
    };
}
```

---

## BLAS Level 1 Operations

### Summary Table

| Operation | BLAS | Expression | Description |
|-----------|------|------------|-------------|
| DOT | `cblas_?dot` | `x.dot(y).eval()` | x · y |
| AXPY | `cblas_?axpy` | `y.axpy(α, x).eval(&z)` | z = y + α*x |
| SCAL | `cblas_?scal` | `x.scale(α).eval(&y)` | y = α*x |
| NRM2 | `cblas_?nrm2` | `x.norm2().eval()` | ‖x‖₂ |
| ASUM | `cblas_?asum` | `x.asum().eval()` | Σ\|xᵢ\| |
| IAMAX | `cblas_i?amax` | `x.iamax().eval()` | argmax \|xᵢ\| |
| COPY | `cblas_?copy` | `x.copyTo(&y)` | y = x |
| SWAP | `cblas_?swap` | `x.swap(&y)` | x ↔ y |

### Usage Examples

```zig
const Vec = @import("zig-blaze").Vector;

var x = try Vec(f64, 1000).initWithValue(allocator, 1.0);
var y = try Vec(f64, 1000).initWithValue(allocator, 2.0);
var z = try Vec(f64, 1000).init(allocator);
defer { x.deinit(); y.deinit(); z.deinit(); }

// Dot product
const s = comptime x.dot(y).eval();  // → cblas_ddot

// AXPY: z = y + 2.0*x
const axpy_expr = comptime y.axpy(2.0, x);
axpy_expr.eval(&z);

// Chained AXPY: z = y + 2.0*x + 0.5*w
const chain = comptime y.axpy(2.0, x).axpy(0.5, w);
chain.eval(&z);

// Euclidean norm
const n = comptime x.norm2().eval();  // → cblas_dnrm2
```

---

## BLAS Level 2 Operations

### Summary Table

| Operation | BLAS | Expression | Description |
|-----------|------|------------|-------------|
| GEMV | `cblas_?gemv` | `A.mulVec(x).eval(&y)` | y = α*A*x + β*y |
| TRMV | `cblas_?trmv` | `A.trmv(x).eval(&y)` | y = A*x (triangular) |
| TRSV | `cblas_?trsv` | `A.trsv(x).eval(&y)` | y = A⁻¹*x (triangular solve) |
| SYMV | `cblas_?symv` | `A.symv(x).eval(&y)` | y = α*A*x + β*y (symmetric) |
| GER | `cblas_?ger` | `A.ger(x, y).eval(&A)` | A = α*x*yᵀ + A |
| SYR | `cblas_?syr` | `A.syr(x).eval(&A)` | A = α*x*xᵀ + A |

### Usage Examples

```zig
const Mat = @import("zig-blaze").Matrix;
const Vec = @import("zig-blaze").Vector;

var A = try Mat(f64, 100, 100).init(allocator);
var x = try Vec(f64, 100).init(allocator);
var y = try Vec(f64, 100).init(allocator);
defer { A.deinit(); x.deinit(); y.deinit(); }

// Basic GEMV: y = A*x
const gemv_expr = comptime A.mulVec(x);
gemv_expr.eval(&y);

// Scaled GEMV: y = 2.0*A*x
const scaled = comptime A.mulVec(x).scale(2.0);
scaled.eval(&y);

// Full GEMV: y = 2.0*A*x + 0.5*y (α=2, β=0.5)
const full_gemv = comptime A.mulVec(x).scale(2.0).add(y.scale(0.5));
// This gets pattern-matched to single cblas_dgemv call

// Triangular: y = A*x (A upper triangular)
const trmv_expr = comptime A.trmv(x);
trmv_expr.eval(&y);

// Lower triangular with unit diagonal
const trmv_lower = comptime A.trmv(x).lower().unitDiag();
trmv_lower.eval(&y);

// Rank-1 update: A = A + 2.0*x*yᵀ
const ger_expr = comptime A.ger(x, y).scale(2.0);
ger_expr.eval(&A);
```

---

## BLAS Level 3 Operations

### Summary Table

| Operation | BLAS | Expression | Description |
|-----------|------|------------|-------------|
| GEMM | `cblas_?gemm` | `A.mul(B).eval(&C)` | C = α*A*B + β*C |
| TRMM | `cblas_?trmm` | `A.trmm(B).eval(&C)` | C = α*A*B (A triangular) |
| TRSM | `cblas_?trsm` | `A.trsm(B).eval(&C)` | C = α*A⁻¹*B (triangular solve) |

### Usage Examples

```zig
const Mat = @import("zig-blaze").Matrix;

var A = try Mat(f64, 100, 200).init(allocator);
var B = try Mat(f64, 200, 150).init(allocator);
var C = try Mat(f64, 100, 150).init(allocator);
var D = try Mat(f64, 100, 150).init(allocator);
defer { A.deinit(); B.deinit(); C.deinit(); D.deinit(); }

// Basic GEMM: C = A*B
const gemm_expr = comptime A.mul(B);
gemm_expr.eval(&C);

// Scaled: C = 2.0*A*B
const scaled = comptime A.mul(B).scale(2.0);
scaled.eval(&C);

// Full GEMM: C = 2.0*A*B + 0.5*D
const full = comptime A.mul(B).scale(2.0).add(D.scale(0.5));
full.eval(&C);

// Chained multiply: C = A*B*E (evaluates left-to-right)
const chain = comptime A.mul(B).mul(E);
chain.eval(&C);
```

---

## SIMD Kernels

### File: `src/kernels/simd.zig`

```zig
const std = @import("std");

/// SIMD dot product
pub fn dot(comptime T: type, comptime n: usize, x: *const [n]T, y: *const [n]T) T {
    const VecLen = std.simd.suggestVectorLength(T) orelse 4;
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
    const VecLen = std.simd.suggestVectorLength(T) orelse 4;
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
    const VecLen = std.simd.suggestVectorLength(T) orelse 4;
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
    const VecLen = std.simd.suggestVectorLength(T) orelse 4;
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

/// SIMD gemv: y = α*A*x + β*y
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
    const VecLen = std.simd.suggestVectorLength(T) orelse 4;
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
    const BLOCK = 64;
    const VecLen = std.simd.suggestVectorLength(T) orelse 4;
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
```

---

## Usage Examples

### Complete Example Program

```zig
const std = @import("std");
const blaze = @import("zig-blaze");
const Vector = blaze.Vector;
const Matrix = blaze.Matrix;
const evaluate = blaze.evaluate;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // ═══════════════════════════════════════════════════════════════════
    // BLAS Level 1: Vector-Vector Operations
    // ═══════════════════════════════════════════════════════════════════

    var x = try Vector(f64, 1000).initWithValue(allocator, 1.0);
    var y = try Vector(f64, 1000).initWithValue(allocator, 2.0);
    var z = try Vector(f64, 1000).init(allocator);
    var w = try Vector(f64, 1000).initWithValue(allocator, 0.5);
    defer {
        x.deinit();
        y.deinit();
        z.deinit();
        w.deinit();
    }

    // Dot product: s = x · y
    const dot_result = x.dot(y).eval();
    std.debug.print("Dot product: {d}\n", .{dot_result});

    // AXPY: z = y + 2.0*x
    y.axpy(2.0, x).eval(&z);
    std.debug.print("AXPY result[0]: {d}\n", .{z.at(0)});

    // Chained AXPY: z = y + 2.0*x + 0.5*w
    y.axpy(2.0, x).axpy(0.5, w).eval(&z);

    // Euclidean norm: ||x||₂
    const norm = x.norm2().eval();
    std.debug.print("Norm: {d}\n", .{norm});

    // Scaled dot: 2.0 * (x · y)
    const scaled_dot = x.scale(2.0).dot(y).eval();
    std.debug.print("Scaled dot: {d}\n", .{scaled_dot});

    // ═══════════════════════════════════════════════════════════════════
    // BLAS Level 2: Matrix-Vector Operations
    // ═══════════════════════════════════════════════════════════════════

    var A = try Matrix(f64, 100, 100).init(allocator);
    var vec_in = try Vector(f64, 100).initWithValue(allocator, 1.0);
    var vec_out = try Vector(f64, 100).init(allocator);
    defer {
        A.deinit();
        vec_in.deinit();
        vec_out.deinit();
    }

    // Fill A with some values
    for (0..100) |i| {
        for (0..100) |j| {
            A.set(i, j, if (i == j) 2.0 else 0.1);
        }
    }

    // GEMV: vec_out = A * vec_in
    A.mulVec(vec_in).eval(&vec_out);
    std.debug.print("GEMV result[0]: {d}\n", .{vec_out.at(0)});

    // Scaled GEMV: vec_out = 2.0 * A * vec_in
    A.mulVec(vec_in).scale(2.0).eval(&vec_out);

    // Full GEMV pattern: vec_out = 2.0*A*vec_in + 0.5*vec_out
    // This fuses into single cblas_dgemv call with α=2.0, β=0.5
    A.mulVec(vec_in).setAlpha(2.0).add(vec_out).eval(&vec_out);

    // Triangular MV (upper triangular)
    A.trmv(vec_in).eval(&vec_out);

    // Lower triangular with unit diagonal
    A.trmv(vec_in).lower().unitDiag().eval(&vec_out);

    // ═══════════════════════════════════════════════════════════════════
    // BLAS Level 3: Matrix-Matrix Operations
    // ═══════════════════════════════════════════════════════════════════

    var M1 = try Matrix(f64, 100, 200).init(allocator);
    var M2 = try Matrix(f64, 200, 150).init(allocator);
    var M3 = try Matrix(f64, 100, 150).init(allocator);
    defer {
        M1.deinit();
        M2.deinit();
        M3.deinit();
    }

    // GEMM: M3 = M1 * M2
    M1.mul(M2).eval(&M3);

    // Scaled GEMM: M3 = 2.0 * M1 * M2
    M1.mul(M2).scale(2.0).eval(&M3);

    // Full GEMM: M3 = 2.0*M1*M2 + 0.5*M3
    M1.mul(M2).scale(2.0).add(M3.scale(0.5)).eval(&M3);

    std.debug.print("All operations completed successfully!\n", .{});
}
```

---

## Blaze C++ Reference

This section documents the original Blaze C++ implementation for reference.

### Expression Template System

**File:** `blaze/math/expressions/DMatDMatMultExpr.h`

The `operator*` creates a lazy expression:

```cpp
template< typename MT1, typename MT2 >
inline decltype(auto)
   operator*( const DenseMatrix<MT1,false>& lhs, const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   if( (*lhs).columns() != (*rhs).rows() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   using ReturnType = const DMatDMatMultExpr<MT1,MT2,false,false,false,false>;
   return ReturnType( *lhs, *rhs );
}
```

### BLAS Kernel Selection

**File:** `blaze/math/expressions/DMatDMatMultExpr.h` (lines 213-225)

```cpp
template< typename T1, typename T2, typename T3 >
static constexpr bool UseBlasKernel_v =
   ( BLAZE_BLAS_MODE && BLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION &&
     !SYM && !HERM && !LOW && !UPP &&
     IsContiguous_v<T1> && HasMutableDataAccess_v<T1> &&
     IsContiguous_v<T2> && HasConstDataAccess_v<T2> &&
     IsContiguous_v<T3> && HasConstDataAccess_v<T3> &&
     T1::simdEnabled && T2::simdEnabled && T3::simdEnabled &&
     IsBLASCompatible_v<ElementType_t<T1>> &&
     IsSame_v< ElementType_t<T1>, ElementType_t<T2> > &&
     IsSame_v< ElementType_t<T1>, ElementType_t<T3> > );
```

### CBLAS GEMV Wrapper

**File:** `blaze/math/blas/cblas/gemv.h`

```cpp
inline void gemv( CBLAS_ORDER order, CBLAS_TRANSPOSE transA, blas_int_t m, blas_int_t n,
                  double alpha, const double* A, blas_int_t lda, const double* x,
                  blas_int_t incX, double beta, double* y, blas_int_t incY )
{
   cblas_dgemv( order, transA, m, n, alpha, A, lda, x, incX, beta, y, incY );
}
```

### CBLAS GEMM Wrapper

**File:** `blaze/math/blas/cblas/gemm.h`

```cpp
inline void gemm( CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                  blas_int_t m, blas_int_t n, blas_int_t k, double alpha,
                  const double* A, blas_int_t lda, const double* B, blas_int_t ldb,
                  double beta, double* C, blas_int_t ldc )
{
   cblas_dgemm( order, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}
```

### BLAS Compatibility Type Trait

**File:** `blaze/math/typetraits/IsBLASCompatible.h`

```cpp
template< typename T >
struct IsBLASCompatible
   : public BoolConstant< IsFloat_v<T> || IsDouble_v<T> ||
                          IsComplexFloat_v<T> || IsComplexDouble_v<T> >
{};
```

### Key Blaze Files Reference

| File | Purpose |
|------|---------|
| `blaze/math/expressions/DMatDMatMultExpr.h` | Matrix-matrix multiply expression |
| `blaze/math/expressions/DMatDVecMultExpr.h` | Matrix-vector multiply expression |
| `blaze/math/blas/gemm.h` | GEMM high-level wrapper |
| `blaze/math/blas/gemv.h` | GEMV high-level wrapper |
| `blaze/math/blas/cblas/gemm.h` | CBLAS GEMM bindings |
| `blaze/math/blas/cblas/gemv.h` | CBLAS GEMV bindings |
| `blaze/math/blas/cblas/dotu.h` | CBLAS DOT bindings |
| `blaze/math/blas/cblas/axpy.h` | CBLAS AXPY bindings |
| `blaze/math/dense/MMM.h` | SIMD matrix multiply kernel |
| `blaze/math/typetraits/IsBLASCompatible.h` | BLAS type checking |

---

## Project Structure

```
zig-blaze/
├── build.zig                    # Build configuration
├── src/
│   ├── main.zig                 # Library entry point
│   ├── vector.zig               # Vector type
│   ├── matrix.zig               # Matrix type
│   ├── expr.zig                 # Expression types
│   ├── eval.zig                 # Comptime evaluator
│   ├── traits.zig               # Type traits
│   ├── blas/
│   │   ├── cblas.zig            # CBLAS bindings (@cImport)
│   │   └── types.zig            # BLAS type definitions
│   └── kernels/
│       ├── simd.zig             # SIMD kernel implementations
│       ├── blocked.zig          # Cache-blocked algorithms
│       └── scalar.zig           # Scalar fallbacks
├── tests/
│   ├── test_vector.zig          # Vector tests
│   ├── test_matrix.zig          # Matrix tests
│   ├── test_blas_l1.zig         # BLAS Level 1 tests
│   ├── test_blas_l2.zig         # BLAS Level 2 tests
│   ├── test_blas_l3.zig         # BLAS Level 3 tests
│   └── bench/
│       └── benchmark.zig        # Performance benchmarks
└── examples/
    ├── basic_ops.zig            # Basic operation examples
    ├── gemm_example.zig         # Matrix multiply example
    └── neural_net.zig           # Neural network layer example
```

### Build Configuration

**File:** `build.zig`

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addStaticLibrary(.{
        .name = "zig-blaze",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Link BLAS library (MKL or OpenBLAS)
    lib.linkSystemLibrary("cblas");
    lib.linkSystemLibrary("blas");

    // For MKL:
    // lib.addLibraryPath(.{ .cwd_relative = "/opt/intel/mkl/lib/intel64" });
    // lib.linkSystemLibrary("mkl_rt");

    b.installArtifact(lib);

    // Tests
    const tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    tests.linkSystemLibrary("cblas");

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
```

---

## Summary: BLAS Operation Coverage

| Level | Operation | BLAS | Zig-Blaze | Status |
|-------|-----------|------|-----------|--------|
| **L1** | Dot product | `cblas_?dot` | `x.dot(y)` | ✅ |
| **L1** | Scaled add | `cblas_?axpy` | `y.axpy(α, x)` | ✅ |
| **L1** | Scale | `cblas_?scal` | `x.scale(α)` | ✅ |
| **L1** | Norm | `cblas_?nrm2` | `x.norm2()` | ✅ |
| **L1** | Sum abs | `cblas_?asum` | `x.asum()` | ✅ |
| **L1** | Max index | `cblas_i?amax` | `x.iamax()` | ✅ |
| **L2** | General MV | `cblas_?gemv` | `A.mulVec(x)` | ✅ |
| **L2** | Triangular MV | `cblas_?trmv` | `A.trmv(x)` | ✅ |
| **L2** | Symmetric MV | `cblas_?symv` | `A.symv(x)` | ✅ |
| **L2** | Tri solve | `cblas_?trsv` | `A.trsv(x)` | ✅ |
| **L2** | Rank-1 | `cblas_?ger` | `A.ger(x, y)` | ✅ |
| **L2** | Sym rank-1 | `cblas_?syr` | `A.syr(x)` | ✅ |
| **L3** | General MM | `cblas_?gemm` | `A.mul(B)` | ✅ |
| **L3** | Triangular MM | `cblas_?trmm` | `A.trmm(B)` | ✅ |
| **L3** | Tri solve MM | `cblas_?trsm` | `A.trsm(B)` | ✅ |

---

## Future Enhancements

1. **Complex number support** - `Complex(f32)`, `Complex(f64)`
2. **Sparse matrix support** - CSR, CSC formats
3. **GPU acceleration** - CUDA/OpenCL backends via Zig's C interop
4. **Auto-parallelization** - Multi-threaded BLAS calls
5. **Expression optimization** - Compile-time common subexpression elimination
6. **Dynamic matrices** - Runtime-sized matrices with allocator support
