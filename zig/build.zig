const std = @import("std");

/// BLAS backend options
const BlasBackend = enum {
    none,
    openblas,
    mkl,
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build option to select BLAS backend
    const blas_backend = b.option(BlasBackend, "blas", "BLAS backend to use (none, openblas, mkl)") orelse .none;

    // Legacy option for backwards compatibility
    const use_mkl_legacy = b.option(bool, "use-mkl", "Use Intel MKL for BLAS operations (legacy, prefer -Dblas=mkl)") orelse false;

    // Determine effective backend
    const effective_backend: BlasBackend = if (use_mkl_legacy) .mkl else blas_backend;

    // Get library path from environment (set by pixi/conda)
    const conda_prefix = std.process.getEnvVarOwned(b.allocator, "CONDA_PREFIX") catch null;

    // Check if BLAS is actually available
    const blas_available = effective_backend != .none and conda_prefix != null;

    // Build options for the module
    const options = b.addOptions();
    options.addOption(bool, "use_mkl", blas_available);
    options.addOption(bool, "use_openblas", blas_available and effective_backend == .openblas);
    options.addOption(bool, "use_intel_mkl", blas_available and effective_backend == .mkl);

    // Create blaze library module
    const blaze_mod = b.addModule("blaze", .{
        .root_source_file = b.path("src/blaze.zig"),
        .target = target,
        .optimize = optimize,
    });
    blaze_mod.addOptions("build_options", options);

    // Helper function to configure MKL linking
    // Note: MKL from conda-forge may have compatibility issues with Ubuntu's glibc paths.
    // For CI, OpenBLAS is recommended. MKL works best with RHEL-like systems or
    // when the conda environment is properly configured.
    const configureMkl = struct {
        fn configure(exe: *std.Build.Step.Compile, prefix: []const u8, allocator: std.mem.Allocator) void {
            const lib_path = std.fmt.allocPrint(allocator, "{s}/lib", .{prefix}) catch unreachable;
            const include_path = std.fmt.allocPrint(allocator, "{s}/include", .{prefix}) catch unreachable;

            // Add include path for MKL headers
            exe.addIncludePath(.{ .cwd_relative = include_path });

            // Add conda environment library search path and rpath
            exe.addLibraryPath(.{ .cwd_relative = lib_path });
            exe.addRPath(.{ .cwd_relative = lib_path });

            // Use MKL Single Dynamic Library (SDL) for simpler linking
            // mkl_rt automatically loads the appropriate interface, threading, and core libraries
            exe.linkSystemLibrary("mkl_rt");

            // Link libc which provides pthread, m, dl automatically
            exe.linkLibC();
        }
    }.configure;

    // Helper function to configure OpenBLAS linking
    // OpenBLAS is more portable than MKL and works well on Ubuntu/Debian systems.
    const configureOpenBlas = struct {
        fn configure(exe: *std.Build.Step.Compile, prefix: []const u8, allocator: std.mem.Allocator) void {
            const lib_path = std.fmt.allocPrint(allocator, "{s}/lib", .{prefix}) catch unreachable;

            // Add library search path and rpath
            exe.addLibraryPath(.{ .cwd_relative = lib_path });
            exe.addRPath(.{ .cwd_relative = lib_path });

            // Link OpenBLAS (provides CBLAS interface)
            exe.linkSystemLibrary("openblas");

            // Link libc for system dependencies
            exe.linkLibC();
        }
    }.configure;

    // Helper to configure BLAS for an executable
    const configureBlasForExe = struct {
        fn configure(
            exe: *std.Build.Step.Compile,
            backend: BlasBackend,
            prefix: ?[]const u8,
            allocator: std.mem.Allocator,
            mkl_fn: *const fn (*std.Build.Step.Compile, []const u8, std.mem.Allocator) void,
            openblas_fn: *const fn (*std.Build.Step.Compile, []const u8, std.mem.Allocator) void,
        ) void {
            if (prefix) |p| {
                switch (backend) {
                    .mkl => mkl_fn(exe, p, allocator),
                    .openblas => openblas_fn(exe, p, allocator),
                    .none => {},
                }
            }
        }
    }.configure;

    // Example executable
    const example_exe = b.addExecutable(.{
        .name = "blaze_example",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    example_exe.root_module.addImport("blaze", blaze_mod);
    // Note: build_options is accessed through blaze module, no need to add directly
    configureBlasForExe(example_exe, effective_backend, conda_prefix, b.allocator, configureMkl, configureOpenBlas);
    b.installArtifact(example_exe);

    // Run example command
    const run_cmd = b.addRunArtifact(example_exe);
    run_cmd.step.dependOn(b.getInstallStep());
    const run_step = b.step("run", "Run the example application");
    run_step.dependOn(&run_cmd.step);

    // Benchmark executable
    const benchmark_exe = b.addExecutable(.{
        .name = "blaze_benchmark",
        .root_source_file = b.path("src/benchmark.zig"),
        .target = target,
        .optimize = optimize,
    });
    benchmark_exe.root_module.addImport("blaze", blaze_mod);
    // Note: build_options is accessed through blaze module, no need to add directly
    configureBlasForExe(benchmark_exe, effective_backend, conda_prefix, b.allocator, configureMkl, configureOpenBlas);
    b.installArtifact(benchmark_exe);

    // Benchmark command
    const benchmark_cmd = b.addRunArtifact(benchmark_exe);
    benchmark_cmd.step.dependOn(b.getInstallStep());
    const benchmark_step = b.step("benchmark", "Run the benchmark");
    benchmark_step.dependOn(&benchmark_cmd.step);

    // Tests (without BLAS to keep tests simple and portable)
    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/blaze.zig"),
        .target = target,
        .optimize = optimize,
    });
    // Tests use pure Zig implementation (no BLAS)
    const test_options = b.addOptions();
    test_options.addOption(bool, "use_mkl", false);
    test_options.addOption(bool, "use_openblas", false);
    test_options.addOption(bool, "use_intel_mkl", false);
    lib_unit_tests.root_module.addOptions("build_options", test_options);

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
}
