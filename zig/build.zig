const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build option to enable MKL
    const use_mkl = b.option(bool, "use-mkl", "Use Intel MKL for BLAS operations") orelse false;

    // Get MKL library path from environment (set by pixi)
    const conda_prefix = std.process.getEnvVarOwned(b.allocator, "CONDA_PREFIX") catch null;

    // Build options for the module
    const options = b.addOptions();
    options.addOption(bool, "use_mkl", use_mkl and conda_prefix != null);

    // Create blaze library module
    const blaze_mod = b.addModule("blaze", .{
        .root_source_file = b.path("src/blaze.zig"),
        .target = target,
        .optimize = optimize,
    });
    blaze_mod.addOptions("build_options", options);

    // Helper function to configure BLAS linking for an executable (using OpenBLAS)
    const configureBlas = struct {
        fn configure(exe: *std.Build.Step.Compile, prefix: []const u8, allocator: std.mem.Allocator) void {
            const lib_path = std.fmt.allocPrint(allocator, "{s}/lib", .{prefix}) catch unreachable;

            // Add library search path and rpath
            exe.addLibraryPath(.{ .cwd_relative = lib_path });
            exe.addRPath(.{ .cwd_relative = lib_path });

            // Link OpenBLAS (provides CBLAS interface)
            exe.linkSystemLibrary("openblas");
        }
    }.configure;

    // Example executable
    const example_exe = b.addExecutable(.{
        .name = "blaze_example",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "blaze", .module = blaze_mod },
            },
        }),
    });
    example_exe.root_module.addOptions("build_options", options);
    if (use_mkl) {
        if (conda_prefix) |prefix| {
            configureBlas(example_exe, prefix, b.allocator);
        }
    }
    b.installArtifact(example_exe);

    // Run example command
    const run_cmd = b.addRunArtifact(example_exe);
    run_cmd.step.dependOn(b.getInstallStep());
    const run_step = b.step("run", "Run the example application");
    run_step.dependOn(&run_cmd.step);

    // Benchmark executable
    const benchmark_exe = b.addExecutable(.{
        .name = "blaze_benchmark",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/benchmark.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "blaze", .module = blaze_mod },
            },
        }),
    });
    benchmark_exe.root_module.addOptions("build_options", options);
    if (use_mkl) {
        if (conda_prefix) |prefix| {
            configureBlas(benchmark_exe, prefix, b.allocator);
        }
    }
    b.installArtifact(benchmark_exe);

    // Benchmark command
    const benchmark_cmd = b.addRunArtifact(benchmark_exe);
    benchmark_cmd.step.dependOn(b.getInstallStep());
    const benchmark_step = b.step("benchmark", "Run the benchmark");
    benchmark_step.dependOn(&benchmark_cmd.step);

    // Tests (without MKL to keep tests simple)
    const lib_unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blaze.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    // Tests use pure Zig implementation (no MKL)
    const test_options = b.addOptions();
    test_options.addOption(bool, "use_mkl", false);
    lib_unit_tests.root_module.addOptions("build_options", test_options);

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
}
