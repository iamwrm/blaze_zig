const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Get MKL library path from environment (set by pixi)
    const conda_prefix = std.process.getEnvVarOwned(b.allocator, "CONDA_PREFIX") catch null;

    // Create blaze library module with BLAS support
    const blaze_mod = b.addModule("blaze", .{
        .root_source_file = b.path("src/blaze.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Link MKL to the blaze module
    if (conda_prefix) |prefix| {
        const lib_path = std.fmt.allocPrint(b.allocator, "{s}/lib", .{prefix}) catch unreachable;
        blaze_mod.addLibraryPath(.{ .cwd_relative = lib_path });
        blaze_mod.linkSystemLibrary("mkl_rt", .{});
    }

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
    if (conda_prefix) |prefix| {
        const lib_path = std.fmt.allocPrint(b.allocator, "{s}/lib", .{prefix}) catch unreachable;
        example_exe.root_module.addLibraryPath(.{ .cwd_relative = lib_path });
        example_exe.root_module.linkSystemLibrary("mkl_rt", .{});
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
    if (conda_prefix) |prefix| {
        const lib_path = std.fmt.allocPrint(b.allocator, "{s}/lib", .{prefix}) catch unreachable;
        benchmark_exe.root_module.addLibraryPath(.{ .cwd_relative = lib_path });
        benchmark_exe.root_module.linkSystemLibrary("mkl_rt", .{});
    }
    b.installArtifact(benchmark_exe);

    // Benchmark command
    const benchmark_cmd = b.addRunArtifact(benchmark_exe);
    benchmark_cmd.step.dependOn(b.getInstallStep());
    const benchmark_step = b.step("benchmark", "Run the benchmark");
    benchmark_step.dependOn(&benchmark_cmd.step);

    // Tests (without MKL for simplicity)
    const lib_unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blaze.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    if (conda_prefix) |prefix| {
        const lib_path = std.fmt.allocPrint(b.allocator, "{s}/lib", .{prefix}) catch unreachable;
        lib_unit_tests.root_module.addLibraryPath(.{ .cwd_relative = lib_path });
        lib_unit_tests.root_module.linkSystemLibrary("mkl_rt", .{});
    }
    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
}
