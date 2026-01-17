const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Get MKL path from environment
    const mkl_root = std.posix.getenv("MKLROOT") orelse "/usr";
    
    // MKL paths
    const mkl_include = b.fmt("{s}/include", .{mkl_root});
    const mkl_lib = b.fmt("{s}/lib", .{mkl_root});

    // Blaze Zig library
    const blaze_mod = b.addModule("blaze", .{
        .root_source_file = b.path("src/blaze.zig"),
        .target = target,
        .optimize = optimize,
    });
    blaze_mod.addIncludePath(.{ .cwd_relative = mkl_include });

    // Benchmark executable
    const bench_exe = b.addExecutable(.{
        .name = "blaze_zig_bench",
        .root = b.path("src/bench.zig"),
        .target = target,
        .optimize = optimize,
    });

    bench_exe.root_module.addImport("blaze", blaze_mod);

    // Add MKL include path to benchmark too
    bench_exe.addIncludePath(.{ .cwd_relative = mkl_include });

    // Add MKL library path
    bench_exe.addLibraryPath(.{ .cwd_relative = mkl_lib });
    bench_exe.addRPath(.{ .cwd_relative = mkl_lib });

    // Link MKL libraries in correct order (sequential, lp64)
    // MKL requires specific link order and may need --start-group/--end-group
    bench_exe.linkSystemLibrary("mkl_intel_lp64");
    bench_exe.linkSystemLibrary("mkl_sequential");
    bench_exe.linkSystemLibrary("mkl_core");
    // Some systems need to link these again for circular dependencies
    bench_exe.linkSystemLibrary("mkl_intel_lp64");
    bench_exe.linkSystemLibrary("mkl_sequential");
    bench_exe.linkSystemLibrary("mkl_core");
    bench_exe.linkSystemLibrary("pthread");
    bench_exe.linkSystemLibrary("m");
    bench_exe.linkSystemLibrary("dl");
    bench_exe.linkLibC();

    b.installArtifact(bench_exe);

    // Example executable
    const example_exe = b.addExecutable(.{
        .name = "blaze_zig_example",
        .root = b.path("src/example.zig"),
        .target = target,
        .optimize = optimize,
    });

    example_exe.root_module.addImport("blaze", blaze_mod);
    example_exe.addIncludePath(.{ .cwd_relative = mkl_include });
    example_exe.addLibraryPath(.{ .cwd_relative = mkl_lib });
    example_exe.addRPath(.{ .cwd_relative = mkl_lib });
    example_exe.linkSystemLibrary("mkl_intel_lp64");
    example_exe.linkSystemLibrary("mkl_sequential");
    example_exe.linkSystemLibrary("mkl_core");
    example_exe.linkSystemLibrary("mkl_intel_lp64");
    example_exe.linkSystemLibrary("mkl_sequential");
    example_exe.linkSystemLibrary("mkl_core");
    example_exe.linkSystemLibrary("pthread");
    example_exe.linkSystemLibrary("m");
    example_exe.linkSystemLibrary("dl");
    example_exe.linkLibC();

    b.installArtifact(example_exe);

    // Run command
    const run_cmd = b.addRunArtifact(bench_exe);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the benchmark");
    run_step.dependOn(&run_cmd.step);

    // Tests
    const tests = b.addTest(.{
        .root = b.path("src/blaze.zig"),
        .target = target,
        .optimize = optimize,
    });
    tests.addIncludePath(.{ .cwd_relative = mkl_include });
    tests.addLibraryPath(.{ .cwd_relative = mkl_lib });
    tests.linkSystemLibrary("mkl_intel_lp64");
    tests.linkSystemLibrary("mkl_sequential");
    tests.linkSystemLibrary("mkl_core");
    tests.linkLibC();

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);
}
