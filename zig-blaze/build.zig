const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Get MKL path from environment (Zig 0.16+ API)
    const mkl_root = b.graph.environ_map.get("MKLROOT") orelse "/usr";

    // MKL paths
    const mkl_include = b.fmt("{s}/include", .{mkl_root});
    const mkl_lib = b.fmt("{s}/lib", .{mkl_root});

    // Blaze Zig library module
    const blaze_mod = b.createModule(.{
        .root_source_file = b.path("src/blaze.zig"),
        .target = target,
        .optimize = optimize,
    });
    blaze_mod.addIncludePath(.{ .cwd_relative = mkl_include });

    // Benchmark executable
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("src/bench.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    bench_mod.addImport("blaze", blaze_mod);
    bench_mod.addIncludePath(.{ .cwd_relative = mkl_include });
    bench_mod.addLibraryPath(.{ .cwd_relative = mkl_lib });
    bench_mod.linkSystemLibrary("mkl_intel_lp64", .{});
    bench_mod.linkSystemLibrary("mkl_sequential", .{});
    bench_mod.linkSystemLibrary("mkl_core", .{});
    bench_mod.linkSystemLibrary("pthread", .{});
    bench_mod.linkSystemLibrary("m", .{});
    bench_mod.linkSystemLibrary("dl", .{});

    const bench_exe = b.addExecutable(.{
        .name = "blaze_zig_bench",
        .root_module = bench_mod,
    });
    b.installArtifact(bench_exe);

    // Example executable
    const example_mod = b.createModule(.{
        .root_source_file = b.path("src/example.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    example_mod.addImport("blaze", blaze_mod);
    example_mod.addIncludePath(.{ .cwd_relative = mkl_include });
    example_mod.addLibraryPath(.{ .cwd_relative = mkl_lib });
    example_mod.linkSystemLibrary("mkl_intel_lp64", .{});
    example_mod.linkSystemLibrary("mkl_sequential", .{});
    example_mod.linkSystemLibrary("mkl_core", .{});
    example_mod.linkSystemLibrary("pthread", .{});
    example_mod.linkSystemLibrary("m", .{});
    example_mod.linkSystemLibrary("dl", .{});

    const example_exe = b.addExecutable(.{
        .name = "blaze_zig_example",
        .root_module = example_mod,
    });
    b.installArtifact(example_exe);

    // Run command
    const run_cmd = b.addRunArtifact(bench_exe);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the benchmark");
    run_step.dependOn(&run_cmd.step);

    // Tests
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/blaze.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_mod.addIncludePath(.{ .cwd_relative = mkl_include });
    test_mod.addLibraryPath(.{ .cwd_relative = mkl_lib });
    test_mod.linkSystemLibrary("mkl_intel_lp64", .{});
    test_mod.linkSystemLibrary("mkl_sequential", .{});
    test_mod.linkSystemLibrary("mkl_core", .{});

    const tests = b.addTest(.{
        .root_module = test_mod,
    });

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);
}
