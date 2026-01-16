const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create blaze library module
    const blaze_mod = b.addModule("blaze", .{
        .root_source_file = b.path("src/blaze.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Example executable
    const example_exe = b.addExecutable(.{
        .name = "blaze_example",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    example_exe.root_module.addImport("blaze", blaze_mod);
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
    b.installArtifact(benchmark_exe);

    // Benchmark command
    const benchmark_cmd = b.addRunArtifact(benchmark_exe);
    benchmark_cmd.step.dependOn(b.getInstallStep());
    const benchmark_step = b.step("benchmark", "Run the benchmark");
    benchmark_step.dependOn(&benchmark_cmd.step);

    // Tests
    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/blaze.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
}
