const std = @import("std");

pub const CommonOptions = struct {
    target: std.zig.CrossTarget,
    optimize: std.builtin.Mode,
};

pub fn linkPackage(
    b: *std.Build,
    exe: *std.build.CompileStep,
    common_options: CommonOptions,
) !void {
    const module = b.createModule(.{
        .source_file = .{ .path = projectPath("src/lib.zig") },
    });

    exe.addModule("onnxruntime", module);
    try addOnnxRuntime(b, exe, common_options);
}

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const common_options = CommonOptions{
        .target = target,
        .optimize = optimize,
    };

    const lib = try buildLib(b, common_options);
    _ = lib;

    const examples_build_step = b.step("examples", "Build examples");
    try maybeBuildExamples(b, common_options, examples_build_step);
}

pub fn buildLib(b: *std.Build, common_options: CommonOptions) !*std.build.CompileStep {
    var lib = b.addStaticLibrary(.{
        .name = "zig-onnxruntime",
        .root_source_file = .{ .path = projectPath("src/lib.zig") },
        .target = common_options.target,
        .optimize = common_options.optimize,
    });

    try addOnnxRuntime(b, lib, common_options);

    b.installArtifact(lib);

    const main_tests = b.addTest(.{
        .root_source_file = .{ .path = projectPath("src/lib.zig") },
        .target = common_options.target,
        .optimize = common_options.optimize,
    });

    const run_main_tests = b.addRunArtifact(main_tests);

    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_main_tests.step);

    return lib;
}

pub fn addOnnxRuntime(b: *std.Build, unit: *std.build.CompileStep, common_options: CommonOptions) !void {
    _ = common_options;

    b.addSearchPrefix("lib/onnxruntime-linux-x64");
    unit.addIncludePath("lib/onnxruntime-linux-x64/include");
    // unit.each_lib_rpath = false;
    unit.linkSystemLibrary("onnxruntime");
    unit.linkLibC();
}

pub fn maybeBuildExamples(
    b: *std.Build,
    common_options: CommonOptions,
    examples_step: *std.Build.Step,
) !void {
    const silero_vad = @import("examples/silero_vad/build.zig");
    try silero_vad.build(b, common_options, examples_step);

    const nsnet2 = @import("examples/nsnet2/build.zig");
    try nsnet2.build(b, common_options, examples_step);
}

pub fn projectPaths(
    allocator: std.mem.Allocator,
    prefix: []const u8,
    paths: []const []const u8,
) ![]const []const u8 {
    var prefixed_paths = std.ArrayList([]const u8).init(allocator);
    defer prefixed_paths.deinit();

    for (paths) |path| {
        const prefixed_path = try std.fs.path.join(allocator, &.{ prefix, path });
        defer allocator.free(prefixed_path);

        const project_path = try projectPath(allocator, prefixed_path);
        try prefixed_paths.append(project_path);
    }

    return prefixed_paths.toOwnedSlice();
}

pub inline fn projectPath(path: []const u8) []const u8 {
    return comptime projectBaseDir() ++ .{std.fs.path.sep} ++ path;
}

pub inline fn projectBaseDir() []const u8 {
    return comptime std.fs.path.dirname(@src().file).?;
}
