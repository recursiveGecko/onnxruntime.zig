const std = @import("std");

pub const CommonOptions = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.Mode,
};

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const common_options = CommonOptions{
        .target = target,
        .optimize = optimize,
    };

    const static_lib = try buildStaticLib(b, common_options);
    _ = static_lib;

    const lib_mod = b.addModule(
        "onnxruntime",
        .{
            .root_source_file = .{ .path = "src/lib.zig" },
        },
    );
    _ = lib_mod;
}

pub fn buildStaticLib(b: *std.Build, common_options: CommonOptions) !*std.Build.Step.Compile {
    const lib = b.addStaticLibrary(.{
        .name = "onnxruntime-zig",
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

pub fn addOnnxRuntime(b: *std.Build, unit: *std.Build.Step.Compile, common_options: CommonOptions) !void {
    _ = common_options;

    const onnx = b.dependency("onnxruntime_linux_x64", .{});

    unit.addIncludePath(.{ .path = onnx.path("include").getPath(b) });
    unit.addLibraryPath(.{ .path = onnx.path("lib").getPath(b) });

    unit.each_lib_rpath = true;
    unit.linkSystemLibrary("onnxruntime");
    unit.linkLibC();
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
