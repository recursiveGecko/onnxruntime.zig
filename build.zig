const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const onnx_dep = b.dependency("onnxruntime_linux_x64", .{});

    const lib_mod = b.addModule("onnxruntime", .{
        .root_source_file = b.path("src/onnxruntime.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_mod.addIncludePath(onnx_dep.path("include"));

    const onnx_lib_mod = b.addModule("onnxruntime_lib", .{
        .root_source_file = onnx_dep.path("lib"),
    });
    _ = onnx_lib_mod;
}
