const std = @import("std");

pub const CommonOptions = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.Mode,
    onnx_dep: *std.Build.Dependency,
};

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const onnx_dep = b.dependency("onnxruntime", .{
        .optimize = optimize,
        .target = target,
    });

    const common_options = CommonOptions{
        .target = target,
        .optimize = optimize,
        .onnx_dep = onnx_dep,
    };

    const exe = b.addExecutable(.{
        .name = "silero_vad",
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = .{ .path = "src/main.zig" },
        .target = common_options.target,
        .optimize = common_options.optimize,
    });

    exe.linkSystemLibrary("sndfile");
    exe.root_module.addImport("onnxruntime", onnx_dep.module("onnxruntime"));

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the Silero VAD example");
    run_step.dependOn(&run_cmd.step);
}
