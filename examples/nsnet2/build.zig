const std = @import("std");

pub const CommonOptions = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.Mode,
    onnx_dep: *std.Build.Dependency,
    kissfft_dep: *std.Build.Dependency,
};

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const onnx_dep = b.dependency("onnxruntime", .{
        .optimize = optimize,
        .target = target,
    });

    const kissfft_dep = b.dependency("kissfft", .{});

    const common_options = CommonOptions{
        .target = target,
        .optimize = optimize,
        .onnx_dep = onnx_dep,
        .kissfft_dep = kissfft_dep,
    };

    const exe = b.addExecutable(.{
        .name = "nsnet2",
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    exe.linkSystemLibrary("sndfile");
    try addKissFFT(b, exe, common_options);
    try addOnnxRuntime(b, exe, common_options);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the NSNet2 example");
    run_step.dependOn(&run_cmd.step);
}

fn addOnnxRuntime(
    b: *std.Build,
    exe: *std.Build.Step.Compile,
    options: CommonOptions,
) !void {
    _ = b;
    exe.root_module.addImport("onnxruntime", options.onnx_dep.module("onnxruntime"));
}

fn addKissFFT(
    b: *std.Build,
    exe: *std.Build.Step.Compile,
    options: CommonOptions,
) !void {
    const source_files: []const []const u8 = &.{
        "kiss_fft.c",
        "kiss_fftr.c",
    };

    const lib = b.addStaticLibrary(.{
        .name = "kissfft",
        .optimize = options.optimize,
        .target = options.target,
    });

    lib.linkLibC();
    lib.addCSourceFiles(.{
        .root = options.kissfft_dep.path("."),
        .files = source_files,
        .flags = &.{"-Wall"},
    });

    lib.defineCMacro("kiss_fft_scalar", "float");

    exe.addIncludePath(options.kissfft_dep.path("."));
    exe.linkLibrary(lib);
}
