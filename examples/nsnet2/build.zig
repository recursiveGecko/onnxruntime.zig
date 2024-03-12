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

    const onnx_dep = b.dependency("zig_onnxruntime", .{
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
    exe.root_module.addImport("onnxruntime", onnx_dep.module("zig-onnxruntime"));
    try addKissFFT(b, exe, common_options);

    exe.root_module.addRPathSpecial("$ORIGIN/../lib");
    exe.each_lib_rpath = false;

    b.installArtifact(exe);

    // ugly hack to install onnxruntime.so libraries (without symlinks: https://github.com/ziglang/zig/pull/18619)
    // inspired by https://medium.com/@edlyuu/zig-package-manager-2-wtf-is-build-zig-zon-and-build-zig-0-11-0-update-5bc46e830fc1
    const install_onnx_libs = b.addInstallDirectory(.{
        .source_dir = onnx_dep.module("onnxruntime_lib").root_source_file.?,
        .install_dir = .lib,
        .install_subdir = ".",
    });
    b.getInstallStep().dependOn(&install_onnx_libs.step);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the NSNet2 example");
    run_step.dependOn(&run_cmd.step);
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
