const std = @import("std");

pub const CommonOptions = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    kissfft_dep: *std.Build.Dependency,
};

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zig_onnx_dep = b.dependency("zig_onnxruntime", .{
        .optimize = optimize,
        .target = target,
    });
    const onnxruntime_dep = b.dependency("onnxruntime_linux_x64", .{});

    const kissfft_dep = b.dependency("kissfft", .{});
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "data_dir", b.pathFromRoot("data"));

    const common_options = CommonOptions{
        .target = target,
        .optimize = optimize,
        .kissfft_dep = kissfft_dep,
    };

    const exe = b.addExecutable(.{
        .name = "nsnet2",
        .root_module = b.createModule(.{
            // In this case the main source file is merely a path, however, in more
            // complicated build scripts, this could be a generated file.
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    exe.linkSystemLibrary("sndfile");
    exe.root_module.addImport("onnxruntime", zig_onnx_dep.module("onnxruntime"));
    exe.root_module.addOptions("build_options", build_options);
    exe.root_module.addLibraryPath(onnxruntime_dep.path("lib"));
    exe.linkSystemLibrary("onnxruntime");
    try addKissFFT(b, exe, common_options);

    // install
    const install_exe = b.addInstallArtifact(exe, .{});
    b.getInstallStep().dependOn(&install_exe.step);

    {
        // onnxruntime shared library
        const install_onnxruntime_libs = b.addInstallLibFile(
            onnxruntime_dep.path("lib/libonnxruntime.so.1"),
            "libonnxruntime.so.1",
        );
        install_exe.step.dependOn(&install_onnxruntime_libs.step);

        // optional, load library from relative dir
        exe.root_module.addRPathSpecial("$ORIGIN/../lib");
        exe.each_lib_rpath = false;
    }

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(&install_exe.step);

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

    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "kissfft",
        .root_module = b.createModule(.{
            .target = options.target,
            .optimize = options.optimize,
        }),
    });

    lib.linkLibC();
    lib.root_module.addCSourceFiles(.{
        .root = options.kissfft_dep.path("."),
        .files = source_files,
        .flags = &.{"-Wall"},
    });

    lib.root_module.addCMacro("kiss_fft_scalar", "float");

    exe.root_module.addIncludePath(options.kissfft_dep.path("."));
    exe.linkLibrary(lib);
}
