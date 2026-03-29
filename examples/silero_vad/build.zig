const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Zig wrapper
    const zig_onnx_dep = b.dependency("zig_onnxruntime", .{
        .optimize = optimize,
        .target = target,
    });
    const zig_onnx_mod = zig_onnx_dep.module("onnxruntime");
    const onnxruntime_dep = b.dependency("onnxruntime_linux_x64", .{});

    const build_options = b.addOptions();
    build_options.addOption([]const u8, "data_dir", b.pathFromRoot("data"));

    const exe = b.addExecutable(.{
        .name = "silero_vad",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    exe.root_module.addImport("onnxruntime", zig_onnx_mod);
    exe.root_module.addOptions("build_options", build_options);
    exe.root_module.addLibraryPath(onnxruntime_dep.path("lib"));
    exe.linkLibC();
    exe.linkSystemLibrary("onnxruntime");
    exe.linkSystemLibrary("sndfile");

    exe.root_module.addRPathSpecial("$ORIGIN/../lib");
    exe.each_lib_rpath = false;

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

    const run_step = b.step("run", "Run the Silero VAD example");
    run_step.dependOn(&run_cmd.step);
}
