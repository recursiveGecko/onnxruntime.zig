const std = @import("std");
const main_build = @import("../../build.zig");

pub fn build(
    b: *std.Build,
    common_options: main_build.CommonOptions,
    examples_step: *std.Build.Step,
) !void {
    const exe = b.addExecutable(.{
        .name = "nsnet2",
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = .{ .path = projectPath("src/main.zig") },
        .target = common_options.target,
        .optimize = common_options.optimize,
    });

    exe.linkSystemLibrary("sndfile");
    try addKissFFT(b, exe, common_options);
    try main_build.linkPackage(b, exe, common_options);

    const exe_install = b.addInstallArtifact(exe);
    examples_step.dependOn(&exe_install.step);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run-nsnet", "Run the NSNet2 example");
    run_step.dependOn(&run_cmd.step);
}

fn addKissFFT(
    b: *std.Build,
    exe: *std.Build.Step.Compile,
    options: main_build.CommonOptions,
) !void {
    const source_files: []const []const u8 = &.{
        projectPath("lib/kissfft/kiss_fft.c"),
        projectPath("lib/kissfft/kiss_fftr.c"),
    };

    const lib = b.addStaticLibrary(.{
        .name = "kissfft",
        .optimize = options.optimize,
        .target = options.target,
    });

    lib.linkLibC();
    lib.addCSourceFiles(source_files, &.{"-Wall"});

    lib.defineCMacro("kiss_fft_scalar", "float");

    exe.addIncludePath(projectPath("lib/kissfft"));
    exe.linkLibrary(lib);
}

pub inline fn projectPath(path: []const u8) []const u8 {
    return comptime projectBaseDir() ++ .{std.fs.path.sep} ++ path;
}

pub inline fn projectBaseDir() []const u8 {
    return comptime std.fs.path.dirname(@src().file).?;
}
