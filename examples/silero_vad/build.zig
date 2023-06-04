const std = @import("std");
const main_build = @import("../../build.zig");

pub fn build(
    b: *std.Build,
    common_options: main_build.CommonOptions,
    examples_step: *std.Build.Step,
) !void {
    const exe = b.addExecutable(.{
        .name = "silero_vad",
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = .{ .path = projectPath("src/main.zig") },
        .target = common_options.target,
        .optimize = common_options.optimize,
    });
    
    exe.linkSystemLibrary("sndfile");
    try main_build.linkPackage(b, exe, common_options);

    const exe_install = b.addInstallArtifact(exe);
    examples_step.dependOn(&exe_install.step);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run-silero-vad", "Run the Silero VAD example");
    run_step.dependOn(&run_cmd.step);
}

pub inline fn projectPath(path: []const u8) []const u8 {
    return comptime projectBaseDir() ++ .{std.fs.path.sep} ++ path;
}

pub inline fn projectBaseDir() []const u8 {
    return comptime std.fs.path.dirname(@src().file).?;
}
