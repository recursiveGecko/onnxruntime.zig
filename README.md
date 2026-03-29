# Zig wrapper for ONNX Runtime

Work in progress, implementing vertical slices of ONNX Runtime API surface as they're needed.

# Usage

Add the dependency:

```bash
zig fetch --save git+https://github.com/recursiveGecko/onnxruntime.zig
```

Then wire it into `build.zig`:

```zig
const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // the wrapper
    const zig_onnx_dep = b.dependency("onnxruntime_zig", .{
        .target = target,
        .optimize = optimize,
    });

    // onnx runtime shared library
    const onnxruntime_dep = b.dependency("onnxruntime_linux_x64", .{});

    const exe = b.addExecutable(.{
        .name = "app",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.root_module.addImport("onnxruntime", zig_onnx_dep.module("onnxruntime"));
    exe.root_module.addLibraryPath(onnxruntime_dep.path("lib"));

    exe.linkLibC();
    exe.linkSystemLibrary("onnxruntime");

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
}
```

And in `src/main.zig`:

```zig
const onnxruntime = @import("onnxruntime");

pub fn main() !void {
    _ = onnxruntime;
}
```

# Examples

Please note that examples don't have a functioning CLI interface at this point, some paths are hardcoded at the top of `main.zig`.

Both examples currently require `libsndfile` to be installed on the system.
On Debian/Ubuntu, install `libsndfile1-dev`.

To build or run the examples, run:

```bash
# Run Silero VAD
cd examples/silero_vad
zig build run
# Run NSNet2
cd examples/nsnet2
zig build run
```

# Licensing

`/src/*`, `/examples/*/src/*` and any other first party code - Mozilla Public License 2.0 (`/LICENSE.txt`)

`/examples/nsnet2/data/*.onnx` - NSNet2 ONNX models are [MIT licensed by Microsoft](https://github.com/microsoft/DNS-Challenge/tree/v4dnschallenge_ICASSP2022).

`/examples/silero_vad/data/*.onnx` - Silero VAD ONNX models are [MIT licensed by Silero Team](https://github.com/snakers4/silero-vad/tree/7e9680bc83230b745f4794219fe11f4ea50965cd).

`/examples/*/data/*.wav` - Example WAV files are unlicensed. Please open a GitHub issue to request removal if you believe these files are infringing on your copyright and fall outside the scope of fair use.
