# Zig wrapper for ONNX Runtime

Work in progress, implementing vertical slices of ONNX Runtime API surface as they're needed.

# Dependencies

To build & run this project, download ONNX Runtime from https://github.com/microsoft/onnxruntime/releases
and drop the `.so` libraries in `lib/onnxruntime-linux-x64/lib`


```bash
curl https://github.com/microsoft/onnxruntime/releases/download/v1.15.0/onnxruntime-linux-x64-1.15.0.tgz --output lib/onnxruntime.tgz --fail --location
tar xvf lib/onnxruntime.tgz --directory lib/onnxruntime-linux-x64 --strip-components 1
rm lib/onnxruntime.tgz
```

# Examples

Please note that examples don't have a functioning CLI interface at this point, some paths are hardcoded at the top of `main.zig`.

To build or run the examples, run:

```bash
# Run Silero VAD
zig build run-silero-vad
# Run NSNet2
zig build run-nsnet
# Build all examples
zig build examples
```

# Licensing

`/src/*`, `/examples/*/src/*` and any other first party code - Mozilla Public License 2.0 (`/LICENSE.txt`)

`/lib/*`, `/examples/*/lib/*` and any other third party libraries - See original projects for licensing information.

`/examples/nsnet2/data/*.onnx` - NSNet2 ONNX models are [MIT licensed by Microsoft](https://github.com/microsoft/DNS-Challenge/tree/v4dnschallenge_ICASSP2022).

`/examples/silero_vad/data/*.onnx` - Silero VAD ONNX models are [MIT licensed by Silero Team](https://github.com/snakers4/silero-vad/tree/7e9680bc83230b745f4794219fe11f4ea50965cd).

`/examples/*/data/*.wav` - Example WAV files are unlicensed. Please open a GitHub issue to request removal if you believe these files are infringing on your copyright and fall outside the scope of fair use.
