# Zig wrapper for ONNX Runtime

Work in progress, implementing vertical slices of ONNX Runtime API surface as they're needed.

# Examples

Please note that examples don't have a functioning CLI interface at this point, some paths are hardcoded at the top of `main.zig`.

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
