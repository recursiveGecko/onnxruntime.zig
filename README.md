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
