const std = @import("std");
const Allocator = std.mem.Allocator;

/// API docs: https://onnxruntime.ai/docs/api/c/struct_ort_api.html
pub const c_api = @cImport({
    @cInclude("onnxruntime_c_api.h");
});

pub const OrtLoggingLevel = enum(u32) {
    verbose = c_api.ORT_LOGGING_LEVEL_VERBOSE,
    info = c_api.ORT_LOGGING_LEVEL_INFO,
    warning = c_api.ORT_LOGGING_LEVEL_WARNING,
    @"error" = c_api.ORT_LOGGING_LEVEL_ERROR,
    fatal = c_api.ORT_LOGGING_LEVEL_FATAL,
};

pub const ONNXTensorElementDataType = enum(u32) {
    undefined = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
    bool = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
    string = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, // maps to c++ type std::string
    f16 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
    f32 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, // maps to c type float
    f64 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, // maps to c type double
    bf16 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, // Non-IEEE floating-point format based on IEEE754 single-precision
    u8 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, // maps to c type uint8_t
    u16 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, // maps to c type uint16_t
    u32 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, // maps to c type uint32_t
    u64 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, // maps to c type uint64_t
    i8 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, // maps to c type int8_t
    i16 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, // maps to c type int16_t
    i32 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, // maps to c type int32_t
    i64 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, // maps to c type int64_t
    c64 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64, // complex with float32 real and imaginary components
    c128 = c_api.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128, // complex with float64 real and imaginary components
};

pub const OrtAllocatorType = enum(i32) {
    invalid = c_api.OrtInvalidAllocator,
    device = c_api.OrtDeviceAllocator,
    arena = c_api.OrtArenaAllocator,
};

pub const OrtMemType = enum(i32) {
    /// The default allocator for execution provider
    default = c_api.OrtMemTypeDefault,
    /// Any CPU memory used by non-CPU execution provider
    cpu_input = c_api.OrtMemTypeCPUInput,
    /// CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
    cpu_output = c_api.OrtMemTypeCPUOutput,
};

pub const OrtApi = struct {
    const Self = @This();

    allocator: Allocator,
    ort_api: *const c_api.OrtApi,

    pub fn init(allocator: Allocator) !*Self {
        var ort_api = c_api.OrtGetApiBase().*.GetApi.?(c_api.ORT_API_VERSION);

        var self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = Self{
            .allocator = allocator,
            .ort_api = ort_api,
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }

    pub fn createEnv(
        self: *Self,
        log_level: OrtLoggingLevel,
        log_id: [:0]const u8,
    ) !*c_api.OrtEnv {
        var ort_env: ?*c_api.OrtEnv = null;
        const status = self.ort_api.CreateEnv.?(@enumToInt(log_level), log_id.ptr, &ort_env);

        try self.checkError(status);
        return ort_env.?;
    }

    pub fn createSessionOptions(
        self: *Self,
    ) !*c_api.OrtSessionOptions {
        var ort_sess_opts: ?*c_api.OrtSessionOptions = null;
        const status = self.ort_api.CreateSessionOptions.?(&ort_sess_opts);

        try self.checkError(status);
        return ort_sess_opts.?;
    }

    pub fn createSession(
        self: *Self,
        ort_env: *c_api.OrtEnv,
        model_path: [:0]const u8,
        ort_sess_opts: *c_api.OrtSessionOptions,
    ) !*c_api.OrtSession {
        var ort_sess: ?*c_api.OrtSession = null;
        const status = self.ort_api.CreateSession.?(
            ort_env,
            model_path.ptr,
            ort_sess_opts,
            &ort_sess,
        );

        try self.checkError(status);
        return ort_sess.?;
    }

    pub fn createMemoryInfo(
        self: *Self,
        name: [:0]const u8,
        allocator_type: OrtAllocatorType,
        id: i32,
        mem_type: OrtMemType,
    ) !*c_api.OrtMemoryInfo {
        var mem_info: ?*c_api.OrtMemoryInfo = null;
        const status = self.ort_api.CreateMemoryInfo.?(
            name.ptr,
            @enumToInt(allocator_type),
            id,
            @enumToInt(mem_type),
            &mem_info,
        );

        try self.checkError(status);
        return mem_info.?;
    }

    pub fn createTensorWithDataAsOrtValue(
        self: *Self,
        comptime T: type,
        mem_info: *const c_api.OrtMemoryInfo,
        data: []T,
        shape: []const i64,
        tensor_type: ONNXTensorElementDataType,
    ) !*c_api.OrtValue {
        var value: ?*c_api.OrtValue = null;
        const status = self.ort_api.CreateTensorWithDataAsOrtValue.?(
            mem_info,
            data.ptr,
            data.len * @sizeOf(T),
            shape.ptr,
            shape.len,
            @enumToInt(tensor_type),
            &value,
        );

        try self.checkError(status);
        return value.?;
    }

    pub fn createRunOptions(
        self: *Self,
    ) !*c_api.OrtRunOptions {
        var run_opts: ?*c_api.OrtRunOptions = null;
        const status = self.ort_api.CreateRunOptions.?(&run_opts);

        try self.checkError(status);
        return run_opts.?;
    }

    pub fn run(
        self: *Self,
        ort_sess: *c_api.OrtSession,
        run_opts: *c_api.OrtRunOptions,
        input_names: []const [*:0]const u8,
        inputs: []const *c_api.OrtValue,
        output_names: []const [*:0]const u8,
        outputs: []?*c_api.OrtValue,
    ) !void {
        const status = self.ort_api.Run.?(
            ort_sess,
            run_opts,
            input_names.ptr,
            inputs.ptr,
            inputs.len,
            output_names.ptr,
            output_names.len,
            outputs.ptr,
        );

        try self.checkError(status);
    }

    pub fn checkError(
        self: *Self,
        onnx_status: ?*c_api.OrtStatus,
    ) !void {
        if (onnx_status == null) return;
        defer self.ort_api.ReleaseStatus.?(onnx_status);

        const msg = self.ort_api.GetErrorMessage.?(onnx_status);
        std.debug.print("ONNX error: {s}\n", .{msg});

        return error.OnnxError;
    }
};
