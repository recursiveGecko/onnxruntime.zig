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

pub const OnnxInstanceOpts = struct {
    model_path: [:0]const u8,
    log_level: OrtLoggingLevel,
    log_id: [:0]const u8,
    input_names: []const [:0]const u8,
    output_names: []const [:0]const u8,
};

pub const OnnxInstance = struct {
    const Self = @This();

    allocator: Allocator,
    ort_api: *const c_api.OrtApi,
    ort_env: *c_api.OrtEnv,
    session_opts: *c_api.OrtSessionOptions,
    session: *c_api.OrtSession,
    run_opts: ?*c_api.OrtRunOptions,
    input_names: []const [*:0]const u8,
    output_names: []const [*:0]const u8,
    mem_info: ?*c_api.OrtMemoryInfo = null,
    ort_inputs: ?[]*c_api.OrtValue = null,
    ort_outputs: ?[]?*c_api.OrtValue = null,

    pub fn init(
        allocator: Allocator,
        options: OnnxInstanceOpts,
    ) !*Self {
        var ort_api = c_api.OrtGetApiBase().*.GetApi.?(c_api.ORT_API_VERSION);

        const ort_env = try createEnv(ort_api, options);
        const session_opts = try createSessionOptions(ort_api);
        const session = try createSession(ort_api, ort_env, session_opts, options);
        const run_opts = try createRunOptions(ort_api);

        var self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        var input_names = try allocator.alloc([*:0]const u8, options.input_names.len);
        errdefer allocator.free(input_names);
        for (0..input_names.len) |i| {
            input_names[i] = options.input_names[i].ptr;
        }

        var output_names = try allocator.alloc([*:0]const u8, options.output_names.len);
        errdefer allocator.free(output_names);
        for (0..output_names.len) |i| {
            output_names[i] = options.output_names[i].ptr;
        }

        self.* = Self{
            .allocator = allocator,
            .ort_api = ort_api,
            .ort_env = ort_env,
            .session_opts = session_opts,
            .session = session,
            .run_opts = run_opts,
            .input_names = input_names,
            .output_names = output_names,
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.ort_inputs.?);
        self.allocator.free(self.ort_outputs.?);
        self.allocator.free(self.input_names);
        self.allocator.free(self.output_names);
        self.allocator.destroy(self);
    }

    pub fn initMemoryInfo(
        self: *Self,
        name: [:0]const u8,
        allocator_type: OrtAllocatorType,
        id: i32,
        mem_type: OrtMemType,
    ) !void {
        if (self.mem_info != null) @panic("Memory info already created");

        var mem_info: ?*c_api.OrtMemoryInfo = null;
        const status = self.ort_api.CreateMemoryInfo.?(
            name.ptr,
            @enumToInt(allocator_type),
            id,
            @enumToInt(mem_type),
            &mem_info,
        );

        try checkError(self.ort_api, status);
        self.mem_info = mem_info;
    }

    pub fn setManagedInputsOutputs(
        self: *Self,
        inputs: []*c_api.OrtValue,
        outputs: []?*c_api.OrtValue,
    ) void {
        if (self.ort_inputs != null) @panic("Inputs already set");
        if (self.ort_outputs != null) @panic("Outputs already set");

        self.ort_inputs = inputs;
        self.ort_outputs = outputs;
    }

    pub fn createTensorWithDataAsOrtValue(
        self: *Self,
        comptime T: type,
        data: []T,
        shape: []const i64,
        tensor_type: ONNXTensorElementDataType,
    ) !*c_api.OrtValue {
        var value: ?*c_api.OrtValue = null;
        const status = self.ort_api.CreateTensorWithDataAsOrtValue.?(
            self.mem_info.?,
            data.ptr,
            data.len * @sizeOf(T),
            shape.ptr,
            shape.len,
            @enumToInt(tensor_type),
            &value,
        );

        try checkError(self.ort_api, status);
        return value.?;
    }

    pub fn run(self: *Self) !void {
        const status = self.ort_api.Run.?(
            self.session,
            self.run_opts,
            self.input_names.ptr,
            self.ort_inputs.?.ptr,
            self.ort_inputs.?.len,
            self.output_names.ptr,
            self.output_names.len,
            self.ort_outputs.?.ptr,
        );

        try checkError(self.ort_api, status);
    }

    pub fn checkError(
        ort_api: *const c_api.OrtApi,
        onnx_status: ?*c_api.OrtStatus,
    ) !void {
        if (onnx_status == null) return;
        defer ort_api.ReleaseStatus.?(onnx_status);

        const msg = ort_api.GetErrorMessage.?(onnx_status);
        std.debug.print("ONNX error: {s}\n", .{msg});

        return error.OnnxError;
    }

    fn createEnv(
        ort_api: *const c_api.OrtApi,
        options: OnnxInstanceOpts,
    ) !*c_api.OrtEnv {
        var ort_env: ?*c_api.OrtEnv = null;
        const status = ort_api.CreateEnv.?(
            @enumToInt(options.log_level),
            options.log_id.ptr,
            &ort_env,
        );

        try checkError(ort_api, status);
        return ort_env.?;
    }

    fn createSessionOptions(
        ort_api: *const c_api.OrtApi,
    ) !*c_api.OrtSessionOptions {
        var ort_sess_opts: ?*c_api.OrtSessionOptions = null;
        const status = ort_api.CreateSessionOptions.?(&ort_sess_opts);

        try checkError(ort_api, status);
        return ort_sess_opts.?;
    }

    fn createSession(
        ort_api: *const c_api.OrtApi,
        ort_env: *c_api.OrtEnv,
        ort_sess_opts: *c_api.OrtSessionOptions,
        options: OnnxInstanceOpts,
    ) !*c_api.OrtSession {
        var ort_sess: ?*c_api.OrtSession = null;
        const status = ort_api.CreateSession.?(
            ort_env,
            options.model_path.ptr,
            ort_sess_opts,
            &ort_sess,
        );

        try checkError(ort_api, status);
        return ort_sess.?;
    }

    fn createRunOptions(
        ort_api: *const c_api.OrtApi,
    ) !*c_api.OrtRunOptions {
        var run_opts: ?*c_api.OrtRunOptions = null;
        const status = ort_api.CreateRunOptions.?(&run_opts);

        try checkError(ort_api, status);
        return run_opts.?;
    }
};
