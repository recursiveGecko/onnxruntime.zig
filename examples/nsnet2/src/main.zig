const std = @import("std");
const Allocator = std.mem.Allocator;
const pow = std.math.pow;
const onnx = @import("onnxruntime");
const AudioFileStream = @import("AudioFileStream.zig");
const FFT = @import("FFT.zig");
const window_fn = @import("window_fn.zig");

const example_wav_path = srcDir() ++ "/../data/example3.wav";
const output_wav_path = srcDir() ++ "/../data/example3.out.wav";
const nsnet_model_path = srcDir() ++ "/../data/nsnet2-20ms-baseline.onnx";

inline fn srcDir() []const u8 {
    return std.fs.path.dirname(@src().file).?;
}

const OnnxState = struct {
    allocator: std.mem.Allocator,
    ort_api: *onnx.OrtApi,
    ort_session: *onnx.c_api.OrtSession,
    ort_run_options: *onnx.c_api.OrtRunOptions,

    ort_inputs: [1]*onnx.c_api.OrtValue,
    input_names: []const [*:0]const u8,
    features: []f32,
    ort_outputs: [1]?*onnx.c_api.OrtValue,
    output_names: []const [*:0]const u8,
    // shape equal to specgram & features
    gains: []f32,

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.features);
        self.allocator.free(self.gains);
        self.ort_api.deinit();
    }
};

const TempBuffers = struct {
    allocator: Allocator,
    // Audio file read buffer, multi-channel
    audio_read_buffer: []const []f32,
    // Audio input buffer after it's downsampled to 16kHz, single channel
    audio_input: []f32,
    // Audio output buffer
    audio_output: []f32,
    // Stores forward FFT output
    specgram: []FFT.Complex,
    // Stores inverse FFT output
    inv_fft_buffer: []f32,

    pub fn deinit(self: *@This()) void {
        for (0..self.audio_read_buffer.len) |i| {
            self.allocator.free(self.audio_read_buffer[i]);
        }
        self.allocator.free(self.audio_read_buffer);
        self.allocator.free(self.audio_input);
        self.allocator.free(self.audio_output);
        self.allocator.free(self.specgram);
        self.allocator.free(self.inv_fft_buffer);
    }
};

const InferenceState = struct {
    allocator: std.mem.Allocator,
    onnx_state: OnnxState,
    audio_input_stream: *AudioFileStream,
    audio_output_stream: *AudioFileStream,
    fwd_fft: *FFT,
    inv_fft: *FFT,
    /// Progress this many samples for every sample as a rudiementary form of downsampling
    /// e.g. if this is 3, we use every 3rd sample and throw away the rest, effectively
    /// downsampling the audio from 48kHz to 16kHz
    sample_tick_rate: usize,
    // FFT size
    n_fft: usize,
    // hop size
    n_hop: usize,
    // number of samples to process per iteration/inference run, multiple of n_hop
    chunk_size: usize,
    // length equal to n_fft
    window: []const f32,
    temp_buffers: TempBuffers,
};

///
/// Entrypoint
///
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var allocator = gpa.allocator();
    defer _ = gpa.deinit();

    //
    // Initialize audio preprocessing parameters and buffers
    //

    // These parameters are fixed in the original implementation
    const n_fft = 320;
    const n_hop = 160;
    // 16kHz is the only supported sample rate
    const sample_rate = 16000;

    // Number of samples to process per iteration, this can be adjusted to
    // trade between latency and overhead/throughput
    const chunk_size = n_hop * 50;

    //
    // Initialize audio input stream
    //
    var audio_input_stream = try AudioFileStream.openForRead(allocator, example_wav_path);
    defer audio_input_stream.close();

    var audio_output_stream = try AudioFileStream.openForWrite(allocator, output_wav_path, 1, 16000);
    defer audio_output_stream.close();

    const sample_tick_rate = val: {
        if (audio_input_stream.sample_rate % 16000 != 0) {
            @panic("Sample rate must be a multiple of 16000\n");
        }
        break :val audio_input_stream.sample_rate / 16000;
    };

    //
    // Initialize ONNX runtime
    //

    var onnx_state = try initOnnx(allocator, chunk_size, n_fft, n_hop);
    defer onnx_state.deinit();

    //
    // Initialize forward and inverse FFT instances and the window function
    //

    var fwd_fft = try FFT.init(allocator, n_fft, sample_rate, false);
    defer fwd_fft.deinit();

    var inv_fft = try FFT.init(allocator, n_fft, sample_rate, true);
    defer inv_fft.deinit();

    const window = try createWindow(allocator, n_fft);
    defer allocator.free(window);

    var temp_buffers = try initTempBuffers(
        allocator,
        n_fft,
        n_hop,
        chunk_size,
        audio_input_stream.n_channels,
        sample_tick_rate,
    );
    defer temp_buffers.deinit();

    //
    // Initialize ONNX runtime
    //

    var inference_state = InferenceState{
        .allocator = allocator,
        .audio_input_stream = audio_input_stream,
        .audio_output_stream = audio_output_stream,
        .fwd_fft = &fwd_fft,
        .inv_fft = &inv_fft,
        .sample_tick_rate = sample_tick_rate,
        .n_fft = n_fft,
        .n_hop = n_hop,
        .chunk_size = chunk_size,
        .window = window,
        .onnx_state = onnx_state,
        .temp_buffers = temp_buffers,
    };

    try runInference(&inference_state);
}

fn initOnnx(
    allocator: std.mem.Allocator,
    chunk_size: usize,
    n_fft: usize,
    n_hop: usize,
) !OnnxState {
    var ort_api = try onnx.OrtApi.init(allocator);
    var ort_env = try ort_api.createEnv(.warning, "ZIG");
    var sess_opts = try ort_api.createSessionOptions();
    var ort_sess = try ort_api.createSession(ort_env, nsnet_model_path, sess_opts);
    var mem_info = try ort_api.createMemoryInfo("Cpu", .arena, 0, .default);
    var run_opts = try ort_api.createRunOptions();

    // Number of frames per input audio chunk
    const n_frames = calcNFrames(chunk_size, n_fft, n_hop);
    // Number of spectrogram bins
    const n_bins = calcNBins(n_fft);

    if (n_bins != 161) {
        // There's a mismatch between our code and original implementation
        // The number of bins is hardcoded to 161 in the ONNX model
        @panic("Invalid number of FFT bins");
    }

    //
    // Spectrogram input
    //
    var features_node_dimms: []const i64 = &.{
        1,
        @intCast(i64, n_frames),
        @intCast(i64, n_bins),
    };
    var features = try allocator.alloc(f32, n_frames * n_bins);
    errdefer allocator.free(features);
    @memset(features, 0);
    var features_ort_input = try ort_api.createTensorWithDataAsOrtValue(
        f32,
        mem_info,
        features,
        features_node_dimms,
        .f32,
    );
    const input_names: []const [*:0]const u8 = &.{"input"};
    const ort_inputs = [_]*onnx.c_api.OrtValue{
        features_ort_input,
    };

    //
    // Gain output
    //
    var gain_node_dimms: []const i64 = &.{
        1,
        @intCast(i64, n_frames),
        @intCast(i64, n_bins),
    };
    var gains = try allocator.alloc(f32, n_frames * n_bins);
    errdefer allocator.free(gains);
    var gains_ort_output = try ort_api.createTensorWithDataAsOrtValue(
        f32,
        mem_info,
        gains,
        gain_node_dimms,
        .f32,
    );
    const output_names: []const [*:0]const u8 = &.{"output"};
    var ort_outputs = [_]?*onnx.c_api.OrtValue{
        gains_ort_output,
    };

    return OnnxState{
        .allocator = allocator,
        .ort_api = ort_api,
        .ort_session = ort_sess,
        .ort_run_options = run_opts,

        .ort_inputs = ort_inputs,
        .input_names = input_names,
        .features = features,
        .ort_outputs = ort_outputs,
        .output_names = output_names,
        .gains = gains,
    };
}

fn initTempBuffers(
    allocator: Allocator,
    n_fft: usize,
    n_hop: usize,
    chunk_size: usize,
    n_channels: usize,
    downsample_rate: usize,
) !TempBuffers {
    const n_frames = calcNFrames(chunk_size, n_fft, n_hop);
    const n_bins = calcNBins(n_fft);

    var audio_read_buffer = try allocator.alloc([]f32, n_channels);
    for (0..n_channels) |i| {
        audio_read_buffer[i] = try allocator.alloc(f32, chunk_size * downsample_rate);
    }

    var audio_input = try allocator.alloc(f32, chunk_size + n_hop);
    @memset(audio_input, 0);

    var audio_output = try allocator.alloc(f32, chunk_size + n_hop);
    @memset(audio_output, 0);

    var specgram = try allocator.alloc(FFT.Complex, n_frames * n_bins);
    @memset(specgram, FFT.Complex{.r = 0, .i = 0});

    var inv_fft_buffer = try allocator.alloc(f32, n_fft);
    @memset(inv_fft_buffer, 0);

    return TempBuffers{
        .allocator = allocator,
        .audio_read_buffer = audio_read_buffer,
        .audio_input = audio_input,
        .audio_output = audio_output,
        .specgram = specgram,
        .inv_fft_buffer = inv_fft_buffer,
    };
}

///
/// Main inference loop
///
fn runInference(state: *InferenceState) !void {
    const downsample_rate = state.sample_tick_rate;
    const chunk_size = state.chunk_size;
    const n_fft = state.n_fft;
    const n_hop = state.n_hop;
    const n_frames = calcNFrames(chunk_size, n_fft, n_hop);

    const buffers = state.temp_buffers;

    //
    // Create logical slices into the input and output buffers
    // for easier access to first and last `n_hop` samples (overlap)
    //
    var in_last_hop: []f32 = buffers.audio_input[chunk_size .. chunk_size + n_hop];
    var in_first_hop: []f32 = buffers.audio_input[0..n_hop];
    // This is where the new downsampled audio will be stored, first `n_hop` samples are
    // skipped because they are copied from previous iteration
    var in_read_slice: []f32 = buffers.audio_input[n_hop..];

    var out_last_hop: []f32 = buffers.audio_output[chunk_size .. chunk_size + n_hop];
    var out_first_hop: []f32 = buffers.audio_output[0..n_hop];
    var out_completed_slice: []f32 = buffers.audio_output[0..chunk_size];
    var out_after_first_hop: []f32 = buffers.audio_output[n_hop..];

    while (true) {
        // Copy the last n_hop samples from the previous chunk to the beginning
        // of the next chunk for overlap
        @memcpy(in_first_hop, in_last_hop);
        @memcpy(out_first_hop, out_last_hop);

        // We don't need to zero the INput buffer because overwritten during downsampling
        // Zero out the OUTput buffer excluding the last n_hop samples for overlap
        @memset(out_after_first_hop, 0);

        const n_to_read = downsample_rate * chunk_size;
        // Read a chunk of audio
        const n_read = try state.audio_input_stream.read(
            buffers.audio_read_buffer,
            0,
            n_to_read,
        );

        if (n_read < n_to_read) {
            // We've reached the end of the audio stream
            // In a production system we'd probably want to form as many
            // frames as we can and apply padding to the last frame
            // but this is good enough for example purposes.
            // Up to 0.1s of audio will be lost at the end of the stream
            return;
        }

        downsampleAudio(
            // Only use the first channel, ignore the rest
            buffers.audio_read_buffer[0],
            in_read_slice,
            downsample_rate,
        );

        try calcSpectrogram(
            state.fwd_fft,
            buffers.audio_input,
            n_frames,
            n_fft,
            n_hop,
            state.window,
            buffers.specgram,
        );

        var os = state.onnx_state;

        calcFeatures(buffers.specgram, os.features);
        try os.ort_api.run(
            os.ort_session,
            os.ort_run_options,
            os.input_names,
            &os.ort_inputs,
            os.output_names,
            &os.ort_outputs,
        );
        applySpecgramGain(buffers.specgram, os.gains);

        try reconstructAudio(
            state.inv_fft,
            buffers.specgram,
            n_fft,
            n_hop,
            state.window,
            buffers.inv_fft_buffer,
            buffers.audio_output,
        );

        try state.audio_output_stream.write(&.{out_completed_slice});
    }
}

pub fn downsampleAudio(
    input_samples: []f32,
    output_samples: []f32,
    downsample_rate: usize,
) void {
    if (input_samples.len != output_samples.len * downsample_rate) {
        @panic("Invalid downsampling inputs");
    }

    const n_steps = input_samples.len / downsample_rate;

    for (0..n_steps) |i| {
        output_samples[i] = input_samples[i * downsample_rate];
    }
}

/// Calculates the spectrogram of the given samples
/// Input length must be a multiple of n_hop
///
pub fn calcSpectrogram(
    fft: *FFT,
    audio_chunk: []const f32,
    n_frames: usize,
    n_fft: usize,
    n_hop: usize,
    window: []const f32,
    result_specgram: []FFT.Complex,
) !void {
    const n_bins = calcNBins(n_fft);

    for (0..n_frames) |frame_idx| {
        const in_start_idx = frame_idx * n_hop;
        const in_end_idx = in_start_idx + n_fft;
        const input_frame = audio_chunk[in_start_idx..in_end_idx];

        const out_start_idx = frame_idx * n_bins;
        const out_end_idx = out_start_idx + n_bins;
        const output_bins = result_specgram[out_start_idx..out_end_idx];

        // Applies the window function, computes the FFT, and stores
        // a real-valued result equal to sqrt(re^2 + im^2) into output_bins
        try fft.fft(input_frame, window, output_bins);
    }
}

pub fn calcFeatures(
    specgram: []const FFT.Complex,
    result_features: []f32,
) void {
    if (specgram.len != result_features.len) {
        @panic("specgram and features must have the same length");
    }

    // Original implementation: calcFeat() in featurelib.py (LogPow)
    const p_min = std.math.pow(f32, 10, -12);

    // Calculate Log10 of the power spectrum
    for (0..specgram.len) |i| {
        const bin = specgram[i];

        const pow_spec = pow(f32, bin.r, 2) + pow(f32, bin.i, 2);
        const p_out = @max(pow_spec, p_min);
        const log_p_out = std.math.log(f32, 10, p_out);

        result_features[i] = log_p_out;
    }
}

pub fn applySpecgramGain(
    specgram: []FFT.Complex,
    gains: []f32,
) void {
    std.debug.assert(specgram.len == gains.len);

    const p_min = -80;
    const p_max = 1;

    for (0..gains.len) |i| {
        var el_gain: f32 = gains[i];

        if (el_gain < p_min) {
            el_gain = p_min;
        } else if (el_gain > p_max) {
            el_gain = p_max;
        }

        specgram[i].r *= el_gain;
        specgram[i].i *= el_gain;
    }
}

pub fn reconstructAudio(
    fft: *FFT,
    specgram: []FFT.Complex,
    n_fft: usize,
    n_hop: usize,
    window: []const f32,
    inv_fft_buffer: []f32,
    audio_output: []f32,
) !void {
    const n_bins = calcNBins(n_fft);
    const n_frames = specgram.len / n_bins;

    // Volume normalization factor
    const vol_norm_factor: f32 = 1 / @intToFloat(f32, n_fft);

    for (0..n_frames) |frame_idx| {
        const in_start_idx = frame_idx * n_bins;
        const in_end_idx = in_start_idx + n_bins;
        const input_bins = specgram[in_start_idx..in_end_idx];

        try fft.invFft(input_bins, inv_fft_buffer);

        const out_start_idx = frame_idx * n_hop;

        for (0..n_fft) |i| {
            inv_fft_buffer[i] *= window[i] * vol_norm_factor;
            audio_output[out_start_idx + i] += inv_fft_buffer[i];
        }
    }
}

/// Original:
/// N_frames = int(np.ceil( (Nx+N_win-N_hop)/N_hop ))
///
/// n_fft (N_win) is always 320 and n_hop (N_hop) is always 160, meaning that
/// (n_fft - n_hop) is always 160. This in turn means that we can simplify the
/// calculation to simple integer division if we ensure that the input length is
/// a multiple of n_hop (160) too.
///
/// Simplification of the calculation done in the original python code.
/// We subtract 1 from the result because we won't be padding the input,
/// instead the last n_hop samples of the previous chunk will be copied to the
/// beginning of the next chunk.
///
/// Consider n_samples = 5, n_fft = 2, and n_hop = 1, with boxes [ ] representing samples:
///
/// Fr\Samp: [C][1][2][3][4][5]
/// #1:       C  x
/// #2:          x  x
/// #3:             x  x
/// #4:                x  x
/// #5:                   x  x
///  C:                      C
/// We can form 5 frames from 5 samples without padding, with C representing the carry-over
/// to the next chunk.
pub fn calcNFrames(
    n_samples: usize,
    n_fft: usize,
    n_hop: usize,
) usize {
    if (n_samples % n_hop != 0) {
        @panic("n_samples must be a multiple of n_hop");
    }

    if (n_hop != n_fft / 2) {
        @panic("n_hop must be equal to n_fft / 2");
    }

    return n_samples / n_hop;
}

pub fn calcNBins(n_fft: usize) usize {
    return n_fft / 2 + 1;
}

// Original: featurelib.py - calcSpec()
pub fn createWindow(
    allocator: std.mem.Allocator,
    n_fft: usize,
) ![]f32 {
    const window = try allocator.alloc(f32, n_fft);
    errdefer allocator.free(window);

    window_fn.hannWindowSymmetric(window);
    for (0..window.len) |i| {
        window[i] = std.math.sqrt(window[i]);
    }

    return window;
}
