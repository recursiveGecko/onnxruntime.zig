const std = @import("std");
const onnx = @import("onnxruntime");
const AudioFileStream = @import("AudioFileStream.zig");

const example_wav_path = srcDir() ++ "/../data/example.wav";
const silero_model_path = srcDir() ++ "/../data/silero_vad.onnx";

const threshold = 0.5;
const min_speech_duration_ms: f32 = 250;
const max_silence_duration_ms: f32 = 100;

inline fn srcDir() []const u8 {
    return std.fs.path.dirname(@src().file).?;
}

const InferenceState = struct {
    allocator: std.mem.Allocator,
    onnx_instance: *onnx.OnnxInstance,
    audio_stream: AudioFileStream,
    audio_read_buffer: [][]f32,
    audio_read_n_frames: usize,
    /// Progress this many samples for every sample as a rudiementary form of downsampling
    /// e.g. if this is 3, then we will skip 2 samples after every sample, effectively
    /// downsampling the audio from 48kHz to 16kHz
    sample_tick_rate: usize,
    window_size: usize,
    // Model inputs/outputs
    pcm: []f32,
    h: []f32,
    c: []f32,
    vad: []f32,
    hn: []f32,
    cn: []f32,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const input_file_path = example_wav_path;
    var audio_stream = try AudioFileStream.open(allocator, input_file_path);
    defer audio_stream.close();

    // Hard-coded sample rate, this is what the model needs
    const sample_rate = 16000;
    const window_size_samples: i64 = 1024; // 64ms
    const sample_tick_rate = val: {
        if (audio_stream.sample_rate % 16000 != 0) {
            @panic("Sample rate must be a multiple of 16000\n");
        }

        break :val audio_stream.sample_rate / 16000;
    };

    // Audio read buffers
    const audio_read_n_frames = sample_tick_rate * window_size_samples;
    var audio_read_buffer = try allocator.alloc([]f32, audio_stream.n_channels);
    defer {
        for (0..audio_stream.n_channels) |i| allocator.free(audio_read_buffer[i]);
        allocator.free(audio_read_buffer);
    }
    for (0..audio_stream.n_channels) |i| {
        audio_read_buffer[i] = try allocator.alloc(f32, audio_read_n_frames);
    }

    // Initialize ONNX runtime

    const onnx_opts = onnx.OnnxInstanceOpts{
        .log_id = "ZIG",
        .log_level = .warning,
        .model_path = silero_model_path,
        .input_names = &.{ "input", "sr", "h", "c" },
        .output_names = &.{ "output", "hn", "cn" },
    };
    var onnx_instance = try onnx.OnnxInstance.init(allocator, onnx_opts);
    try onnx_instance.initMemoryInfo("Cpu", .arena, 0, .default);
    defer onnx_instance.deinit();

    // PCM input
    var pcm_node_dimms: []const i64 = &.{ 1, window_size_samples };
    var pcm: [window_size_samples]f32 = undefined;
    @memset(&pcm, 0);
    var pcm_ort_input = try onnx_instance.createTensorWithDataAsOrtValue(
        f32,
        &pcm,
        pcm_node_dimms,
        .f32,
    );

    // Sample rate input
    var sr_node_dimms: []const i64 = &.{1};
    var sr = [1]i64{sample_rate};
    var sr_ort_input = try onnx_instance.createTensorWithDataAsOrtValue(
        i64,
        &sr,
        sr_node_dimms,
        .i64,
    );

    // Hidden and cell state inputs
    const size_hc: usize = 2 * 1 * 64;
    var hc_node_dimms: []const i64 = &.{ 2, 1, 64 };

    var h: [size_hc]f32 = undefined;
    @memset(&h, 0);
    var h_ort_input = try onnx_instance.createTensorWithDataAsOrtValue(
        f32,
        &h,
        hc_node_dimms,
        .f32,
    );

    var c: [size_hc]f32 = undefined;
    @memset(&c, 0);
    var c_ort_input = try onnx_instance.createTensorWithDataAsOrtValue(
        f32,
        &c,
        hc_node_dimms,
        .f32,
    );

    const ort_inputs = try allocator.dupe(*onnx.c_api.OrtValue, &.{
        pcm_ort_input,
        sr_ort_input,
        h_ort_input,
        c_ort_input,
    });

    // Set up outputs
    var vad = [1]f32{1};
    var vad_ort_output = try onnx_instance.createTensorWithDataAsOrtValue(
        f32,
        &vad,
        &.{ 1, 1 },
        .f32,
    );

    var hn: [size_hc]f32 = undefined;
    @memset(&hn, 0);
    var hn_ort_output = try onnx_instance.createTensorWithDataAsOrtValue(
        f32,
        &hn,
        hc_node_dimms,
        .f32,
    );

    var cn: [size_hc]f32 = undefined;
    @memset(&cn, 0);
    var cn_ort_output = try onnx_instance.createTensorWithDataAsOrtValue(
        f32,
        &cn,
        hc_node_dimms,
        .f32,
    );

    var ort_outputs = try allocator.dupe(?*onnx.c_api.OrtValue, &.{
        vad_ort_output,
        hn_ort_output,
        cn_ort_output,
    });

    onnx_instance.setManagedInputsOutputs(ort_inputs, ort_outputs);

    var inference_state = InferenceState{
        .allocator = allocator,
        .audio_stream = audio_stream,
        .audio_read_n_frames = audio_read_n_frames,
        .audio_read_buffer = audio_read_buffer,
        .sample_tick_rate = sample_tick_rate,
        .window_size = window_size_samples,
        // model inputs/outputs
        .onnx_instance = onnx_instance,
        .pcm = &pcm,
        .h = &h,
        .c = &c,
        .vad = &vad,
        .hn = &hn,
        .cn = &cn,
    };

    try runInference(&inference_state);
}

fn runInference(state: *InferenceState) !void {
    var ts1_buf = try state.allocator.alloc(u8, 32);
    var ts2_buf = try state.allocator.alloc(u8, 32);
    defer state.allocator.free(ts1_buf);
    defer state.allocator.free(ts2_buf);

    const sample_rate_f = @intToFloat(f32, state.audio_stream.sample_rate);

    // Parameters
    const channel = 0;
    const min_speech_samples = @floatToInt(
        usize,
        sample_rate_f * min_speech_duration_ms / 1000,
    );
    const max_silence_samples = @floatToInt(
        usize,
        sample_rate_f * max_silence_duration_ms / 1000,
    );
    const on_threshold = threshold;
    const off_threshold = on_threshold * 0.85;

    // State
    var start_sample: ?usize = null;
    var end_sample: ?usize = null;

    var frames_processed: usize = 0;
    while (true) {
        // Read a chunk of audio
        const n_read = try state.audio_stream.read(state.audio_read_buffer, 0, state.audio_read_n_frames);

        // Copy one of the channels into the input buffer
        // Apply downsampling by skipping samples
        for (0..state.window_size) |i| {
            const frame_idx = i * state.sample_tick_rate;
            state.pcm[i] = state.audio_read_buffer[channel][frame_idx];
        }

        try state.onnx_instance.run();

        // Output VAD value
        const vad = state.vad[0];

        // 1. Start speaking
        if (vad >= on_threshold and start_sample == null) {
            start_sample = frames_processed;
        }

        // 2. Maybe stop speaking
        if (vad < off_threshold and end_sample == null) {
            end_sample = frames_processed;
        }

        // 3. Continue speaking
        if (vad >= on_threshold and end_sample != null) {
            end_sample = null;
        }

        // 4. Stop speaking
        if (start_sample != null and
            end_sample != null and
            frames_processed - end_sample.? >= max_silence_samples)
        {
            if (end_sample.? - start_sample.? >= min_speech_samples) {
                const ts1 = try formatTs(ts1_buf, start_sample.?, state.audio_stream.sample_rate);
                const ts2 = try formatTs(ts2_buf, end_sample.?, state.audio_stream.sample_rate);
                std.debug.print("Speech: {s} - {s}\n", .{ ts1, ts2 });
            }

            start_sample = null;
            end_sample = null;
        }

        if (n_read < state.audio_read_n_frames) break;
        frames_processed += n_read;

        // Copy the hidden and cell states for the next iteration
        @memcpy(state.h, state.hn);
        @memcpy(state.c, state.cn);
    }
}

fn formatTs(buf: []u8, frame_idx: usize, sample_rate: usize) ![]const u8 {
    const frame_idx_f = @intToFloat(f32, frame_idx);
    const sample_rate_f = @intToFloat(f32, sample_rate);

    const secondsTotal = frame_idx_f / sample_rate_f;

    const hours = @floatToInt(u64, secondsTotal / 3600);
    const minutes = @floatToInt(u64, (secondsTotal - @intToFloat(f32, hours * 3600)) / 60);
    const seconds = @floatToInt(u64, @rem(secondsTotal, 60));
    const ms = @floatToInt(u64, secondsTotal * 1000) % 1000;

    var ts = try std.fmt.bufPrint(buf, "{d:0>2}:{d:0>2}:{d:0>2}.{d:0>3}", .{ hours, minutes, seconds, ms });
    return ts;
}
