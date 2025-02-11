//! Wrapper around the KissFFT library

const std = @import("std");
const Allocator = std.mem.Allocator;
const kissfft = @cImport({
    @cInclude("_kiss_fft_guts.h");
    @cInclude("kiss_fft.h");
    @cInclude("kiss_fftr.h");
});
const window_fn = @import("./window_fn.zig");

pub const Complex = extern struct {
    r: f32,
    i: f32,

    pub fn abs(self: @This()) f32 {
        return @sqrt(self.r * self.r + self.i * self.i);
    }
};

const Self = @This();

// =============
// Struct fields
// =============
allocator: Allocator,
kiss_cfg: kissfft.kiss_fftr_cfg = null,
buf_real: []f32,
buf_cpx: []kissfft.kiss_fft_cpx,
n_fft: usize,
sample_rate: usize,
mode_inverse: bool,

/// Initialize a new reusable FFT instance for given FFT size and sample rate.
pub fn init(
    allocator: Allocator,
    n_fft: usize,
    sample_rate: usize,
    mode_inverse: bool,
) !Self {
    if (n_fft == 0 or @mod(n_fft, 2) != 0) {
        return error.InvalidFFTSize;
    }

    // Allocate input and output buffers
    const buf_real = try allocator.alloc(f32, n_fft);
    errdefer allocator.free(buf_real);

    const buf_cpx = try allocator.alloc(kissfft.kiss_fft_cpx, n_fft);
    errdefer allocator.free(buf_cpx);

    const kiss_cfg = kissfft.kiss_fftr_alloc(
        @intCast(n_fft),
        @intFromBool(mode_inverse),
        null,
        null,
    );
    if (kiss_cfg == null) {
        return error.KissFFTAllocFailed;
    }

    const self = Self{
        .allocator = allocator,
        .n_fft = n_fft,
        .kiss_cfg = kiss_cfg,
        .buf_real = buf_real,
        .buf_cpx = buf_cpx,
        .sample_rate = sample_rate,
        .mode_inverse = mode_inverse,
    };

    return self;
}

pub fn deinit(self: *Self) void {
    kissfft.kiss_fftr_free(self.kiss_cfg);
    self.allocator.free(self.buf_real);
    self.allocator.free(self.buf_cpx);
}

pub fn fft(
    self: *Self,
    samples: []const f32,
    window: []const f32,
    result: []Complex,
) !void {
    if (samples.len != self.n_fft) {
        return error.InvalidSamplesLength;
    }

    if (window.len != self.n_fft) {
        return error.InvalidWindowLength;
    }

    if (result.len != self.binCount()) {
        return error.InvalidResultLength;
    }

    // Applies the window function and loads the samples into the KissFFT input buffer
    const in_samples = self.loadSamplesFwd(samples, window);

    // Run FFT
    kissfft.kiss_fftr(self.kiss_cfg, in_samples.ptr, @ptrCast(result.ptr));
}

pub fn invFft(
    self: *Self,
    bins: []const Complex,
    result: []f32,
) !void {
    if (bins.len != self.binCount()) {
        return error.InvalidBinsLength;
    }

    if (result.len != self.n_fft) {
        return error.InvalidResultLength;
    }

    // Run FFT
    kissfft.kiss_fftri(
        self.kiss_cfg,
        @as(*const kissfft.kiss_fft_cpx, @ptrCast(bins.ptr)),
        result.ptr,
    );
}

/// Query the number of usable bins in the FFT output
pub fn binCount(self: Self) usize {
    return (self.n_fft / 2) + 1;
}

/// Query the width of each bin in Hz
pub fn binWidth(self: Self) f32 {
    const sample_rate_f: f32 = @floatFromInt(self.sample_rate);
    const n_fft_f: f32 = @floatFromInt(self.n_fft);

    return sample_rate_f / n_fft_f;
}

/// Query the Nyquist frequency of the FFT
pub fn nyquistFreq(self: Self) f32 {
    const sample_rate_f: f32 = @floatFromInt(self.sample_rate);
    return sample_rate_f / 2;
}

/// Converts given frequency in Hz to the nearest FFT bin index
pub fn freqToBin(self: Self, freq: f32) !usize {
    if (freq > self.nyquistFreq()) {
        return error.OutOfRange;
    }

    if (freq < 0) {
        return error.NegativeFrequency;
    }

    const bin_f = @round(freq / self.binWidth());
    return @intFromFloat(bin_f);
}

/// Converts given FFT bin index to the corresponding frequency in Hz
pub fn binToFreq(self: Self, bin_index: usize) !f32 {
    const max_bin_index = self.binCount() - 1;

    if (bin_index > max_bin_index) {
        return error.OutOfRange;
    }

    const bin_f: f32 = @floatFromInt(bin_index);
    const bin_width = self.binWidth();
    return bin_f * bin_width;
}

/// Loads samples into the KissFFT input buffer for forward FFT
fn loadSamplesFwd(
    self: *Self,
    samples: []const f32,
    window: []const f32,
) []const f32 {
    std.debug.assert(samples.len == window.len);

    for (samples, 0..) |sample, idx| {
        self.buf_real[idx] = sample * window[idx];
    }

    return self.buf_real;
}
