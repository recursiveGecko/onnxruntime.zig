const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const sndfile = @cImport({
    @cInclude("sndfile.h");
});

const Self = @This();

allocator: Allocator,
sf_info: sndfile.SF_INFO,
sf_file: ?*sndfile.SNDFILE,
n_channels: usize,
sample_rate: usize,
length: usize,
interleaved_buffer: []f32,

pub fn open(allocator: Allocator, path: []const u8) !Self {
    const path_Z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_Z);

    var sf_info = std.mem.zeroInit(sndfile.SF_INFO, .{});
    var sf_file = sndfile.sf_open(path_Z.ptr, sndfile.SFM_READ, &sf_info);
    errdefer _ = sndfile.sf_close(sf_file);

    if (sf_file == null) {
        return error.SndfileOpenError;
    }

    const n_channels = @intCast(usize, sf_info.channels);
    const sample_rate = @intCast(usize, sf_info.samplerate);
    const length = @intCast(usize, sf_info.frames);

    var interleaved_buffer = try allocator.alloc(f32, n_channels * sample_rate);
    errdefer allocator.free(interleaved_buffer);

    var self = Self{
        .allocator = allocator,
        .sf_info = sf_info,
        .sf_file = sf_file,
        .n_channels = n_channels,
        .sample_rate = sample_rate,
        .length = length,
        .interleaved_buffer = interleaved_buffer,
    };

    return self;
}

/// Reads a given maximum number of frames from the file and writes them into
/// the given destination buffer, starting at the given offset.
/// Returns the number of frames read.
/// Returns an error if the destination is full, callers should close the stream
/// when the number of frames read is less than the number of frames expected.
///
pub fn read(self: *Self, result_pcm: [][]f32, result_offset: usize, max_frames: usize) !usize {
    if (self.sf_file == null) {
        return error.FileNotOpen;
    }
    const sf_file = self.sf_file.?;

    assert(result_pcm.len == self.n_channels);

    const rem_result_size = result_pcm[0].len - result_offset;
    const total_frames_to_read = @min(max_frames, rem_result_size);

    if (rem_result_size == 0) {
        return error.DestinationBufferFull;
    }

    const max_frames_per_step = self.interleaved_buffer.len / self.n_channels;

    var total_read_count: usize = 0;
    while (true) {
        const frames_to_read = @min(max_frames_per_step, total_frames_to_read - total_read_count);
        
        // Read samples into the interleaved buffer
        const c_frames_read = sndfile.sf_readf_float(sf_file, self.interleaved_buffer.ptr, @intCast(i64, frames_to_read));
        const frames_read = @intCast(usize, c_frames_read);

        // Organize samples into separated channel buffers
        const base_write_offset = result_offset + total_read_count;

        for (0..frames_read) |frame_idx| {
            const write_idx = base_write_offset + frame_idx;

            for (0..self.n_channels) |channel_idx| {
                const read_idx = frame_idx * self.n_channels + channel_idx;
                result_pcm[channel_idx][write_idx] = self.interleaved_buffer[read_idx];
            }
        }

        if (frames_read == 0) break;
        total_read_count += frames_read;
    }

    return total_read_count;
}

pub fn seekToSample(self: *Self, sample: usize) !void {
    if (self.sf_file == null) {
        return error.FileNotOpen;
    }

    const sf_file = self.sf_file.?;

    const c_sample_index = @intCast(i64, sample);
    const c_seek_result = sndfile.sf_seek(sf_file, c_sample_index, sndfile.SEEK_SET);

    if (c_seek_result == -1) {
        return error.SeekFailed;
    }
}

pub fn close(self: *Self) void {
    if (self.sf_file) |sf_file| {
        _ = sndfile.sf_close(sf_file);
        self.sf_file = null;
    }

    self.allocator.free(self.interleaved_buffer);
}

pub fn durationSeconds(self: Self) f32 {
    return @intToFloat(f32, self.length) / @intToFloat(f32, self.sample_rate);
}
