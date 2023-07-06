const std = @import("std");
const pi = std.math.pi;

fn pow(x: f32, y: f32) f32 {
    return std.math.pow(f32, x, y);
}

pub fn windowNormFactor(window: []const f32) f32 {
    var sum: f32 = 0;

    for (window) |x| {
        sum += x;
    }

    return @as(f32, @floatFromInt(window.len)) / sum;
}

pub fn squareWindow(result: []f32) void {
    @memset(result, 1);
}

pub fn hannWindowPeriodic(result: []f32) void {
    const K = 1;
    const a0 = 0.5;
    const a1 = 1 - a0;

    cosineSumWindowPeriodic(result, K, .{ a0, a1 });
}

pub fn hannWindowSymmetric(result: []f32) void {
    const a0 = 0.5;
    const a1 = 0.5;

    const N: f32 = @floatFromInt(result.len);
    var step = 2 * pi / (N - 1);

    for (0..result.len) |n_idx| {
        const n: f32 = @floatFromInt(n_idx);
        result[n_idx] = a0 - a1 * @cos(n * step);
    }
}

pub fn hammingWindowPeriodic(result: []f32) void {
    const K = 1;
    const a0 = 0.53836;
    const a1 = 1 - a0;

    cosineSumWindowPeriodic(result, K, .{ a0, a1 });
}

pub fn cosineSumWindowPeriodic(
    result: []f32,
    comptime K: usize,
    comptime alphas: [K + 1]f32,
) void {
    const N: f32 = @floatFromInt(result.len);

    for (0..result.len) |n_idx| {
        const n: f32 = @floatFromInt(n_idx);
        result[n_idx] = 0;

        for (0..K + 1) |k_idx| {
            const k: f32 = @floatFromInt(k_idx);

            result[n_idx] += pow(-1, k) * alphas[k_idx] * @cos((2 * pi * k * n) / N);
        }
    }
}
