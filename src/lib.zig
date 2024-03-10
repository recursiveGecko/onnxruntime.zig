const std = @import("std");

pub usingnamespace @import("./onnxruntime.zig");

comptime {
    std.testing.refAllDecls(@This());
}
