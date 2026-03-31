`timescale 1ns / 1ps
// ============================================================================
// exception_comparator.sv
//
// Detects outlier activations exceeding a calibrated threshold.
// Threshold is hardwired at synthesis time from offline p99 calibration
// of BitNet 2B4T down_proj activation distributions.
//
// Hardware cost: one comparator per activation value.
// No runtime overhead — threshold is a constant.
//
// Author: Rajat Gupta
// Project: bitnet2lut
// ============================================================================

module exception_comparator #(
    parameter ACT_WIDTH  = 16,   // input activation bit width
    parameter FRAC_BITS  = 8,    // fractional bits in fixed-point
    // Threshold = p99 of down_proj activations (layer-specific)
    // Hardwired from offline calibration. Example: layer 6 p99 = 1.9008
    // In fixed-point Q8.8: 1.9008 * 256 = 486
    parameter [ACT_WIDTH-1:0] THRESHOLD = 486
) (
    input  wire signed [ACT_WIDTH-1:0] act_in,
    output wire                         is_exception
);
    // Compare absolute value against threshold
    wire signed [ACT_WIDTH-1:0] abs_val;
    assign abs_val = (act_in[ACT_WIDTH-1]) ? -act_in : act_in;
    assign is_exception = (abs_val > $signed({1'b0, THRESHOLD[ACT_WIDTH-2:0]}));

endmodule
