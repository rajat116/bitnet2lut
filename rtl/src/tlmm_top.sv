// ============================================================================
// tlmm_top.sv — Top-Level TLMM Wrapper
//
// Connects the TLMM engine with its weight index BRAM.
// This is the unit we synthesize for resource/timing comparison.
//
// Interface:
//   - Load phase: write activation LUT entries + weight indices
//   - Compute phase: start → engine reads weight indices, looks up LUT,
//     accumulates, outputs result
//
// For synthesis comparison, we instantiate this with:
//   G=3 (TeLLMe baseline): 27-entry LUT, 5-bit indices
//   G=4 (our design):      81-entry LUT, 7-bit indices
//
// Author: Rajat Gupta
// ============================================================================

module tlmm_top #(
    parameter int G           = 4,
    parameter int K           = 2560,
    parameter int VALUE_WIDTH = 16,
    parameter int ACC_WIDTH   = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,

    // === Load activation LUT ===
    input  logic                    act_lut_wen,
    input  logic [$clog2(3**G)-1:0] act_lut_waddr,
    input  logic [VALUE_WIDTH-1:0]  act_lut_wdata,

    // === Load weight indices ===
    input  logic                    widx_wen,
    input  logic [$clog2((K+G-1)/G)-1:0] widx_waddr,
    input  logic [$clog2(3**G)-1:0]      widx_wdata,

    // === Control ===
    input  logic                    start,
    output logic                    done,
    output logic                    busy,

    // === Output ===
    output logic [ACC_WIDTH-1:0]    result,
    output logic                    result_valid
);

    localparam int NUM_GROUPS = (K + G - 1) / G;

    // Internal wires: engine <-> weight BRAM
    logic [$clog2(NUM_GROUPS)-1:0]  widx_raddr;
    logic [$clog2(3**G)-1:0]        widx_rdata;

    // Weight index BRAM
    weight_index_bram #(
        .G(G),
        .K(K)
    ) u_widx_bram (
        .clk   (clk),
        .wen   (widx_wen),
        .waddr (widx_waddr),
        .wdata (widx_wdata),
        .raddr (widx_raddr),
        .rdata (widx_rdata)
    );

    // TLMM engine
    tlmm_engine #(
        .G          (G),
        .K          (K),
        .VALUE_WIDTH(VALUE_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH)
    ) u_engine (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (start),
        .done        (done),
        .busy        (busy),
        .lut_wen     (act_lut_wen),
        .lut_waddr   (act_lut_waddr),
        .lut_wdata   (act_lut_wdata),
        .widx_addr   (widx_raddr),
        .widx_data   (widx_rdata),
        .result      (result),
        .result_valid(result_valid)
    );

endmodule
