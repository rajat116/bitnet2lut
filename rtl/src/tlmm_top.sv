`timescale 1ns / 1ps
// ============================================================================
// tlmm_top.sv — Top-Level TLMM Wrapper
// Vivado 2020.2 compatible
//
// Author: Rajat Gupta
// ============================================================================

module tlmm_top #(
    parameter G           = 4,
    parameter K           = 2560,
    parameter ACT_WIDTH   = 8,
    parameter VALUE_WIDTH = (ACT_WIDTH == 4) ? 8 : 16,
    parameter ACC_WIDTH   = 32
) (
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    act_lut_wen,
    input  wire [6:0]              act_lut_waddr,
    input  wire [VALUE_WIDTH-1:0]  act_lut_wdata,

    input  wire                    widx_wen,
    input  wire [12:0]             widx_waddr,
    input  wire [6:0]              widx_wdata,

    input  wire                    start,
    output wire                    done,
    output wire                    busy,

    output wire [ACC_WIDTH-1:0]    result,
    output wire                    result_valid
);

    localparam NUM_LUT_ENTRIES = (G == 3) ? 27  :
                                 (G == 4) ? 81  :
                                 (G == 5) ? 243 : 9;

    localparam LUT_ADDR_W = (G == 3) ? 5 :
                            (G == 4) ? 7 :
                            (G == 5) ? 8 : 4;

    localparam NUM_GROUPS = (K + G - 1) / G;

    localparam GROUP_IDX_W = (NUM_GROUPS <= 256)   ? 8  :
                             (NUM_GROUPS <= 512)   ? 9  :
                             (NUM_GROUPS <= 1024)  ? 10 :
                             (NUM_GROUPS <= 2048)  ? 11 :
                             (NUM_GROUPS <= 4096)  ? 12 : 13;

    wire [GROUP_IDX_W-1:0]  widx_raddr;
    wire [LUT_ADDR_W-1:0]   widx_rdata;

    weight_index_bram #(
        .NUM_GROUPS  (NUM_GROUPS),
        .GROUP_IDX_W (GROUP_IDX_W),
        .LUT_ADDR_W  (LUT_ADDR_W)
    ) u_widx_bram (
        .clk   (clk),
        .wen   (widx_wen),
        .waddr (widx_waddr[GROUP_IDX_W-1:0]),
        .wdata (widx_wdata[LUT_ADDR_W-1:0]),
        .raddr (widx_raddr),
        .rdata (widx_rdata)
    );

    tlmm_engine #(
        .G              (G),
        .K              (K),
        .VALUE_WIDTH    (VALUE_WIDTH),
        .ACC_WIDTH      (ACC_WIDTH),
        .NUM_LUT_ENTRIES(NUM_LUT_ENTRIES),
        .LUT_ADDR_W     (LUT_ADDR_W),
        .NUM_GROUPS      (NUM_GROUPS),
        .GROUP_IDX_W    (GROUP_IDX_W)
    ) u_engine (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (start),
        .done        (done),
        .busy        (busy),
        .lut_wen     (act_lut_wen),
        .lut_waddr   (act_lut_waddr[LUT_ADDR_W-1:0]),
        .lut_wdata   (act_lut_wdata),
        .widx_addr   (widx_raddr),
        .widx_data   (widx_rdata),
        .result      (result),
        .result_valid(result_valid)
    );

endmodule
