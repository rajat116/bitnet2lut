`timescale 1ns / 1ps
// ============================================================================
// weight_index_bram.sv — Weight Index Storage
// Vivado 2020.2 compatible
//
// Author: Rajat Gupta
// ============================================================================

module weight_index_bram #(
    parameter NUM_GROUPS  = 640,
    parameter GROUP_IDX_W = 10,
    parameter LUT_ADDR_W  = 7
) (
    input  wire                      clk,

    input  wire                      wen,
    input  wire [GROUP_IDX_W-1:0]   waddr,
    input  wire [LUT_ADDR_W-1:0]    wdata,

    input  wire [GROUP_IDX_W-1:0]   raddr,
    output reg  [LUT_ADDR_W-1:0]    rdata
);

    reg [LUT_ADDR_W-1:0] mem [0:NUM_GROUPS-1];

    always @(posedge clk) begin
        rdata <= mem[raddr];
    end

    always @(posedge clk) begin
        if (wen)
            mem[waddr] <= wdata;
    end

endmodule
