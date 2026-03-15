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

    // Write port
    input  wire                      wen,
    input  wire [GROUP_IDX_W-1:0]   waddr,
    input  wire [LUT_ADDR_W-1:0]    wdata,

    // Read port
    input  wire [GROUP_IDX_W-1:0]   raddr,
    output reg  [LUT_ADDR_W-1:0]    rdata
);

    reg [LUT_ADDR_W-1:0] mem [0:NUM_GROUPS-1];

    // Synchronous read (BRAM inference)
    always @(posedge clk) begin
        rdata <= mem[raddr];
    end

    // Synchronous write
    always @(posedge clk) begin
        if (wen)
            mem[waddr] <= wdata;
    end

endmodule
