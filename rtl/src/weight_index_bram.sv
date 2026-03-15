// ============================================================================
// weight_index_bram.sv — Weight Index Storage
//
// Stores LUT indices for one weight row (one output neuron).
// Each entry is a base-3 encoded index for a group of G ternary weights.
//
// For G=4: index range [0, 80], needs 7 bits
// For G=3: index range [0, 26], needs 5 bits
//
// Inferred as BRAM by Vivado when depth >= 64.
//
// Author: Rajat Gupta
// ============================================================================

module weight_index_bram #(
    parameter int G          = 4,
    parameter int K          = 2560,
    parameter int INIT_FILE  = 0    // If nonzero, use $readmemh
) (
    input  logic                              clk,

    // Write port (for loading weights)
    input  logic                              wen,
    input  logic [$clog2((K+G-1)/G)-1:0]     waddr,
    input  logic [$clog2(3**G)-1:0]           wdata,

    // Read port (for inference)
    input  logic [$clog2((K+G-1)/G)-1:0]     raddr,
    output logic [$clog2(3**G)-1:0]           rdata
);

    localparam int NUM_GROUPS = (K + G - 1) / G;
    localparam int IDX_WIDTH  = $clog2(3**G);

    // Storage
    logic [IDX_WIDTH-1:0] mem [0:NUM_GROUPS-1];

    // Synchronous read (BRAM inference)
    always_ff @(posedge clk) begin
        rdata <= mem[raddr];
    end

    // Synchronous write
    always_ff @(posedge clk) begin
        if (wen) begin
            mem[waddr] <= wdata;
        end
    end

endmodule
