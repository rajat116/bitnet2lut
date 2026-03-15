// ============================================================================
// tlmm_top.sv — Top-Level TLMM Wrapper
// Vivado 2020.2 compatible
//
// All derived parameters (NUM_LUT_ENTRIES, LUT_ADDR_W, NUM_GROUPS, 
// GROUP_IDX_W) are computed as localparams inside the module and 
// passed down explicitly to submodules. No $clog2() in port declarations.
//
// Author: Rajat Gupta
// ============================================================================

module tlmm_top #(
    parameter G           = 4,
    parameter K           = 2560,
    parameter VALUE_WIDTH = 16,
    parameter ACC_WIDTH   = 32
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // === Load activation LUT ===
    input  wire                    act_lut_wen,
    input  wire [6:0]              act_lut_waddr,  // 7 bits covers up to 128 entries (G<=4: max 81)
    input  wire [VALUE_WIDTH-1:0]  act_lut_wdata,

    // === Load weight indices ===
    input  wire                    widx_wen,
    input  wire [12:0]             widx_waddr,     // 13 bits covers up to 8192 groups
    input  wire [6:0]              widx_wdata,

    // === Control ===
    input  wire                    start,
    output wire                    done,
    output wire                    busy,

    // === Output ===
    output wire [ACC_WIDTH-1:0]    result,
    output wire                    result_valid
);

    // ---------------------------------------------------------------
    // Derived parameters — computed once, passed to submodules
    // ---------------------------------------------------------------
    // For G=3: NUM_LUT_ENTRIES=27, LUT_ADDR_W=5, NUM_GROUPS=854, GROUP_IDX_W=10
    // For G=4: NUM_LUT_ENTRIES=81, LUT_ADDR_W=7, NUM_GROUPS=640, GROUP_IDX_W=10

    localparam NUM_LUT_ENTRIES = (G == 3) ? 27  :
                                 (G == 4) ? 81  :
                                 (G == 5) ? 243 : 9;  // G=2 fallback

    localparam LUT_ADDR_W = (G == 3) ? 5 :
                            (G == 4) ? 7 :
                            (G == 5) ? 8 : 4;

    localparam NUM_GROUPS = (K + G - 1) / G;

    // GROUP_IDX_W = ceil(log2(NUM_GROUPS))
    // For K=2560: G=3->854 needs 10 bits, G=4->640 needs 10 bits
    // For K=6912: G=3->2304 needs 12 bits, G=4->1728 needs 11 bits
    localparam GROUP_IDX_W = (NUM_GROUPS <= 256)   ? 8  :
                             (NUM_GROUPS <= 512)   ? 9  :
                             (NUM_GROUPS <= 1024)  ? 10 :
                             (NUM_GROUPS <= 2048)  ? 11 :
                             (NUM_GROUPS <= 4096)  ? 12 : 13;

    // Internal wires
    wire [GROUP_IDX_W-1:0]  widx_raddr;
    wire [LUT_ADDR_W-1:0]   widx_rdata;

    // Weight index BRAM
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

    // TLMM engine
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
