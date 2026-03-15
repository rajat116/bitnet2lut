`timescale 1ns / 1ps
// ============================================================================
// tlmm_engine.sv — Table Lookup Matrix Multiply Engine
// Vivado 2020.2 compatible
//
// Fixed: PRIME state handles 1-cycle BRAM read latency before pipeline starts.
//
// Author: Rajat Gupta
// Project: bitnet2lut — https://github.com/rajat116/bitnet2lut
// ============================================================================

module tlmm_engine #(
    parameter G               = 4,
    parameter K               = 2560,
    parameter VALUE_WIDTH     = 16,
    parameter ACC_WIDTH       = 32,
    parameter NUM_LUT_ENTRIES = 81,
    parameter LUT_ADDR_W     = 7,
    parameter NUM_GROUPS      = 640,
    parameter GROUP_IDX_W    = 10
) (
    input  wire                      clk,
    input  wire                      rst_n,

    input  wire                      start,
    output reg                       done,
    output reg                       busy,

    input  wire                      lut_wen,
    input  wire [LUT_ADDR_W-1:0]    lut_waddr,
    input  wire [VALUE_WIDTH-1:0]    lut_wdata,

    output wire [GROUP_IDX_W-1:0]    widx_addr,
    input  wire [LUT_ADDR_W-1:0]    widx_data,

    output reg  [ACC_WIDTH-1:0]      result,
    output reg                       result_valid
);

    // Activation LUT storage
    reg [VALUE_WIDTH-1:0] act_lut [0:NUM_LUT_ENTRIES-1];

    always @(posedge clk) begin
        if (lut_wen)
            act_lut[lut_waddr] <= lut_wdata;
    end

    // FSM states
    localparam IDLE   = 3'd0;
    localparam PRIME  = 3'd1;  // Wait 1 cycle for first BRAM read to appear
    localparam FETCH  = 3'd2;  // Steady-state pipeline
    localparam DRAIN1 = 3'd3;  // Drain pipeline stage 1
    localparam DRAIN2 = 3'd4;  // Drain pipeline stage 2
    localparam FINISH = 3'd5;

    reg [2:0] state;
    reg [GROUP_IDX_W-1:0]   group_cnt;
    reg [ACC_WIDTH-1:0]     accumulator;
    reg [LUT_ADDR_W-1:0]   widx_data_r;     // Registered BRAM output
    reg [VALUE_WIDTH-1:0]   partial_sum_r;   // LUT read result
    reg                     partial_valid;   // partial_sum_r is valid

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= IDLE;
            group_cnt     <= 0;
            accumulator   <= 0;
            done          <= 1'b0;
            busy          <= 1'b0;
            result        <= 0;
            result_valid  <= 1'b0;
            widx_data_r   <= 0;
            partial_sum_r <= 0;
            partial_valid <= 1'b0;
        end else begin
            done         <= 1'b0;
            result_valid <= 1'b0;

            case (state)
                // --------------------------------------------------------
                IDLE: begin
                    if (start) begin
                        state       <= PRIME;
                        group_cnt   <= 1;       // group 0 address is already on widx_addr
                        accumulator <= 0;
                        busy        <= 1'b1;
                        partial_valid <= 1'b0;
                    end
                end

                // --------------------------------------------------------
                // PRIME: group_cnt was 0 in IDLE (via widx_addr = group_cnt).
                // BRAM has 1-cycle latency, so widx_data for group 0 appears
                // at the END of this cycle. We advance group_cnt to 1.
                // --------------------------------------------------------
                PRIME: begin
                    // widx_data now holds the index for group 0
                    // Register it and issue group 1 address
                    widx_data_r <= widx_data;
                    group_cnt   <= 2;
                    state       <= FETCH;
                end

                // --------------------------------------------------------
                // FETCH: Steady-state 2-stage pipeline
                //   This cycle: 
                //     - widx_data has result for group (group_cnt - 2)... 
                //       Actually let's be precise:
                //
                //   Pipeline timing:
                //     widx_addr = group_cnt (combinational)
                //     widx_data appears 1 cycle later (BRAM latency)
                //     widx_data_r captures it 1 more cycle later
                //     partial_sum_r = act_lut[widx_data_r] (same cycle as capture)
                //     accumulate partial_sum_r (if valid)
                //
                //   So each cycle in FETCH:
                //     1. Accumulate partial_sum_r from previous iteration
                //     2. Read LUT using widx_data_r from previous iteration  
                //     3. Capture widx_data into widx_data_r
                //     4. Advance group_cnt
                // --------------------------------------------------------
                FETCH: begin
                    // Accumulate previous partial sum
                    if (partial_valid)
                        accumulator <= accumulator + {{(ACC_WIDTH-VALUE_WIDTH){partial_sum_r[VALUE_WIDTH-1]}}, partial_sum_r};

                    // Read LUT for the captured BRAM output
                    partial_sum_r <= act_lut[widx_data_r];
                    partial_valid <= 1'b1;

                    // Capture current BRAM output
                    widx_data_r <= widx_data;

                    // Advance or transition to drain
                    if (group_cnt == NUM_GROUPS) begin
                        // All addresses have been issued; start draining
                        state <= DRAIN1;
                    end else begin
                        group_cnt <= group_cnt + 1;
                    end
                end

                // --------------------------------------------------------
                // DRAIN1: One more LUT read for the last captured BRAM data
                // --------------------------------------------------------
                DRAIN1: begin
                    // Accumulate
                    if (partial_valid)
                        accumulator <= accumulator + {{(ACC_WIDTH-VALUE_WIDTH){partial_sum_r[VALUE_WIDTH-1]}}, partial_sum_r};

                    // Last LUT read
                    partial_sum_r <= act_lut[widx_data_r];
                    partial_valid <= 1'b1;

                    state <= DRAIN2;
                end

                // --------------------------------------------------------
                // DRAIN2: Accumulate the last partial sum
                // --------------------------------------------------------
                DRAIN2: begin
                    accumulator <= accumulator + {{(ACC_WIDTH-VALUE_WIDTH){partial_sum_r[VALUE_WIDTH-1]}}, partial_sum_r};
                    state <= FINISH;
                end

                // --------------------------------------------------------
                FINISH: begin
                    result       <= accumulator;
                    result_valid <= 1'b1;
                    done         <= 1'b1;
                    busy         <= 1'b0;
                    state        <= IDLE;
                    group_cnt    <= 0;  // Reset for next start
                end

                default: state <= IDLE;
            endcase
        end
    end

    assign widx_addr = group_cnt;

endmodule
