// ============================================================================
// tlmm_engine.sv — Table Lookup Matrix Multiply Engine
// 
// Parameterized compute unit for ternary LLM inference.
// Computes one output element of y = W @ x using LUT lookups.
//
// Parameters:
//   G           — Group size (3 for TeLLMe baseline, 4 for our design)
//   K           — Input dimension (number of ternary weights per row)
//   VALUE_WIDTH — Bit width of partial sums / LUT entries (default 16)
//   ACC_WIDTH   — Accumulator bit width (default 32)
//
// Operation:
//   1. activation_lut_gen builds a 3^G-entry LUT from G activation values
//   2. For each group of G weights, the LUT index selects a partial sum
//   3. Accumulator sums all partial sums to produce the output
//
// Timing: NUM_GROUPS cycles per output element (pipelined BRAM read + add)
//   NUM_GROUPS = ceil(K / G)
//
// Author: Rajat Gupta
// Project: bitnet2lut — https://github.com/rajat116/bitnet2lut
// ============================================================================

module tlmm_engine #(
    parameter int G           = 4,
    parameter int K           = 2560,
    parameter int VALUE_WIDTH = 16,
    parameter int ACC_WIDTH   = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,

    // Control
    input  logic                    start,       // Pulse to begin computation
    output logic                    done,        // Pulse when result is ready
    output logic                    busy,        // High while computing

    // Activation LUT interface (3^G entries, written before start)
    input  logic                    lut_wen,     // Write enable
    input  logic [$clog2(3**G)-1:0] lut_waddr,  // Write address [0, 3^G-1]
    input  logic [VALUE_WIDTH-1:0]  lut_wdata,   // Partial sum value

    // Weight index interface (BRAM read port, one index per cycle)
    output logic [$clog2((K+G-1)/G)-1:0] widx_addr, // Which group to read
    input  logic [$clog2(3**G)-1:0]      widx_data,  // LUT index for that group

    // Output
    output logic [ACC_WIDTH-1:0]    result,
    output logic                    result_valid
);

    // ---------------------------------------------------------------
    // Derived parameters
    // ---------------------------------------------------------------
    localparam int NUM_GROUPS    = (K + G - 1) / G;
    localparam int NUM_LUT_ENTRIES = 3 ** G;
    localparam int LUT_ADDR_W   = $clog2(NUM_LUT_ENTRIES);
    localparam int GROUP_IDX_W  = $clog2(NUM_GROUPS);

    // ---------------------------------------------------------------
    // Activation LUT (dual-port: write from host, read for lookup)
    // Inferred as distributed RAM or BRAM depending on size
    // ---------------------------------------------------------------
    logic [VALUE_WIDTH-1:0] act_lut [0:NUM_LUT_ENTRIES-1];

    always_ff @(posedge clk) begin
        if (lut_wen) begin
            act_lut[lut_waddr] <= lut_wdata;
        end
    end

    // ---------------------------------------------------------------
    // FSM states
    // ---------------------------------------------------------------
    typedef enum logic [2:0] {
        IDLE,
        FETCH,      // Issue BRAM read address for weight index
        LOOKUP,     // Read activation LUT using weight index
        ACCUMULATE, // Add partial sum to accumulator
        FINISH      // Output result
    } state_t;

    state_t state, next_state;

    // ---------------------------------------------------------------
    // Counters and registers
    // ---------------------------------------------------------------
    logic [GROUP_IDX_W-1:0] group_cnt;
    logic [ACC_WIDTH-1:0]   accumulator;
    logic [LUT_ADDR_W-1:0]  lut_raddr;
    logic [VALUE_WIDTH-1:0] partial_sum;

    // Pipeline: we can overlap FETCH and ACCUMULATE for throughput
    // For now, simple 3-stage pipeline per group:
    //   Cycle 0: FETCH — issue widx_addr
    //   Cycle 1: LOOKUP — widx_data arrives, read act_lut
    //   Cycle 2: ACCUMULATE — add partial_sum to accumulator

    // ---------------------------------------------------------------
    // Pipelined datapath
    // ---------------------------------------------------------------
    // Stage tracking
    logic pipe_valid_s1, pipe_valid_s2;
    logic pipe_last_s1, pipe_last_s2;
    logic [LUT_ADDR_W-1:0] widx_data_s1;
    logic [VALUE_WIDTH-1:0] partial_sum_s2;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= IDLE;
            group_cnt   <= '0;
            accumulator <= '0;
            done        <= 1'b0;
            busy        <= 1'b0;
            result      <= '0;
            result_valid <= 1'b0;
            pipe_valid_s1 <= 1'b0;
            pipe_valid_s2 <= 1'b0;
            pipe_last_s1  <= 1'b0;
            pipe_last_s2  <= 1'b0;
            widx_data_s1  <= '0;
            partial_sum_s2 <= '0;
        end else begin
            // Defaults
            done         <= 1'b0;
            result_valid <= 1'b0;

            case (state)
                IDLE: begin
                    if (start) begin
                        state       <= FETCH;
                        group_cnt   <= '0;
                        accumulator <= '0;
                        busy        <= 1'b1;
                        pipe_valid_s1 <= 1'b0;
                        pipe_valid_s2 <= 1'b0;
                    end
                end

                FETCH: begin
                    // ---- Stage 0: Issue read address ----
                    // widx_addr is driven combinationally below

                    // ---- Stage 1: Capture weight index, read LUT ----
                    pipe_valid_s1 <= 1'b1;
                    pipe_last_s1  <= (group_cnt == NUM_GROUPS - 1);
                    widx_data_s1  <= widx_data;

                    // ---- Stage 2: Capture partial sum, accumulate ----
                    pipe_valid_s2  <= pipe_valid_s1;
                    pipe_last_s2   <= pipe_last_s1;
                    partial_sum_s2 <= act_lut[widx_data_s1];

                    // Accumulate from stage 2
                    if (pipe_valid_s2) begin
                        accumulator <= accumulator + 
                                       {{(ACC_WIDTH-VALUE_WIDTH){partial_sum_s2[VALUE_WIDTH-1]}}, 
                                        partial_sum_s2};
                    end

                    // Check if pipeline is draining
                    if (pipe_valid_s2 && pipe_last_s2) begin
                        state <= FINISH;
                    end

                    // Advance group counter
                    if (group_cnt < NUM_GROUPS - 1) begin
                        group_cnt <= group_cnt + 1'b1;
                    end else begin
                        // Stop issuing new reads, but pipeline still draining
                        // We stay in FETCH until pipe_last_s2
                    end
                end

                FINISH: begin
                    result       <= accumulator;
                    result_valid <= 1'b1;
                    done         <= 1'b1;
                    busy         <= 1'b0;
                    state        <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

    // ---------------------------------------------------------------
    // Combinational outputs
    // ---------------------------------------------------------------
    assign widx_addr = group_cnt;

endmodule
