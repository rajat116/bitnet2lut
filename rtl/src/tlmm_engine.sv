`timescale 1ns / 1ps
// ============================================================================
// tlmm_engine.sv — Table Lookup Matrix Multiply Engine
// Vivado 2020.2 compatible
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

    reg [VALUE_WIDTH-1:0] act_lut [0:NUM_LUT_ENTRIES-1];

    always @(posedge clk) begin
        if (lut_wen)
            act_lut[lut_waddr] <= lut_wdata;
    end

    localparam IDLE   = 2'd0;
    localparam FETCH  = 2'd1;
    localparam FINISH = 2'd2;

    reg [1:0] state;
    reg [GROUP_IDX_W-1:0]   group_cnt;
    reg [ACC_WIDTH-1:0]     accumulator;
    reg                     pipe_valid_s1, pipe_valid_s2;
    reg                     pipe_last_s1,  pipe_last_s2;
    reg [LUT_ADDR_W-1:0]   widx_data_s1;
    reg [VALUE_WIDTH-1:0]   partial_sum_s2;
    reg                     issuing_done;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= IDLE;
            group_cnt      <= 0;
            accumulator    <= 0;
            done           <= 1'b0;
            busy           <= 1'b0;
            result         <= 0;
            result_valid   <= 1'b0;
            pipe_valid_s1  <= 1'b0;
            pipe_valid_s2  <= 1'b0;
            pipe_last_s1   <= 1'b0;
            pipe_last_s2   <= 1'b0;
            widx_data_s1   <= 0;
            partial_sum_s2 <= 0;
            issuing_done   <= 1'b0;
        end else begin
            done         <= 1'b0;
            result_valid <= 1'b0;

            case (state)
                IDLE: begin
                    if (start) begin
                        state         <= FETCH;
                        group_cnt     <= 0;
                        accumulator   <= 0;
                        busy          <= 1'b1;
                        pipe_valid_s1 <= 1'b0;
                        pipe_valid_s2 <= 1'b0;
                        issuing_done  <= 1'b0;
                    end
                end

                FETCH: begin
                    pipe_valid_s1 <= ~issuing_done;
                    pipe_last_s1  <= (group_cnt == NUM_GROUPS - 1);
                    widx_data_s1  <= widx_data;

                    pipe_valid_s2  <= pipe_valid_s1;
                    pipe_last_s2   <= pipe_last_s1;
                    partial_sum_s2 <= act_lut[widx_data_s1];

                    if (pipe_valid_s2)
                        accumulator <= accumulator + {{(ACC_WIDTH-VALUE_WIDTH){partial_sum_s2[VALUE_WIDTH-1]}}, partial_sum_s2};

                    if (pipe_valid_s2 && pipe_last_s2)
                        state <= FINISH;

                    if (!issuing_done) begin
                        if (group_cnt == NUM_GROUPS - 1)
                            issuing_done <= 1'b1;
                        else
                            group_cnt <= group_cnt + 1;
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

    assign widx_addr = group_cnt;

endmodule
