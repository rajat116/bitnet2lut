`timescale 1ns / 1ps
// ============================================================================
// tb_tlmm_engine.sv — Testbench for TLMM Engine
// Vivado 2020.2 compatible
//
// Author: Rajat Gupta
// ============================================================================

`ifndef TEST_G
`define TEST_G 4
`endif

`ifndef TEST_K
`define TEST_K 2560
`endif

module tb_tlmm_engine;

    localparam G           = `TEST_G;
    localparam K           = `TEST_K;
    localparam VALUE_WIDTH = 16;
    localparam ACC_WIDTH   = 32;

    localparam NUM_LUT_ENTRIES = (G == 3) ? 27 : (G == 4) ? 81 : (G == 5) ? 243 : 9;
    localparam LUT_ADDR_W = (G == 3) ? 5 : (G == 4) ? 7 : (G == 5) ? 8 : 4;
    localparam NUM_GROUPS = (K + G - 1) / G;
    localparam GROUP_IDX_W = (NUM_GROUPS <= 256) ? 8 :
                             (NUM_GROUPS <= 512) ? 9 :
                             (NUM_GROUPS <= 1024) ? 10 :
                             (NUM_GROUPS <= 2048) ? 11 :
                             (NUM_GROUPS <= 4096) ? 12 : 13;

    reg clk;
    reg rst_n;
    initial clk = 0;
    always #2.5 clk = ~clk;

    reg                    act_lut_wen;
    reg  [6:0]             act_lut_waddr;
    reg  [VALUE_WIDTH-1:0] act_lut_wdata;
    reg                    widx_wen;
    reg  [12:0]            widx_waddr;
    reg  [6:0]             widx_wdata;
    reg                    start;
    wire                   done;
    wire                   busy;
    wire [ACC_WIDTH-1:0]   result;
    wire                   result_valid;

    tlmm_top #(
        .G          (G),
        .K          (K),
        .VALUE_WIDTH(VALUE_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .act_lut_wen    (act_lut_wen),
        .act_lut_waddr  (act_lut_waddr),
        .act_lut_wdata  (act_lut_wdata),
        .widx_wen       (widx_wen),
        .widx_waddr     (widx_waddr),
        .widx_wdata     (widx_wdata),
        .start          (start),
        .done           (done),
        .busy           (busy),
        .result         (result),
        .result_valid   (result_valid)
    );

    // Test vector storage
    reg signed [VALUE_WIDTH-1:0] tv_act_lut [0:NUM_LUT_ENTRIES-1];
    reg [LUT_ADDR_W-1:0]         tv_widx    [0:NUM_GROUPS-1];

    // Working variables
    integer cycle_count;
    integer num_passed;
    integer num_failed;
    integer i, j, t;
    reg [31:0] lfsr;
    integer temp, trit;
    integer signed w_val;
    integer signed partial;
    integer signed act_vals [0:3];  // max G=4
    reg signed [ACC_WIDTH-1:0] expected;
    integer idx;

    // Simple LFSR step function — called explicitly each time
    // Uses xorshift32 for better distribution than LCG
    task lfsr_step;
        begin
            lfsr = lfsr ^ (lfsr << 13);
            lfsr = lfsr ^ (lfsr >> 17);
            lfsr = lfsr ^ (lfsr << 5);
        end
    endtask

    initial begin
        act_lut_wen   = 0;
        act_lut_waddr = 0;
        act_lut_wdata = 0;
        widx_wen      = 0;
        widx_waddr    = 0;
        widx_wdata    = 0;
        start         = 0;
        num_passed    = 0;
        num_failed    = 0;

        rst_n = 0;
        repeat (10) @(posedge clk);
        rst_n = 1;
        repeat (5) @(posedge clk);

        $display("============================================================");
        $display("TLMM Engine Testbench: G=%0d, K=%0d, NUM_GROUPS=%0d", G, K, NUM_GROUPS);
        $display("LUT entries: %0d, Index width: %0d bits", NUM_LUT_ENTRIES, LUT_ADDR_W);
        $display("============================================================");

        for (t = 0; t < 5; t = t + 1) begin
            $display("\n--- Test %0d ---", t);

            // Seed LFSR (different per test, avoid 0)
            lfsr = 32'd100001 + t * 32'd77773;

            // Generate G activation values (INT8 range)
            for (j = 0; j < G; j = j + 1) begin
                lfsr_step;
                act_vals[j] = (lfsr[7:0] > 127) ? (lfsr[7:0] - 256) : lfsr[7:0];
            end
            // Zero unused slots
            for (j = G; j < 4; j = j + 1) begin
                act_vals[j] = 0;
            end

            // Build LUT: for each entry compute partial dot product with activations
            for (i = 0; i < NUM_LUT_ENTRIES; i = i + 1) begin
                partial = 0;
                temp = i;
                for (j = 0; j < G; j = j + 1) begin
                    trit = temp % 3;
                    temp = temp / 3;
                    w_val = trit - 1;  // {-1, 0, +1}
                    partial = partial + w_val * act_vals[j];
                end
                tv_act_lut[i] = partial[VALUE_WIDTH-1:0];
            end

            // Generate weight indices and compute expected result
            expected = 0;
            for (i = 0; i < NUM_GROUPS; i = i + 1) begin
                lfsr_step;
                idx = lfsr[30:0] % NUM_LUT_ENTRIES;  // use 31 bits, positive
                tv_widx[i] = idx[LUT_ADDR_W-1:0];
                // Sign-extend partial sum and accumulate
                expected = expected + {{(ACC_WIDTH-VALUE_WIDTH){tv_act_lut[idx][VALUE_WIDTH-1]}}, tv_act_lut[idx]};
            end

            // Load activation LUT into DUT
            for (i = 0; i < NUM_LUT_ENTRIES; i = i + 1) begin
                @(posedge clk);
                act_lut_wen   <= 1'b1;
                act_lut_waddr <= i[6:0];
                act_lut_wdata <= tv_act_lut[i];
            end
            @(posedge clk);
            act_lut_wen <= 1'b0;

            // Load weight indices into DUT
            for (i = 0; i < NUM_GROUPS; i = i + 1) begin
                @(posedge clk);
                widx_wen   <= 1'b1;
                widx_waddr <= i[12:0];
                widx_wdata <= tv_widx[i][6:0];
            end
            @(posedge clk);
            widx_wen <= 1'b0;

            repeat (2) @(posedge clk);

            // Start computation
            @(posedge clk);
            start <= 1'b1;
            cycle_count = 0;
            @(posedge clk);
            start <= 1'b0;

            // Wait for done
            while (!done) begin
                @(posedge clk);
                cycle_count = cycle_count + 1;
            end

            // Check result
            if ($signed(result) == expected) begin
                $display("TEST %0d PASSED: result=%0d expected=%0d cycles=%0d",
                         t, $signed(result), expected, cycle_count);
                num_passed = num_passed + 1;
            end else begin
                $display("TEST %0d FAILED: result=%0d expected=%0d cycles=%0d",
                         t, $signed(result), expected, cycle_count);
                num_failed = num_failed + 1;
            end
        end

        $display("\n============================================================");
        $display("SUMMARY: G=%0d K=%0d", G, K);
        $display("  Passed: %0d / 5", num_passed);
        $display("  Failed: %0d / 5", num_failed);
        $display("  Groups per dot product: %0d", NUM_GROUPS);
        $display("  Cycles per dot product: %0d (last test)", cycle_count);
        $display("============================================================");

        if (num_failed > 0)
            $display("*** SOME TESTS FAILED ***");
        else
            $display("*** ALL TESTS PASSED ***");

        $finish;
    end

    initial begin
        #50000000;
        $display("TIMEOUT!");
        $finish;
    end

endmodule
