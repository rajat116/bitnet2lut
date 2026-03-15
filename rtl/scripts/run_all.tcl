# ============================================================================
# run_all.tcl — Complete Vivado flow for G=3 vs G=4 comparison
#
# Usage from Vivado TCL console:
#   cd <path_to_rtl_directory>
#   source scripts/run_all.tcl
#
# Or from command line:
#   vivado -mode batch -source scripts/run_all.tcl
#
# This script:
#   1. Creates a project for G=3, synthesizes, extracts reports
#   2. Creates a project for G=4, synthesizes, extracts reports
#   3. Runs behavioral simulation for both
#   4. Prints a comparison summary
#
# Target device: xcku5p-ffvb676-2-e (Kintex UltraScale+)
#   Change PART below if you have a different device/license.
#
# Author: Rajat Gupta
# ============================================================================

# ---- Configuration ----
set PART "xcku5p-ffvb676-2-e"
# Alternative parts (uncomment one if needed):
# set PART "xc7k325tffg900-2"      ;# Kintex-7 (free WebPACK)
# set PART "xc7a200tsbg484-1"      ;# Artix-7 200T (free WebPACK)
# set PART "xczu7ev-ffvc1156-2-e"   ;# Zynq UltraScale+ (ZCU104)

set TOP_DIR [pwd]
set SRC_DIR "${TOP_DIR}/src"
set TB_DIR  "${TOP_DIR}/tb"
set XDC_DIR "${TOP_DIR}/constraints"
set OUT_DIR "${TOP_DIR}/results"

file mkdir $OUT_DIR

# K dimension for the test (use 2560 for real BitNet, or smaller for quick test)
set K_DIM 2560

# ============================================================================
# Procedure: Run synthesis + implementation for a given G
# ============================================================================
proc run_synthesis {G K PART SRC_DIR XDC_DIR OUT_DIR} {
    set proj_name "tlmm_G${G}"
    set proj_dir "${OUT_DIR}/${proj_name}"
    
    puts "============================================================"
    puts "Synthesizing TLMM engine: G=${G}, K=${K}"
    puts "Target device: ${PART}"
    puts "Project: ${proj_dir}"
    puts "============================================================"
    
    # Clean previous project
    if {[file exists $proj_dir]} {
        file delete -force $proj_dir
    }
    
    # Create project
    create_project $proj_name $proj_dir -part $PART -force
    set_property target_language Verilog [current_project]
    
    # Add source files
    add_files [glob ${SRC_DIR}/*.sv]
    
    # Add constraints
    add_files -fileset constrs_1 ${XDC_DIR}/timing.xdc
    
    # Set generics/parameters for top module
    set_property generic "G=${G} K=${K}" [current_fileset]
    set_property top tlmm_top [current_fileset]
    
    # ---- Synthesis ----
    puts "\n--- Running Synthesis for G=${G} ---"
    set synth_start [clock seconds]
    
    # Out-of-context synthesis (no I/O planning needed)
    synth_design -top tlmm_top -part $PART \
        -generic "G=${G} K=${K}" \
        -mode out_of_context
    
    set synth_end [clock seconds]
    set synth_time [expr {$synth_end - $synth_start}]
    puts "Synthesis completed in ${synth_time} seconds"
    
    # ---- Reports ----
    set rpt_dir "${OUT_DIR}/reports_G${G}"
    file mkdir $rpt_dir
    
    # Utilization report
    report_utilization -file "${rpt_dir}/utilization.rpt"
    report_utilization -hierarchical -file "${rpt_dir}/utilization_hier.rpt"
    
    # Timing summary
    report_timing_summary -file "${rpt_dir}/timing_summary.rpt"
    
    # Power estimate (post-synthesis)
    report_power -file "${rpt_dir}/power.rpt"
    
    # Design statistics
    report_design_analysis -file "${rpt_dir}/design_analysis.rpt"
    
    # ---- Optional: Implementation (Place & Route) ----
    puts "\n--- Running Implementation for G=${G} ---"
    set impl_start [clock seconds]
    
    opt_design
    place_design
    route_design
    
    set impl_end [clock seconds]
    set impl_time [expr {$impl_end - $impl_start}]
    puts "Implementation completed in ${impl_time} seconds"
    
    # Post-implementation reports
    report_utilization -file "${rpt_dir}/utilization_impl.rpt"
    report_timing_summary -file "${rpt_dir}/timing_summary_impl.rpt"
    report_power -file "${rpt_dir}/power_impl.rpt"
    
    # Extract key metrics
    puts "\n--- Key Metrics for G=${G} ---"
    
    # Get timing slack
    set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
    set achieved_period [expr {3.333 - $wns}]
    set achieved_fmax [expr {1000.0 / $achieved_period}]
    
    puts "WNS (Worst Negative Slack): ${wns} ns"
    puts "Achieved Fmax: ${achieved_fmax} MHz"
    puts "Synthesis time: ${synth_time} s"
    puts "Implementation time: ${impl_time} s"
    
    # Save summary
    set fp [open "${rpt_dir}/summary.txt" w]
    puts $fp "TLMM Engine Synthesis Summary"
    puts $fp "=============================="
    puts $fp "G = ${G}"
    puts $fp "K = ${K}"
    puts $fp "NUM_GROUPS = [expr {(${K} + ${G} - 1) / ${G}}]"
    puts $fp "LUT_ENTRIES = [expr {int(pow(3, ${G}))}]"
    puts $fp "Target Part = ${PART}"
    puts $fp "Clock Target = 300 MHz (3.333 ns)"
    puts $fp "WNS = ${wns} ns"
    puts $fp "Achieved Fmax = ${achieved_fmax} MHz"
    puts $fp "Synth Time = ${synth_time} s"
    puts $fp "Impl Time = ${impl_time} s"
    close $fp
    
    # Close project
    close_project
    
    return [list $wns $achieved_fmax $synth_time $impl_time]
}

# ============================================================================
# Procedure: Run behavioral simulation for a given G
# ============================================================================
proc run_simulation {G K PART SRC_DIR TB_DIR OUT_DIR} {
    set proj_name "tlmm_sim_G${G}"
    set proj_dir "${OUT_DIR}/${proj_name}"
    
    puts "============================================================"
    puts "Simulating TLMM engine: G=${G}, K=${K}"
    puts "============================================================"
    
    if {[file exists $proj_dir]} {
        file delete -force $proj_dir
    }
    
    create_project $proj_name $proj_dir -part $PART -force
    set_property target_language Verilog [current_project]
    
    # Add source files
    add_files [glob ${SRC_DIR}/*.sv]
    
    # Add testbench to simulation fileset
    add_files -fileset sim_1 ${TB_DIR}/tb_tlmm_engine.sv
    set_property top tb_tlmm_engine [get_filesets sim_1]
    
    # Set defines for G and K
    set_property verilog_define "TEST_G=${G} TEST_K=${K}" [get_filesets sim_1]
    
    # Run simulation
    set sim_dir "${OUT_DIR}/sim_G${G}"
    file mkdir $sim_dir
    
    launch_simulation -mode behavioral
    run all
    
    close_project
}

# ============================================================================
# Main flow
# ============================================================================
puts "\n"
puts "############################################################"
puts "# TLMM G=3 vs G=4 Comparison Flow"
puts "# Target: ${PART}"
puts "# K = ${K_DIM}"
puts "############################################################"
puts "\n"

# ---- Step 1: Simulate G=3 ----
puts "\n>>> SIMULATION G=3 <<<\n"
run_simulation 3 $K_DIM $PART $SRC_DIR $TB_DIR $OUT_DIR

# ---- Step 2: Simulate G=4 ----
puts "\n>>> SIMULATION G=4 <<<\n"
run_simulation 4 $K_DIM $PART $SRC_DIR $TB_DIR $OUT_DIR

# ---- Step 3: Synthesize G=3 ----
puts "\n>>> SYNTHESIS G=3 <<<\n"
set results_g3 [run_synthesis 3 $K_DIM $PART $SRC_DIR $XDC_DIR $OUT_DIR]

# ---- Step 4: Synthesize G=4 ----
puts "\n>>> SYNTHESIS G=4 <<<\n"
set results_g4 [run_synthesis 4 $K_DIM $PART $SRC_DIR $XDC_DIR $OUT_DIR]

# ---- Step 5: Print comparison ----
puts "\n"
puts "############################################################"
puts "# COMPARISON SUMMARY"
puts "############################################################"
puts ""
puts "G=3 (TeLLMe baseline):"
puts "  WNS:  [lindex $results_g3 0] ns"
puts "  Fmax: [lindex $results_g3 1] MHz"
puts "  Groups per dot product: [expr {($K_DIM + 2) / 3}]"
puts ""
puts "G=4 (Our design):"
puts "  WNS:  [lindex $results_g4 0] ns"
puts "  Fmax: [lindex $results_g4 1] MHz"
puts "  Groups per dot product: [expr {($K_DIM + 3) / 4}]"
puts ""
puts "Lookup reduction: [format %.1f [expr {(1.0 - (($K_DIM+3)/4.0) / (($K_DIM+2)/3.0)) * 100}]]%"
puts ""
puts "Detailed reports in: ${OUT_DIR}/reports_G3/ and ${OUT_DIR}/reports_G4/"
puts "############################################################"
