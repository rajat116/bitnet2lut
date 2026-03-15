# ============================================================================
# synth_G4.tcl — Synthesis + Implementation for G=4 (our design)
#
# Usage: Open Vivado TCL console, then:
#   cd C:/path/to/bitnet2lut/rtl
#   source scripts/synth_G4.tcl
#
# Or from command line:
#   vivado -mode batch -source scripts/synth_G4.tcl
#
# Output reports saved to: results/reports_G4/
#
# Author: Rajat Gupta
# ============================================================================

set G 4
set K 2560
set PART "xcku5p-ffvb676-2-e"

set TOP_DIR [pwd]
set SRC_DIR "${TOP_DIR}/src"
set XDC_DIR "${TOP_DIR}/constraints"
set OUT_DIR "${TOP_DIR}/results"

file mkdir $OUT_DIR

set proj_name "tlmm_synth_G${G}"
set proj_dir "${OUT_DIR}/${proj_name}"
set rpt_dir "${OUT_DIR}/reports_G${G}"
file mkdir $rpt_dir

puts "============================================================"
puts "SYNTHESIS: G=${G}, K=${K}"
puts "Target: ${PART}"
puts "============================================================"

if {[file exists $proj_dir]} {
    file delete -force $proj_dir
}

create_project $proj_name $proj_dir -part $PART -force
set_property target_language Verilog [current_project]

# Add RTL sources
add_files ${SRC_DIR}/tlmm_engine.sv
add_files ${SRC_DIR}/weight_index_bram.sv
add_files ${SRC_DIR}/tlmm_top.sv

# Add constraints
add_files -fileset constrs_1 ${XDC_DIR}/timing.xdc

# Set top module and parameters
set_property top tlmm_top [current_fileset]
set_property generic "G=${G} K=${K}" [current_fileset]

update_compile_order -fileset sources_1

# ---- Synthesis ----
puts "\n--- Running Synthesis ---"
set t0 [clock seconds]

synth_design -top tlmm_top -part $PART \
    -generic "G=${G} K=${K}" \
    -mode out_of_context

set t1 [clock seconds]
puts "Synthesis completed in [expr {$t1 - $t0}] seconds"

# Post-synthesis reports
report_utilization -file "${rpt_dir}/utilization_synth.rpt"
report_timing_summary -file "${rpt_dir}/timing_synth.rpt"

# ---- Implementation ----
puts "\n--- Running Implementation ---"
set t2 [clock seconds]

opt_design
place_design
route_design

set t3 [clock seconds]
puts "Implementation completed in [expr {$t3 - $t2}] seconds"

# Post-implementation reports
report_utilization -file "${rpt_dir}/utilization_impl.rpt"
report_utilization -hierarchical -file "${rpt_dir}/utilization_hier.rpt"
report_timing_summary -file "${rpt_dir}/timing_impl.rpt"
report_power -file "${rpt_dir}/power_impl.rpt"

# ---- Print Summary ----
set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
set achieved_period [expr {3.333 - $wns}]
set achieved_fmax [expr {1000.0 / $achieved_period}]

puts ""
puts "============================================================"
puts "RESULTS: G=${G}, K=${K}"
puts "  Target clock: 300 MHz (3.333 ns)"
puts "  WNS (slack): ${wns} ns"
puts "  Achieved Fmax: ${achieved_fmax} MHz"
puts "  Synth time: [expr {$t1 - $t0}] s"
puts "  Impl time: [expr {$t3 - $t2}] s"
puts ""
puts "Reports saved to: ${rpt_dir}/"
puts "============================================================"

# Save summary to file
set fp [open "${rpt_dir}/summary.txt" w]
puts $fp "G = ${G}"
puts $fp "K = ${K}"
puts $fp "NUM_GROUPS = [expr {(${K} + ${G} - 1) / ${G}}]"
puts $fp "LUT_ENTRIES = [expr {int(pow(3, ${G}))}]"
puts $fp "Part = ${PART}"
puts $fp "WNS = ${wns} ns"
puts $fp "Fmax = ${achieved_fmax} MHz"
puts $fp "Synth time = [expr {$t1 - $t0}] s"
puts $fp "Impl time = [expr {$t3 - $t2}] s"
close $fp

close_project
