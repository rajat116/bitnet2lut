# ============================================================================
# run_synth_only.tcl — Quick synthesis for one G value
#
# Usage:
#   vivado -mode batch -source scripts/run_synth_only.tcl -tclargs <G> [<PART>]
#
# Examples:
#   vivado -mode batch -source scripts/run_synth_only.tcl -tclargs 3
#   vivado -mode batch -source scripts/run_synth_only.tcl -tclargs 4
#   vivado -mode batch -source scripts/run_synth_only.tcl -tclargs 4 xc7k325tffg900-2
#
# Author: Rajat Gupta
# ============================================================================

# Parse arguments
if {$argc < 1} {
    puts "Usage: vivado -mode batch -source scripts/run_synth_only.tcl -tclargs <G> \[PART\]"
    puts "  G: group size (3 or 4)"
    puts "  PART: FPGA part (optional, default xcku5p-ffvb676-2-e)"
    exit 1
}

set G [lindex $argv 0]
set PART "xcku5p-ffvb676-2-e"
if {$argc >= 2} {
    set PART [lindex $argv 1]
}

set K 2560
set TOP_DIR [pwd]
set SRC_DIR "${TOP_DIR}/src"
set XDC_DIR "${TOP_DIR}/constraints"
set OUT_DIR "${TOP_DIR}/results"

file mkdir $OUT_DIR

set proj_name "tlmm_G${G}"
set proj_dir "${OUT_DIR}/${proj_name}"

puts "============================================================"
puts "Quick Synthesis: G=${G}, K=${K}, Part=${PART}"
puts "============================================================"

if {[file exists $proj_dir]} {
    file delete -force $proj_dir
}

create_project $proj_name $proj_dir -part $PART -force
set_property target_language Verilog [current_project]

add_files [glob ${SRC_DIR}/*.sv]
add_files -fileset constrs_1 ${XDC_DIR}/timing.xdc

set_property generic "G=${G} K=${K}" [current_fileset]
set_property top tlmm_top [current_fileset]

# Synthesis
synth_design -top tlmm_top -part $PART \
    -generic "G=${G} K=${K}" \
    -mode out_of_context

# Reports
set rpt_dir "${OUT_DIR}/reports_G${G}"
file mkdir $rpt_dir

report_utilization -file "${rpt_dir}/utilization_synth.rpt"
report_timing_summary -file "${rpt_dir}/timing_synth.rpt"

# Implementation
opt_design
place_design
route_design

report_utilization -file "${rpt_dir}/utilization_impl.rpt"
report_timing_summary -file "${rpt_dir}/timing_impl.rpt"
report_power -file "${rpt_dir}/power_impl.rpt"

# Print key metrics
set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
puts ""
puts "============================================================"
puts "RESULTS: G=${G}"
puts "  WNS: ${wns} ns"
puts "  Achieved Fmax: [expr {1000.0 / (3.333 - $wns)}] MHz"
puts "============================================================"

close_project
