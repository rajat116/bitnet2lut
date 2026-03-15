# ============================================================================
# run_sim_only.tcl — Behavioral simulation for one G value
#
# Usage:
#   vivado -mode batch -source scripts/run_sim_only.tcl -tclargs <G>
#
# Examples:
#   vivado -mode batch -source scripts/run_sim_only.tcl -tclargs 3
#   vivado -mode batch -source scripts/run_sim_only.tcl -tclargs 4
#
# Author: Rajat Gupta
# ============================================================================

if {$argc < 1} {
    puts "Usage: vivado -mode batch -source scripts/run_sim_only.tcl -tclargs <G>"
    exit 1
}

set G [lindex $argv 0]
set K 2560
set PART "xcku5p-ffvb676-2-e"

set TOP_DIR [pwd]
set SRC_DIR "${TOP_DIR}/src"
set TB_DIR  "${TOP_DIR}/tb"
set OUT_DIR "${TOP_DIR}/results"

file mkdir $OUT_DIR

set proj_name "tlmm_sim_G${G}"
set proj_dir "${OUT_DIR}/${proj_name}"

puts "============================================================"
puts "Behavioral Simulation: G=${G}, K=${K}"
puts "============================================================"

if {[file exists $proj_dir]} {
    file delete -force $proj_dir
}

create_project $proj_name $proj_dir -part $PART -force
set_property target_language SystemVerilog [current_project]

add_files [glob ${SRC_DIR}/*.sv]
add_files -fileset sim_1 ${TB_DIR}/tb_tlmm_engine.sv
set_property top tb_tlmm_engine [get_filesets sim_1]
set_property verilog_define "TEST_G=${G} TEST_K=${K}" [get_filesets sim_1]

launch_simulation -mode behavioral
run all

close_project
