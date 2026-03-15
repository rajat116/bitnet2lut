# ============================================================================
# sim_G3.tcl — Behavioral simulation for G=3 (TeLLMe baseline)
#
# Usage: Open Vivado TCL console, then:
#   cd C:/path/to/bitnet2lut/rtl
#   source scripts/sim_G3.tcl
#
# Author: Rajat Gupta
# ============================================================================

set G 3
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
set_property target_language Verilog [current_project]

# Add RTL sources
add_files ${SRC_DIR}/tlmm_engine.sv
add_files ${SRC_DIR}/weight_index_bram.sv
add_files ${SRC_DIR}/tlmm_top.sv

# Add testbench
add_files -fileset sim_1 ${TB_DIR}/tb_tlmm_engine.sv
set_property top tb_tlmm_engine [get_filesets sim_1]
set_property verilog_define "TEST_G=${G} TEST_K=${K}" [get_filesets sim_1]

# Update and compile
update_compile_order -fileset sim_1

puts "Launching simulation..."
launch_simulation -mode behavioral
run all

puts "============================================================"
puts "Simulation complete for G=${G}. Check transcript for PASS/FAIL."
puts "============================================================"

close_project
