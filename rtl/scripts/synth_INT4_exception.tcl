# ============================================================================
# synth_INT4_exception.tcl
# Synthesis of exception-aware INT4 activation quantization
# VALUE_WIDTH=8 because INT4 partial sums max = 4x7 = 28 (fits in 6 bits)
# Exception comparator adds one comparator per activation value
# ============================================================================

set G 4
set K 2560
set ACT_WIDTH 4
set PART "xcku5p-ffvb676-2-e"

set TOP_DIR [pwd]
set SRC_DIR "${TOP_DIR}/src"
set XDC_DIR "${TOP_DIR}/constraints"
set OUT_DIR "${TOP_DIR}/results"
set rpt_dir "${OUT_DIR}/reports_INT4_exception"

file mkdir $OUT_DIR
file mkdir $rpt_dir

puts "============================================================"
puts "SYNTHESIS: Exception-aware INT4 (ACT_WIDTH=4, VALUE_WIDTH=8)"
puts "Target: ${PART}"
puts "============================================================"

set proj_name "tlmm_INT4_exception"
set proj_dir "${OUT_DIR}/${proj_name}"
if {[file exists $proj_dir]} { file delete -force $proj_dir }

create_project $proj_name $proj_dir -part $PART -force
set_property target_language Verilog [current_project]

add_files ${SRC_DIR}/tlmm_engine.sv
add_files ${SRC_DIR}/weight_index_bram.sv
add_files ${SRC_DIR}/tlmm_top.sv
add_files ${SRC_DIR}/exception_comparator.sv
add_files -fileset constrs_1 ${XDC_DIR}/timing.xdc

set_property top tlmm_top [current_fileset]
set_property generic "G=${G} K=${K} ACT_WIDTH=${ACT_WIDTH}" [current_fileset]
update_compile_order -fileset sources_1

# Synthesis
set t0 [clock seconds]
synth_design -top tlmm_top -part $PART \
    -generic "G=${G} K=${K} ACT_WIDTH=${ACT_WIDTH}" \
    -mode out_of_context
set t1 [clock seconds]

report_utilization    -file "${rpt_dir}/utilization_synth.rpt"
report_timing_summary -file "${rpt_dir}/timing_synth.rpt"

# Implementation
opt_design
place_design
route_design

report_utilization    -file "${rpt_dir}/utilization_impl.rpt"
report_timing_summary -file "${rpt_dir}/timing_impl.rpt"
report_power          -file "${rpt_dir}/power_impl.rpt"

set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
set fmax [expr {1000.0 / (3.333 - $wns)}]

puts ""
puts "============================================================"
puts "RESULTS: Exception-aware INT4"
puts "  WNS: ${wns} ns"
puts "  Fmax: ${fmax} MHz"
puts "  Reports: ${rpt_dir}/"
puts "============================================================"

set fp [open "${rpt_dir}/summary.txt" w]
puts $fp "Config: INT4_exception_aware"
puts $fp "ACT_WIDTH: 4"
puts $fp "VALUE_WIDTH: 8"
puts $fp "WNS: ${wns}"
puts $fp "Fmax: ${fmax}"
close $fp

close_project
