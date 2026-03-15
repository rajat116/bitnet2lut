# ============================================================================
# timing.xdc — Timing constraints for TLMM synthesis
#
# Target: Out-of-context synthesis (no pin assignments)
# Clock: 300 MHz target (3.333 ns period)
#
# We synthesize both G=3 and G=4 with the same clock constraint
# to get a fair Fmax comparison.
# ============================================================================

# Primary clock
create_clock -period 3.333 -name clk [get_ports clk]

# Input/output delay constraints (for timing analysis completeness)
set_input_delay -clock clk 0.5 [get_ports -filter {NAME != clk && DIRECTION == IN}]
set_output_delay -clock clk 0.5 [get_ports -filter {DIRECTION == OUT}]
