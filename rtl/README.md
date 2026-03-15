# TLMM Engine RTL — G=3 vs G=4 FPGA Comparison

## What this is

SystemVerilog RTL for a Table Lookup Matrix Multiply (TLMM) engine that computes
one dot product using precomputed LUT lookups instead of arithmetic.

Parameterized by group size G:
- **G=3** (TeLLMe baseline): 27-entry LUT, 854 groups per K=2560 row
- **G=4** (our design): 81-entry LUT, 640 groups per K=2560 row → 25% fewer lookups

## Files

```
src/
  tlmm_engine.sv         # Core pipelined compute engine
  weight_index_bram.sv   # Weight index storage (BRAM-inferred)
  tlmm_top.sv            # Top wrapper connecting engine + BRAM
tb/
  tb_tlmm_engine.sv      # Self-checking testbench with cycle counting
constraints/
  timing.xdc             # 300 MHz clock target
scripts/
  run_all.tcl            # Full flow: simulate + synthesize G=3 and G=4
  run_synth_only.tcl     # Synthesis only for one G value
  run_sim_only.tcl       # Simulation only for one G value
```

## Step-by-step: Running on your Pitt Vivado machine

### Step 0: Get the files

```bash
cd ~/bitnet2lut        # or wherever your repo is
git pull               # get latest with rtl/ directory
cd rtl
```

### Step 1: Run behavioral simulation (verify correctness)

Simulate G=4 first:
```bash
vivado -mode batch -source scripts/run_sim_only.tcl -tclargs 4
```

Then G=3:
```bash
vivado -mode batch -source scripts/run_sim_only.tcl -tclargs 3
```

**What to check**: Look for "ALL TESTS PASSED" in the console output.
Both G=3 and G=4 should pass all 5 tests. Note the cycle counts.

### Step 2: Run synthesis + implementation (get resource numbers)

Synthesize G=4:
```bash
vivado -mode batch -source scripts/run_synth_only.tcl -tclargs 4
```

Synthesize G=3:
```bash
vivado -mode batch -source scripts/run_synth_only.tcl -tclargs 3
```

Each run takes roughly 5-15 minutes depending on the machine.

**What to check**: At the end, it prints WNS and achieved Fmax.
Full reports are saved to `results/reports_G3/` and `results/reports_G4/`.

### Step 3: Or run everything at once

```bash
vivado -mode batch -source scripts/run_all.tcl
```

This runs simulation + synthesis + implementation for both G=3 and G=4,
and prints a comparison summary at the end.

### Step 4: Collect results

The key files you need:

```
results/reports_G3/utilization_impl.rpt   # LUT, FF, BRAM usage for G=3
results/reports_G3/timing_impl.rpt        # Fmax for G=3
results/reports_G3/power_impl.rpt         # Power estimate for G=3

results/reports_G4/utilization_impl.rpt   # LUT, FF, BRAM usage for G=4
results/reports_G4/timing_impl.rpt        # Fmax for G=4
results/reports_G4/power_impl.rpt         # Power estimate for G=4
```

Copy these back (or screenshot them) — these are the numbers for the paper.

## Changing the target FPGA

Edit the PART variable in the scripts, or pass it as argument:

```bash
# Kintex-7 (free WebPACK license)
vivado -mode batch -source scripts/run_synth_only.tcl -tclargs 4 xc7k325tffg900-2

# Zynq UltraScale+
vivado -mode batch -source scripts/run_synth_only.tcl -tclargs 4 xczu7ev-ffvc1156-2-e
```

**Important**: Check which parts your Vivado license supports.
The free WebPACK edition supports Artix-7, Kintex-7, Zynq-7000, and some
Zynq UltraScale+ devices. UltraScale+ Kintex may need a paid license.

If you only have WebPACK, change PART to `xc7k325tffg900-2` (Kintex-7 325T).

## Expected comparison (from analytical model)

| Metric | G=3 (TeLLMe) | G=4 (Ours) | Difference |
|--------|-------------|-----------|------------|
| Groups per row (K=2560) | 854 | 640 | -25.1% |
| LUT entries | 27 | 81 | +200% |
| BRAM18K bits used | 432 | 1,296 | Both ≪ 18,432 |
| Lookups per token | 695M | 521M | -25.0% |
| Cycles per dot product | ~856 | ~642 | -25.0% |

The synthesis results should confirm similar Fmax and resource usage
between G=3 and G=4, validating that the 25% cycle reduction is "free".
