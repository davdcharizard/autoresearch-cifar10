# Experiment Log EXP-026: Activation modernization — ReLU → GELU throughout

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-026.md
- **Plan**: plans/plan-026.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-026 (cut from autoresearch/dev @ 1990397)
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: failed (verification condition 1 not met — converged no-improvement, 96.61 vs 96.81 bar; Runs 1–2 gate-killed: GELU 26.2–27.0ms, SiLU 24.8–25.5ms)

## Implementation Notes

### Summary
Plan-026 Milestone 1 executed exactly: three one-token substitutions `F.relu` → `F.gelu` in train.py — BasicBlock.forward post-bn1, BasicBlock.forward post-residual-add, and the ResNet.forward stem activation. Default erf-exact GELU (no `approximate=` argument). Init, constants, loaders, schedule, optimizer, compile path, and eval untouched. Sanity checks passed: AST parses, `F.relu` count 0, `F.gelu` count 3, `git diff --stat` = 1 file, +3/−3.

### Surprises & Discoveries
- None at implementation time. The three activation sites are the only nonlinearities in the network (adaptive_avg_pool/fc head has none), so the swap is complete with three edits.

### Decisions
- Kept Kaiming init with ReLU gain (√2): GELU's optimal gain differs by <5%, airbench likewise uses standard init with GELU, and changing init would confound the single-variable design.
- Used erf-exact GELU rather than tanh-approximate: it is the canonical op, inductor fuses both equally, and exactness removes one degree of freedom from the analysis.

## Run Log

### Run 1
- **Description**: Full 300s-budget run on GPU 0. First probe of the activation axis in 26 experiments: GELU in place of ReLU at all three sites, motivated by both regime-matched budget-race lineages using smooth activations (airbench GELU, cifar10-fast CELU). Expected: signatures near-baseline (dt 22.4±0.3ms — early gate kills at >23.5ms sustained, ~137–139 epochs, VRAM ≈1613MB, params exactly 4,286,026), early trail at/above family (no deferral), converged plateau ≥0.1pp above baseline. Success bar best_test_acc ≥ 96.81 with final-7 median ≥ 96.6.
- **Job ID**: local background composite, task br75dcah4 (pre-check + launch + inline watchdog with early-dt gate: kill at 3 consecutive windows >23.5ms within first 7 ticks, contention kill at 4 consecutive >30ms, startup gate tick 10, NaN/inf guard)
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: KILLED by early-dt gate (EARLY_DT_KILL, rc=143) — by design, not an error
- **Started**: 2026-06-10T13:25:30Z
- **Ended**: 2026-06-10T13:27:01Z
- **Observations**: erf-GELU is NOT free: steady windowed dt 27.0/26.2/27.0ms vs baseline 22.4 (+~20%, no alternation/spikes — kernel cost, not contention; GPU0 pre-check clean). Projected epochs 139×22.4/26.7 ≈ 117 — far below the 133 floor; epoch deficit ≈ −0.33pp by EXP-006 arithmetic would require GELU to deliver +0.43, implausible. Gate killed at tick 5 (~91s) per plan. Diagnosis: erf is a libdevice special function evaluated forward+backward over every activation tensor (19 sites × large tensors); the fusion happens but the math itself is ~20% of step time. Startup also ~2 ticks slower to first WIN (new graph compile).
- **Key Metrics**: 12 epochs / ~1250 steps before kill; loss trajectory normal (1.55→1.20 — no divergence). Source: task br75dcah4 output.

## Run Log (continued)

### Run 2
- **Description**: Same single-variable experiment with SiLU in place of GELU(erf) at the three activation sites (`sed` swap, AST + counts verified: silu 3, gelu/relu 0). Same composite launcher and gates. If the early-dt gate kills again, Run 3 tries `F.hardswish`; a third gate-out = research failure (smooth activations unaffordable at this dt).
- **Job ID**: local background composite, task bwl1p15vh
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: KILLED by early-dt gate (EARLY_DT_KILL, rc=143) — by design
- **Started**: 2026-06-10T13:28:51Z
- **Ended**: 2026-06-10T13:30:21Z
- **Observations**: SiLU windows 24.8/25.5/25.3ms — cheaper than erf-GELU (26.2–27.0) but still +~2.8ms (+13%) over baseline 22.4; projected epochs ≈124 < 133 floor. Steady windows, no contention signature. Even sigmoid-based smoothness is too expensive at these activation-tensor sizes (19 sites, forward+backward). Loss trajectory normal (1.47→1.21).
- **Key Metrics**: 14 epochs / ~1400 steps before kill. Source: task bwl1p15vh output.

### Run 3
- **Description**: Final activation variant: `F.hardswish` (x·relu6(x+3)/6 — piecewise, comparisons+arithmetic only, NO special functions; MobileNetV3's drop-in cheap swish substitute). Same launcher and gates. If this also gates out >23.5ms, the smooth-activation axis is unaffordable at the 22.4ms dt budget → research failure, proceed to analyze.
- **Job ID**: local background composite, task bdx10djuh
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed, rc=0, clean signatures
- **Started**: 2026-06-10T13:31:14Z
- **Ended**: 2026-06-10T13:39:29Z
- **Observations**: Hardswish passed the gate at 23.1–24.0ms (mean win 23.5, +1.1ms over baseline) → 132 epochs (−7). Pristine profile: 0/254 windows >30ms, expected 132.5 vs 132 actual. TWO diagnostic signatures: (1) ep1 test_acc 20.24% vs baseline family ~38–39 — the swap depressed early learning (likely init-gain/BN-stat mismatch: Kaiming √2 is tuned for ReLU; hardswish has different effective gain and negative outputs), recovering to near-family by ep5 (60.99 vs ~64); (2) best AT the final epoch (96.61 at ep132, plateau 96.4–96.5 before) — still climbing at cutoff, the converged-plateau harvest the max-statistic needs never fully formed. Net 96.61 = −0.10: the smooth-activation gain roughly paid only its own dt cost (7 epochs ≈ −0.10pp by EXP-006 arithmetic) while ALSO paying an early-heat toll.
- **Key Metrics**: best_test_acc 96.61 | final 96.61 | final_test_loss 0.1909 | training_seconds 300.0 | total 477.2s | startup 24.1s (new graph, no warm inductor cache) | VRAM 2015.2MB (+402 — hardswish autograd saves inputs; ReLU backward uses a mask) | 132 epochs | 12764 steps | params 4,286,026 ✓ | eval_lines 132 = num_epochs. Final-7 median ≈ 96.47. Source: run.log summary + task bdx10djuh output.

## Experimental Adjustments

### 2026-06-10 — Run 2: switch GELU(erf) → SiLU after early-dt gate kill
- Evidence: Run 1 windows 26.2–27.0ms (task br75dcah4) — erf math costs ~+4.5ms/step; the smooth-activation hypothesis itself is untested, only erf's cost premise failed.
- Adjustment: replace the three `F.gelu` calls with `F.silu` (sigmoid-based — the cheapest standard smooth activation; tl.sigmoid has a fast libdevice path, backward reuses the forward sigmoid). Same single-variable design, same gate (kill at 3 consecutive windows >23.5ms within first 7 ticks).
- Anticipated in plan/brainstorm: brainstorm-026 § Chosen Idea names SiLU/CELU as one-knob variants; plan-026 risk (a) covers the dt-regression kill. If SiLU also gates out, try `F.hardswish` (piecewise, no special functions) as Run 3; if that too fails, the smooth-activation axis is unaffordable at this dt budget → research failure.

## Errors & Dead Ends

## Verification Results

### Conditions Checked

**Pre-condition — clean post-hoc contention profile**: PASSED. `windows>30ms: 0 of 254 | mean win 23.5 ms | expected epochs 132.5` vs 132 actual (within ±3). Run analyzable, not contaminated. (Note: the 23.5ms mean is the hardswish kernel cost, steady with zero spikes — not contention.)

**Condition 1 — best_test_acc ≥ 96.81 (baseline 96.71 + 0.1pp)**: **FAILED**. `grep "^best_test_acc:" run.log` → **96.61%** (−0.10 vs baseline, −0.20 vs bar). Genuine converged result on a clean run. Three stacked costs visible: dt +1.1ms → −7 epochs (≈ −0.10pp); early-heat toll (ep1 20.24 vs family ~38–39 — init/gain mismatch under peak-LR warmup); plateau still climbing at cutoff (best at final epoch — no converged harvest window). First-failure-stop: conditions 2–3 not evaluated.

**Condition 2 — completes within budget (rc=0, total ≤600s)**: skipped per first-failure-stop; would have passed (TRAIN_EXIT rc=0, total_seconds 477.2).

**Condition 3 — validation at most once per epoch**: skipped per first-failure-stop; would have passed (eval_lines 132 = num_epochs 132).

### Informational Metrics

## Human Notes
