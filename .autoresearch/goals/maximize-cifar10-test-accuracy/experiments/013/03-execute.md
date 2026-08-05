# EXP-013: Tail-only Sharpness-Aware Minimization (SAM)

## Execution

Overall Status & Info:
- **Created**: 2026-06-29
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-013
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented tail-only SAM in `train.py` only, per the plan. Added `import os`; two env hyperparameters
`SAM_RHO` (default 0.0 = OFF → baseline) and `SAM_START_FRAC` (default 0.65). Added module-level
`_bn_freeze_stats`/`_bn_restore_stats` (toggle `track_running_stats`, not momentum, so the perturbed
pass updates NO BN running buffers) and a `sam_step()` function doing the full two-pass SAM (1st bwd at
w → ascent e_w=ρ·g/‖g‖ on fp32 master params → 2nd bwd at perturbed point with BN frozen via
try/finally → restore w → optimizer.step()). The training loop gates `use_sam = SAM_RHO>0 and progress
>= SAM_START_FRAC`; off → the byte-identical plain step (`report_loss=loss`), on → `sam_step(...)`,
`sam_step_count += 1`. Logging line switched to `report_loss.item()` (unperturbed 1st-pass loss). Added
`sam_params`/`sam_step_count` pre-loop and 4 summary prints (sam_rho/start_frac/steps/step_frac). Maps
to plan Milestone 1.

### Surprises & Discoveries
None. Smoke confirmed 10 BN layers, 32 trainable tensors (20 BN γ/β + 1 ReZero α + 10 convs + 1 fc),
and all correctness invariants: optimizer stepped, `num_batches_tracked` incremented by exactly +1 per
`sam_step` (the perturbed pass left BN buffers untouched), BN `track_running_stats` restored with no
leftover `_sam_saved_trs`, isolated ascent→restore within atol=1e-5, gate logic correct.

### Decisions
Used `track_running_stats` toggle rather than `momentum=0` for the BN freeze (per plan-review §3) — the
latter would still increment `num_batches_tracked` on the perturbed pass. Refactored SAM into a
module-level `sam_step()` so the smoke exercises the exact code path the loop runs (plan-review §2).

## Experimental Adjustments

## Run Log

All three cells run back-to-back in one background process, GPU 1, same session.

All three cells ran back-to-back in one background process (exit 0), GPU 1 uncontended (3 MiB at
launch), ~26.6k img/s in the plain-SGD phase. No NaN/inf anywhere (the fp32-perturbation ascent was
numerically stable under bf16). Logs preserved at experiments/013/run_c*.log.

### Run 1 — cell-0 (baseline control)
Metadata:
- **Job ID**: background bash bqfa8fum6
- **Log file(s)**: run_c0.log (preserved at experiments/013/run_c0.log)
- **Status**: completed
- **Started/Ended**: 2026-06-29

Description:
- `SAM_RHO=0.0` reproduces EXP-008 training exactly (SAM off) → same-session baseline + throughput band.

Observations:
- Clean 150 epochs, sam_steps=0, 26.6k img/s. A strong same-session draw (96.47, +0.09 over the stored
  96.38 — within the ~0.1pp epoch-jitter floor). (source: run_c0.log)

Key Metrics:
- best_test_acc: 96.47% @ 150 ep (source: run_c0.log summary); final 96.47; training_seconds 300.0; total 431.0.

### Run 2 — cell-A (SAM final 35%, headline)
Metadata:
- **Log file(s)**: run_cA.log (preserved at experiments/013/run_cA.log)
- **Status**: completed
- **Started/Ended**: 2026-06-29

Description:
- `SAM_RHO=0.05 SAM_START_FRAC=0.65` — SAM in the final 35% of the budget.

Observations:
- Gate fired correctly: sam_steps=2670, sam_step_frac=0.223 (the final-35%-of-TIME tail runs at ~2× cost
  so it's ~22% of the step COUNT — consistent). num_epochs 124 (≈ the ~124 prediction, valid ≥110). The
  tail trajectory is monotonically RISING to the last epoch (ep119→124: 96.10→96.13→96.19→96.20→96.27→
  96.29, best==final) → under-anneal signature: the 26 epochs lost to SAM's 2× cost left the net still
  climbing. No NaN/instability. (source: run_cA.log L tail)

Key Metrics:
- best_test_acc: 96.29% @ 124 ep (source: run_cA.log summary) — **−0.18pp vs same-session cell-0 (96.47)**;
  final 96.29; sam_steps 2670; training_seconds 300.0.

### Run 3 — cell-B (SAM final 25%, lighter/fallback)
Metadata:
- **Log file(s)**: run_cB.log (preserved at experiments/013/run_cB.log)
- **Status**: completed
- **Started/Ended**: 2026-06-29

Description:
- `SAM_RHO=0.05 SAM_START_FRAC=0.75` — lighter SAM (final 25%), safer on epochs; controls for under-anneal.

Observations:
- sam_steps=1908, sam_step_frac=0.150; num_epochs 132. Unlike cell-A this one ANNEALED (peaked 96.18 @
  ep129, dipped to 96.11 by ep132 → best 96.18 > final 96.11). So the clean-annealed SAM read (under-anneal
  controlled) still loses to baseline. (source: run_cB.log L tail)

Key Metrics:
- best_test_acc: 96.18% @ 132 ep (source: run_cB.log summary) — **−0.29pp vs same-session cell-0**; final 96.11.

## Verification Results

### Conditions Checked

- **NC1 — completes in budget, valid metric, ≤10 min**: PASS. All three cells: training_seconds=300.0,
  total 414–431s (<600), exit 0, numeric best_test_acc. (source: run_c*.log summaries)
- **NC2 — beats baseline by ≥0.10pp, clearly above noise (≥96.48)**: **FAIL**. Best SAM cell = cell-A
  96.29%, which is < 96.48 AND −0.18pp BELOW same-session cell-0 (96.47). cell-B 96.18 (−0.29pp). No SAM
  cell wins → no confirmation re-run triggered. Anti-bookkeeping passes (max per-epoch test_acc ==
  summary best for every cell: c0 96.47, cA 96.29, cB 96.18). Under-anneal diagnostic: cell-A best==final
  (still climbing at ep124); cell-B is the clean-annealed lighter control (peaked ep129, dipped) and it
  ALSO loses — so under-anneal is controlled for and SAM still provides no lift.
- **NC3 — genuine/in-scope**: PASS (recorded for completeness; not decisive given NC2 fail).
  `git status --porcelain` = only `M train.py`; prepare.py unchanged; num_params 7,784,627 (all cells);
  `manual_seed(42)`/`cuda.manual_seed(42)` intact; 1 `evaluator.evaluate` call.

**Verdict: no-improvement** — all cells valid and stable (no NaN, epochs ≥110), but both SAM cells fall
clearly below the same-session baseline. SAM's flat-minima selection does not survive the ~26-epoch cost
of its 2× tail compute on this small wide-shallow net at 300s.

### Informational Metrics

- peak_vram_mb: 1635.4 (all cells) — the e_w buffers add nothing measurable. (source: run_c*.log)
- num_epochs: 150 / 124 / 132 (c0/cA/cB) — cell-0 throughput-free at 150; SAM cells cost 18–26 epochs as
  designed (both ≥110 valid). (source: run_c*.log)
- sam_steps / sam_step_frac: c0 0/0.000 · cA 2670/0.223 · cB 1908/0.150 — gate fired in the intended tail.
- num_params: 7,784,627 invariant.

## Errors & Dead Ends

## Human Notes

> (none — autopilot)
