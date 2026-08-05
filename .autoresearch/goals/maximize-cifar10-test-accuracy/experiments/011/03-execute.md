# EXP-011: CutMix data-mixing regularization

## Execution

Overall Status & Info:
- **Created**: 2026-06-29
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-011
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed (valid runs; NC2 not met → no-improvement, rendered in analyze)

## Implementation Notes

### Summary
Implemented the CutMix plan entirely in `train.py` (sole editable file), matching `02-plan.md`. Added `import os`; made `LABEL_SMOOTHING` and `CUTMIX_P` env-overridable (defaults 0.2 / 0.5) and added `CUTMIX_ALPHA=1.0`, `CUTMIX_OFF_TAIL_FRAC=0.85`. Added a module-level `cutmix_batch(inputs, alpha)` helper that draws the box center / λ / apply-coin on **CPU** (no CUDA `.item()` sync inside the timed step) and does the on-device permutation paste, returning the area-corrected `lam`. Edited the training step to branch on a CPU-drawn apply-coin (disabled once `progress >= 0.85`), compute the two-term mixed loss `lam·CE(out,y)+(1-lam)·CE(out,y_perm)` reusing the same `criterion` (so label_smoothing composes linearly), and increment a `cutmix_applied` counter when a non-empty box is pasted. Added self-describing config + realized-rate prints to the final summary block. Milestone-1 smoke (50 draws on a dummy `[512,3,32,32]` CUDA tensor) passed: shapes, `0<=lam<=1`, `perm` long shape, and a confined non-empty paste all verified. `ast.parse` clean; `git status` shows only `M train.py`; `prepare.py` byte-unchanged.

### Surprises & Discoveries
- None so far. The training step's `progress` variable (already computed for the LR schedule) was directly reusable for the CutMix tail-disable gate, so no extra time bookkeeping was needed.

### Decisions
- **Env-overridable LABEL_SMOOTHING + CUTMIX_P** (within scope — only train.py): lets the review-mandated LS-interaction 2-cell decision (LS 0.2 → 0.1) and the optional p=0.25 de-risk run without editing the tracked file between runs. Defaults reproduce the proposal's primary config. If a non-default cell wins, the plan requires baking it as the static default and a confirmation re-run so the committed file reproduces the metric.
- **`cutmix_applied` counter + summary prints**: added per plan-review hardening so each `run.log` is self-describing (echoes LS/α/p/off-tail and the realized non-empty CutMix rate), closing the bookkeeping hole from env overrides and making the empty-box/coin path observable.

## Experimental Adjustments
- **Run cell-2 (LABEL_SMOOTHING 0.2→0.1)**: cell-1 landed +0.02pp (96.40, within noise) with a depressed ep25 91.21 (<91.5 pre-registered under-fit threshold), the over-softening signature. Plan pre-registered this exact trigger for the LS-0.1 companion. (ref: Run 1 — ep25 91.21, best 96.40)

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID recorded at launch)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-29
- **Ended**: 2026-06-29

Description:
- Cell-1, the clean single-variable test: CutMix with the proposal's primary config (CUTMIX_P=0.5, CUTMIX_ALPHA=1.0, tail-disable 0.85, LABEL_SMOOTHING=0.2 held = baseline value) on the proven EXP-008 recipe. Run on GPU 1 (free; foreign job is on GPU 0). Expected: ep25 modestly below EXP-008's 92.31 (harder-but-not-broken), tail overtakes; best_test_acc target ≥96.48; num_epochs in the clean band ≥142; cutmix_applied ≈42% of steps. Tests whether region-mixing adds a complementary regularization gain over the existing occlusion aug.

Observations:
- Clean run, full throughput ~26k img/s, no divergence (source: run.log). num_params 7,784,627 unchanged; cutmix_applied 5751/13744 = 41.8% (matches expected ~42%) so CutMix was genuinely applied at p=0.5 with tail-disable.
- **ep25 = 91.21%** vs EXP-008's ~92.31 — depressed ~1.1pp, BELOW the pre-registered ~91.5 under-fit threshold → over-softening/over-augmentation signature (CutMix soft labels stacking with LS 0.2 + Cutout12 + RandomErasing) (source: run.log "eval ep 25").
- Tail fully annealed: best 96.40 at ep133, then dipped to 96.27 by ep142 (peaked-then-dipped = converged, not truncated) (source: run.log "eval ep 133"+).
- Anti-bookkeeping: max per-epoch test_acc 96.40 == summary best_test_acc 96.40 (no tampering).

Key Metrics:
- best_test_acc: 96.40% @ ep133 (source: run.log "best_test_acc:" / "eval ep 133") — baseline 96.38, delta **+0.02pp**, within the ~0.1pp noise floor → fails NC2 (<96.48).
- num_epochs: 142 (clean band ≥142, comparable to baseline) (source: run.log "num_epochs:")
- ep25: 91.21% (under-fit signature) ; training_seconds 300.0 ; total_seconds 449.5 ; peak_vram_mb 1641.4 (source: run.log)

### Run 2

Metadata:
- **Job ID**: (PID recorded at launch)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/run_ls01.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-29
- **Ended**: 2026-06-29

Description:
- Cell-2 (pre-registered fallback, triggered by cell-1's depressed ep25 + below-bar result): identical CutMix config but `LABEL_SMOOTHING=0.1` via env. Tests the over-softening hypothesis — if CutMix's area-mixed soft labels make the heavy LS 0.2 redundant/over-regularizing, lowering LS to 0.1 should lift ep25 back toward ~92 and raise the annealed ceiling, ideally clearing 96.48. Same GPU 1, same seed.

Observations:
- Clean run, full throughput, no divergence; num_epochs 143; cutmix_applied 5747/13820 = 41.6% (source: run_ls01.log).
- **ep25 = 91.65%** — up modestly from cell-1's 91.21 (lower LS does ease the target softening) but STILL below EXP-008's ~92.31. So the ep25 deficit is driven mainly by CutMix's harder mixed task, not by LS; LS-0.1 only partially offsets it (source: run_ls01.log "eval ep 25").
- Tail fully annealed: best 96.32 at ep133, then dipped (source: run_ls01.log "eval ep 133"+). Lowering LS did NOT raise the ceiling — it landed slightly BELOW cell-1 (96.32 < 96.40), so the annealed optimum is unchanged at ~baseline regardless of LS.
- Anti-bookkeeping: max per-epoch test_acc 96.32 == summary best_test_acc 96.32.

Key Metrics:
- best_test_acc: 96.32% @ ep133 (source: run_ls01.log) — baseline 96.38, delta **−0.06pp**, within noise → fails NC2.
- num_epochs: 143 ; ep25: 91.65% ; training_seconds 300.0 ; total_seconds 446.7 ; peak_vram_mb 1641.4 (source: run_ls01.log)

### Run 3 (optional p=0.25 de-risk) — NOT RUN
- The optional `CUTMIX_P=0.25` cell was pre-registered only "if cell-1 reads clearly over-augmented (ep25 depressed AND below baseline)". cell-1's best (96.40) is marginally ABOVE baseline, and cell-2 showed that reducing regularization (via LS) does not raise the ceiling. A weaker CutMix (p=0.25) would only dilute the recipe back TOWARD the EXP-008 baseline (96.38), which cannot clear the +0.10pp bar (96.48). Skipped as it has no path to passing — recorded for transparency.

## Verification Results

### Conditions Checked

Baseline = 96.38 (`exp-index.sh baseline`, commit 07c3760); bar = 96.48 (+0.10pp). Best of two valid cells = **96.40** (cell-1, LS 0.2).

- **NC1 — completes in budget, valid metric, ≤10 min**: **PASS** (both cells). Exit 0; `training_seconds=300.0`; valid `best_test_acc` printed (96.40 / 96.32); `total_seconds` 449.5 / 446.7 < 600. (source: run_cell1_ls02.log, run_cell2_ls01.log)
- **NC2 — beats baseline by ≥0.10pp, clearly above the ~0.1pp noise floor (≥96.48)**: **FAIL**. Best cell 96.40 (+0.02pp) < 96.48; cell-2 96.32 (−0.06pp). Both within the ~0.1pp noise floor → no improvement. Anti-bookkeeping: max per-epoch test_acc == summary best for each cell (96.40==96.40, 96.32==96.32). Multiple-comparison/thin-winner guard: N/A (no cell reached the [96.48,96.55) band). num_epochs clean (142/143 ≥142).
- **NC3 — genuine/in-scope**: **PASS**. `git status` shows only `M train.py`; prepare.py byte-unchanged; num_params 7,784,627 unchanged; seeds `torch.manual_seed(42)`/`torch.cuda.manual_seed(42)` intact (the line-121 `Generator().manual_seed(0)` is the pre-existing LOCAL whitening RNG, untouched); `evaluator.evaluate` called exactly once, at the unchanged per-epoch site (this experiment did not touch the eval path).

**Result**: NC1 PASS, NC2 FAIL, NC3 PASS → valid completed runs, primary metric did not clear the bar ⇒ **no-improvement** (verdict rendered in analyze). Not a crash (valid metrics) and not invalid (no scope/integrity breach).

### Informational Metrics
- peak_vram_mb: 1641.4 (cell-1), 1641.4 (cell-2) (source: run_cell1_ls02.log, run_cell2_ls01.log) — well within the soft VRAM budget.
- num_epochs / num_steps: 142 / 13744 (cell-1), 143 / 13820 (cell-2) — clean throughput band (~26k img/s; CutMix CPU-draws added no measurable cost, confirming throughput-free).
- num_params: 7,784,627 (both) — unchanged.
- cutmix_applied: 41.8% (cell-1), 41.6% (cell-2) — CutMix genuinely active at the expected ~42% rate.
- ep25 (under-fit detector): 91.21% (cell-1, LS0.2), 91.65% (cell-2, LS0.1) vs EXP-008's ~92.31 — CutMix depresses early convergence; LS-0.1 only partially offsets.

## Errors & Dead Ends

## Human Notes

>
