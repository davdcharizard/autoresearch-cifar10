# EXP-072: AugMix `all_ops=False` — geometric-only AugMix (drop 4 photometric ops)

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-072.md
- **Plan**: plans/plan-072.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-072
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Applied Milestone 1 verbatim: single-kwarg edit at train.py L171, `transforms.AugMix()` → `transforms.AugMix(all_ops=False)`, still wrapped in `RandomApply(..., p=0.5)`. Everything else byte-identical to EXP-054. Smoke test passed: AST OK; `AugMix(all_ops=False)._augmentation_space(31,(32,32))` returns EXACTLY the 9 geometric/lossless ops {ShearX, ShearY, TranslateX, TranslateY, Rotate, Posterize, Solarize, AutoContrast, Equalize} and NONE of the 4 photometric ops {Brightness, Color, Contrast, Sharpness}; the transform applies cleanly to a dummy 32×32 PIL image; `git diff --name-only` == train.py only.

### Surprises & Discoveries
- Confirmed in-env (tv 0.24.1) that `AugMix.__init__` defaults `all_ops=True`, so EXP-054's plain `transforms.AugMix()` was silently running the FULL 13-op pool (photometric ops active). This experiment is therefore a clean isolation of those 4 ops' contribution — the op-set-composition dimension never previously probed (all prior AugMix experiments varied count/magnitude/alpha/coverage).

### Decisions
- No deviations from the plan. Kept `RandomApply(p=0.5)` and all AugMix scalars (severity 3, width 3, depth -1, alpha 1.0) fixed — single-variable isolation of `all_ops`.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (background bash, PID at launch)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed (exit 0)
- **Started**: 2026-06-10
- **Ended**: 2026-06-10

Description:
- Running the EXP-072 geometric-only AugMix probe: `CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1` on idle GPU 1 (both GPUs 0%/0 MiB at launch). Tests whether removing AugMix's 4 photometric ops (Brightness/Color/Contrast/Sharpness) helps or hurts clean CIFAR-10 top-1 vs the 96.45 baseline. Expected: within ±0.3pp of 96.45 (most likely), with a genuine upside shot at clearing 96.55 iff the photometric ops were net-harmful to CIFAR color discrimination. dt should stay 8ms (AugMix is CPU-side, off the timed step).

Observations:
- **Early gate (≤ep7) PASSED**: dt steady 8ms (normal 9-12ms jitter), img/s ~15,300 — no cudagraph break, no contention; the CPU-side op-menu change is throughput-neutral as predicted. Eval climbing strongly and normally: ep4 66.77%, ep5 73.51%, ep6 77.23% — healthy trajectory tracking EXP-054 (slightly faster early-climb, consistent with geometric-only AugMix being marginally easier than the 13-op pool). lr at peak 0.200 annealing. No NaN. (source: run.log eval ep4-6 lines, steps 1600-2700)

Key Metrics:
- ep4 test_acc: 66.77%; ep5 73.51%; ep6 77.23% (source: run.log "eval ep" lines)
- **best_test_acc: 96.43%** (−0.02pp vs baseline 96.45 — a virtual TIE, closest any post-EXP-054 run has come) (source: run.log "best_test_acc:")
- final_test_loss: 0.1911 (LOWER than EXP-054's 0.1968 — better calibration); training_seconds 300.0; total_seconds 570.3; num_epochs 92; num_steps 35636; num_params 4,299,866; peak_vram_mb 453.8 (source: run.log summary)

## Verification Results

### Conditions Checked

- **Necessary condition 1 — `best_test_acc >= 96.55`**: best_test_acc = **96.43** < 96.55. **FAILED** (−0.12pp below bar, −0.02pp below baseline 96.45 — within noise, a virtual tie). Stop at first failed condition.
- **Necessary condition 2 — clean completion within budget** (recorded for completeness): training_seconds 300.0 ✓, total_seconds 570.3 < 600 ✓, num_params 4,299,866 UNCHANGED ✓, 0 nan/traceback/error ✓, 92 epochs.
- **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == train.py only ✓; no new deps; seed 42; evaluate() once/epoch; uncontended (dt steady 8ms).

**Verdict: no-improvement.** Clean valid run (Σdt=300.0, wall 570.3 < 600, dt 8ms / no graph break, train.py only) that missed the bar by a hair (96.43, within noise of baseline). Results trustworthy — direct metric parse, 0 NaN, healthy trajectory. NOT invalid (no breach; aug is data-side, params unchanged) and NOT crash (real interpretable metric).

### Informational Metrics

- num_epochs 92 / num_steps 35636 (vs EXP-054's 91 — geometric-only AugMix is marginally cheaper on CPU, bought ~1 extra epoch; wall 570.3s vs 593s).
- **final_test_loss 0.1911 — LOWER than EXP-054's 0.1968** (better calibration), near the best-ever polish runs (GC 0.1894, grad-clip 0.1939). Classic polish-vs-top1: dropping the 4 photometric ops improved the loss but left top-1 flat.
- peak_vram_mb 453.8.
- **Key observation**: removing AugMix's 4 photometric ops (Brightness/Color/Contrast/Sharpness) is NEAR-NEUTRAL for top-1 (−0.02pp, the smallest post-EXP-054 delta — far less than the −0.2..−0.6pp scalar-knob band) AND improves loss. The photometric ops are NOT load-bearing for top-1 on this net; the geometric+lossless ops carry the diversity benefit. The op-set-composition axis is essentially flat near the optimum — confirms the augmentation lever is genuinely exhausted (even removing 4 ops doesn't move top-1).

## Errors & Dead Ends

<!-- none -->

## Human Notes

> (none — autopilot)
