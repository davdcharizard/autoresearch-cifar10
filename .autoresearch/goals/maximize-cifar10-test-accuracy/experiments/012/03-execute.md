# EXP-012: Recipe-scalar refresh — weight-decay shaping + label-smoothing retune

## Execution

Overall Status & Info:
- **Created**: 2026-06-29
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-012
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented the EXP-012 plan in `train.py` only. Added `import os`; made `LABEL_SMOOTHING`
env-overridable (`float(os.environ.get("LABEL_SMOOTHING","0.2"))`) and added a new
`WD_SHAPING = os.environ.get("WD_SHAPING","0")=="1"` flag (default OFF). Replaced the
single-group SGD construction with a branch: when `WD_SHAPING` is on, parameters are split
into `decay` (ndim>=2 → conv/fc weight matrices, wd=5e-4) and `no_decay` (ndim<=1 → BN γ/β +
ReZero α, wd=0), guarded by an exact in-run assert `len(no_decay)==2*n_bn+1`; when off, the
exact original single-group call is used (baseline-equivalent). Added five summary prints
(`wd_shaping`, `label_smoothing`, `no_decay_params`, `rezero_alpha`, `rezero_alpha_ema`).
Maps to plan Milestone 1.

### Surprises & Discoveries
None. Smoke confirmed the model has 10 BatchNorm2d layers → 21 ndim<=1 learnable tensors
(2·10 + 1 ReZero α) = 5,505 no-decay params, and 11 ndim>=2 decay tensors (10 learnable convs
+ fc). The frozen whitening conv (requires_grad=False) is correctly excluded from both groups.

### Decisions
Kept `PEAK_LR=0.4` fixed across all cells (per plan) to avoid confounding the WD-shaping
effective-LR effect with an LR change.

## Experimental Adjustments

## Run Log

All four cells ran back-to-back in one background process (exit 0), GPU 1 uncontended
(3 MiB used at launch), ~26.5k img/s throughout. Total wall ~30 min.

### Run 1 — cell-0 (baseline control)

Metadata:
- **Job ID**: background bash b093qbbsh
- **Log file(s)**: run_c0.log (preserved at experiments/012/run_c0.log)
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-29
- **Ended**: 2026-06-29

Description:
- Same-session baseline: `WD_SHAPING=0 LABEL_SMOOTHING=0.2` reproduces EXP-008 training
  behavior to establish the in-session reference accuracy and throughput band.

Observations:
- Clean 150 epochs, 26.5k img/s, no param-group changes (no_decay_params=0). (source: run_c0.log)

Key Metrics:
- best_test_acc: 96.32% @ 150 epochs (source: run_c0.log summary)
- rezero_alpha: -1.2533 | rezero_alpha_ema: -1.2568 (source: run_c0.log)
- training_seconds: 300.0 | total_seconds: 447.9 | peak_vram_mb: 1635.4

### Run 2 — cell-A (WD-shaping headline)

Metadata:
- **Log file(s)**: run_cA.log (preserved at experiments/012/run_cA.log)
- **Status**: completed
- **Started/Ended**: 2026-06-29

Description:
- Headline test: `WD_SHAPING=1 LABEL_SMOOTHING=0.2` isolates the weight-decay-shaping effect.

Observations:
- Param split fired (no_decay_params=5505 = 2·10 BN γ/β + ReZero α). 150 epochs, clean.
- ReZero α magnitude SHRANK vs cell-0 (-0.73 vs -1.25) — removing wd from α did change the gate,
  but toward smaller |α|, opposite to the "uniform wd suppresses the gate" hypothesis, and with
  no accuracy benefit. (source: run_cA.log)

Key Metrics:
- best_test_acc: 96.29% @ 150 epochs (source: run_cA.log summary) — −0.03pp vs same-session cell-0
- rezero_alpha: -0.7296 | rezero_alpha_ema: -0.7146 (source: run_cA.log)
- training_seconds: 300.0 | total_seconds: 444.4

### Run 3 — cell-B (shaping + LS retune bundle)

Metadata:
- **Log file(s)**: run_cB.log (preserved at experiments/012/run_cB.log)
- **Status**: completed
- **Started/Ended**: 2026-06-29

Description:
- Bundle: `WD_SHAPING=1 LABEL_SMOOTHING=0.1` combines shaping with the label-smoothing retune.

Observations:
- 148 epochs (still clean band). α ≈ -0.715 (shaping effect again). LS 0.1 depressed the ceiling.

Key Metrics:
- best_test_acc: 96.16% @ 148 epochs (source: run_cB.log summary) — −0.16pp vs cell-0
- rezero_alpha: -0.7150 | rezero_alpha_ema: -0.6984 (source: run_cB.log)

### Run 4 — cell-C (LS isolation, CutMix-free)

Metadata:
- **Log file(s)**: run_cC.log (preserved at experiments/012/run_cC.log)
- **Status**: completed
- **Started/Ended**: 2026-06-29

Description:
- Clean LS isolation: `WD_SHAPING=0 LABEL_SMOOTHING=0.1` (EXP-011's LS-0.1 was CutMix-confounded).

Observations:
- 150 epochs, α ≈ -1.21 (matches cell-0, as expected with no shaping). LS 0.1 clearly worse.

Key Metrics:
- best_test_acc: 96.09% @ 150 epochs (source: run_cC.log summary) — −0.23pp vs cell-0; LS 0.1 hurts
- rezero_alpha: -1.2108 | rezero_alpha_ema: -1.2085 (source: run_cC.log)

## Verification Results

### Conditions Checked

- **NC1 — completes in budget, valid metric, ≤10 min**: PASS. All four cells: training_seconds=300.0,
  total_seconds 437–448s (<600), exit 0, numeric best_test_acc printed. (source: run_c*.log summaries)
- **NC2 — beats baseline by ≥0.10pp, clearly above noise (≥96.48)**: **FAIL**. Best cell = cell-0
  at 96.32% (the WD-shaping/LS cells are all ≤ cell-0). No cell reaches 96.48, and the headline
  cell-A (96.29) is −0.03pp vs same-session cell-0 (96.32). Anti-bookkeeping passes (max per-epoch
  test_acc == summary best for every cell: c0 96.32, cA 96.29, cB 96.16, cC 96.09). No thin-winner
  → no confirmation re-run needed.
- **NC3 — genuine/in-scope**: PASS (not decisive given NC2 fail, recorded for completeness).
  `git status --porcelain` = only `M train.py`; `git diff --quiet -- prepare.py` clean;
  num_params 7,784,627 (all cells); `manual_seed(42)`/`cuda.manual_seed(42)` intact;
  1 `evaluator.evaluate` call (≤1/epoch).

**Verdict: no-improvement** — all cells valid (clean throughput 148–150 epochs, scope intact)
but none clears NC2. WD-shaping ties the same-session baseline within noise; LS 0.1 degrades.

### Informational Metrics

- peak_vram_mb: 1635.4 (all cells) — param-group split adds nothing. (source: run_c*.log)
- num_epochs: 150/150/148/150 (c0/cA/cB/cC) — all in the clean ≥142 band, spread 2 epochs
  (< the 5-epoch sequential-drift threshold → same-session comparison is sound). (source: run_c*.log)
- num_params: 7,784,627 invariant. (source: run_c*.log)
- rezero_alpha (raw / ema): c0 -1.2533/-1.2568 · cA -0.7296/-0.7146 · cB -0.7150/-0.6984 ·
  cC -1.2108/-1.2085. **Mechanism fired** (WD-shaping cells A/B α magnitude ≈ 0.72 vs no-shaping
  cells 0/C ≈ 1.25), but in the direction of SMALLER |α| and with no accuracy gain. (source: run_c*.log)
- no_decay_params: 5505 (cells A/B) / 0 (cells 0/C) — confirms the shaping group is BN γ/β + α only.

## Errors & Dead Ends

## Human Notes

> (none — autopilot)
