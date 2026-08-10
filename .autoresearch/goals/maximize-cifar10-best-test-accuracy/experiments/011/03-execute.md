# EXP-011: Increase CutMix Probability to 0.75

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-011
- **Commit**: (pending — committed on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Changed exactly `CUTMIX_PROBABILITY = 0.5` to `0.75` in `train.py`. The complete accepted EXP-010 CutMix implementation, RNG isolation, strong/weak loader lifecycle, model, optimizer, schedule, seed, timer, and evaluator remain byte-identical.

### Surprises & Discoveries

None. The exact diff is one literal and all static checks pass.

### Decisions

EXP-010 already established hard/soft GPU parity and full lifecycle feasibility, so the proportional preflight measures only the changed host behavior: p=0.75 target provenance, real-loader throughput, target integrity, and worker shutdown.

## Experimental Adjustments

- None.

## Run Log

### Run 1

Metadata:
- **Job ID**: PID 2262669 (timeout supervisor 2262668; tool session 73507)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-06 03:51 UTC
- **Ended**: 2026-08-06 03:57 UTC

Description:
- One fixed-seed run increases only the share of alpha-1 CutMix N1/M7 plateau batches from 50% to 75%. The hypothesis predicts stronger regional invariance will raise the 94.15% baseline to at least 94.25% without material exposure loss; the existing hard weak tail remains the recovery stage.

Observations:
- Compilation, Ruff, formatting, pre-commit, whitespace, and exact one-line diff checks passed.
- The p=0.75 real-loader preflight delivered 179.02 batches/s, 751/1,000 mixed batches (75.10%), valid hard/probability targets, and eight clean worker exits.
- Full-run startup succeeded on CUDA with 1,073,962 parameters; early loss remained finite with approximately 11 ms synchronized steps. (source: `run.log` startup/progress records)
- The strong checkpoint was 86.82%, below the 87.08 underfit marker, with 16,151/21,502 mixed batches (75.11%) and eight clean worker exits. The first weak result jumped to 93.40%, but the tail plateaued at 94.00%. (source: `run.log` L14-L43)
- Exit 0 produced all finite summary fields after 300.0 counted and 332.9 total seconds; no traceback, OOM, assertion, `nan`, or `inf` occurred. (source: process status; `run.log` L45-L54)

Key Metrics:
- best/final test accuracy: 94.00% / 94.00% (source: `run.log` L45-L46)
- final test loss: 0.1933 (source: `run.log` L47)
- final train-loss EMA: 0.0608 @ step 26,900 (source: final progress record)
- training/total/startup: 300.0s / 332.9s / 1.0s (source: `run.log` L48-L50)
- peak VRAM 598.7 MB; 70 epochs; 26,919 steps; 1,073,962 parameters (source: `run.log` L51-L54)
- realized CutMix: 16,151/21,502 = 75.11% (source: `run.log` L15)

## Verification Results

### Conditions Checked

- **Primary accuracy**: failed. 94.00% <94.25%; -0.15 points versus the 94.15 baseline. No retry is allowed. (source: index baseline; `run.log` L45)
- **Run validity**: passed before verdict. Exit 0, ten finite fields, 300.0 counted seconds, 332.9 total, one H20, only `train.py`, one switch/eight workers, 20 evaluations on 20 unique epochs, and exact parameter count.
- **Mechanism integrity**: passed. 75.11% mixing, 26,919 steps (100.08% of EXP-010), unchanged memory. The 86.82% switch checkpoint crossed the underfit diagnostic and explains the failed recovery.
- Remaining success-only checks stopped after the primary failure; valid result must not be rerun.

### Informational Metrics

- Formal success-only collection skipped. Metrics required for no-improvement analysis are preserved under Run 1 Key Metrics.

## Errors & Dead Ends

## Human Notes

> Autopilot session; no execution-phase intervention.
