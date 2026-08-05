# EXP-024: Two Diagonal Conditional Stage-3 Gates

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-024
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - local-only run
- **Outcome**: failed - scored accuracy missed the acceptance threshold

## Implementation Notes

### Summary

Added two zero-initialized diagonal conditional gates after the stage-3 residual endpoints and before shortcut addition. Each channel scale depends only on its own signed pooled residual; accepted state, initialization RNG, and initial logits remain exact.

### Surprises & Discoveries

No implementation surprises. All four gate vectors received finite nonzero aggregate first-step gradients despite exact-neutral initialization.

### Decisions

Kept all gate vectors in the existing no-decay group because they are one-dimensional parameters. Added no runtime diagnostics so timing measures only the proposed mechanism.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 15596
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-26 19:51:01 UTC
- **Ended**: 2026-07-26 19:57:13 UTC

Description:
- One fixed-seed local score of exact-neutral diagonal self-gating on both accepted stage-3 residual branches. The experiment tests whether per-channel input dependence is sufficient without global cross-channel interaction. Success requires at least 94.17% best accuracy.

Observations:
- Semantic preflight passed exact accepted state/RNG/logits, unit gates, correct residual placement, 692,186 parameters, open gradients, and no-decay grouping. (source: semantic preflight stdout)
- Timing measured accepted/candidate weighted steps at 13.145734/13.429472 ms, retention 0.978872, and 138.901932 projected passes. (source: throughput preflight stdout)
- Scored log initialized on CUDA with 692,186 parameters, a 300-second budget, and 195 batches per epoch. (source: `run.log` initial lines)
- Mixup disabled once at step 17,425 and 195.0 counted seconds. The run completed without errors or duplicate epoch evaluations. (source: `run.log` line 40 and verification commands)
- Final evaluations rose from 93.75% at epoch 130 to 93.91% at epoch 140 but remained below baseline. (source: `run.log` lines 58-65)

Key Metrics:
- Mixup accepted/candidate medians: 13.290412/13.533795 ms; hard-label: 12.877045/13.235730 ms. (source: throughput preflight stdout)
- best/final accuracy: 93.91%/93.91%; final loss: 0.2379. (source: `run.log` lines 64-66)
- training/wall seconds: 300.0/341.7; steps/epochs/passes: 27,141/140/138.961920. (source: `run.log` lines 67-72)
- peak VRAM/parameters: 1094.0 MiB/692,186. (source: `run.log` lines 70 and 73)

## Verification Results

### Conditions Checked

- **Completion/runtime**: PASS - exit 0, finite summary, 300.0 counted and 341.7 wall seconds, correct count/transition, one H20, and unique epoch evaluations. (source: `run.log` lines 40 and 62-73)
- **Primary metric**: FAIL - `best_test_acc=93.91%`, below baseline 94.07% and threshold 94.17%. Verification stopped. (source: `run.log` line 64)
- **Final diff audit/informational collection**: skipped after the metric failure; pre-run audit showed only planned `train.py` changes.

### Informational Metrics

## Errors & Dead Ends

## Human Notes

> Autopilot local-only execution; no intervention requested.
