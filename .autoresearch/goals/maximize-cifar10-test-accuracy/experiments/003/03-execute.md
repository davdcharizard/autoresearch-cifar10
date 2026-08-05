# EXP-003: Early CutMix With a Hard-Label Tail

## Execution

Overall Status & Info:
- **Created**: 2026-07-24
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-003
- **Commit**: (pending — committed on loop success)
- **PR**: N/A — local-only workflow requested
- **Outcome**: failed

## Implementation Notes

### Summary

Replaced the first 65% alpha-0.2 mixup path in `train.py` with shared-rectangle, area-corrected CutMix while preserving the full accepted WRN-16-2 architecture, optimizer, time-based schedule, seed, loader, hard-label tail, evaluation cadence, and summary contract. A dedicated CPU generator supplies the mathematically equivalent `Beta(1,1)` uniform coefficient and rectangle center without a CUDA scalar synchronization; the image permutation remains device-local. Exact pasted area controls the two-label loss and is summarized at the single 65% transition.

### Surprises & Discoveries

The alternating synthetic benchmark showed no measurable throughput penalty: matched mixup projected 121.6 passes and CutMix 121.8, a ratio of 1.0017. Destination cloning plus an explicitly materialized donor patch passed pixel-identity tests without requiring per-example loops.

### Decisions

Used a small pure `cutmix_bounds` helper so floor rounding, clipping, and the zero-area case can be tested directly. The CutMix sampling generator is seeded from the existing `torch.initial_seed()` rather than a new result-selected seed, and is isolated from the DataLoader's CPU RNG stream. The experiment does not claim a bit-identical CUDA RNG trajectory relative to mixup because the regularizer necessarily changes random draws.

## Experimental Adjustments

None.

## Run Log

### Run 1

Metadata:
- **Job ID**: local timeout PID 1134568
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-24 12:39:43 UTC
- **Ended**: 2026-07-24 12:45:50 UTC

Description:
- Run the accepted WRN-16-2 configuration with area-corrected `Beta(1,1)` CutMix during the first 65% of counted training and the unchanged hard-label cosine path for the final 35%. This tests whether localized sample composition improves spatial generalization beyond the 94.07% mixup baseline without reducing exposure. Success requires `best_test_acc >= 94.17%`, a complete 300-second training summary, and total runtime below 600 seconds.

Observations:

- Process launched successfully and `run.log` confirmed CUDA, WRN-16-2 with 691,674 parameters, a 300-second budget, and 195 batches per epoch. (source: local PID 1134568; `run.log` L1-L4)
- CutMix disabled exactly once at 195.0 counted seconds (65.0%), epoch 92, step 17,886, LR 0.0612, with mean pasted-area fraction 0.3099. (source: `run.log` L42)
- The hard-label tail recovered to 93.72% best accuracy at epoch 140 and finished at 93.70%, below the accepted 94.07% mixup baseline despite normal exposure. (source: `run.log` L62-L67)

Key Metrics:

- `best_test_acc`: 93.72% @ epoch 140 (source: `run.log` L62, L66)
- `final_test_acc`: 93.70%; `final_test_loss`: 0.2844 (source: `run.log` L67-L68)
- `training_seconds`: 300.0; `total_seconds`: 340.1 (source: `run.log` L69-L70)
- `num_steps`: 27,831; `num_epochs`: 143; realized dataset passes: 142.5 (source: `run.log` L73-L74)
- `peak_vram_mb`: 1,094.0; `num_params`: 691,674 (source: `run.log` L72, L75)
- CutMix mean pasted-area fraction: 0.3099 (source: `run.log` L42)

## Verification Results

### Conditions Checked

- **Run completes without crashing and within 10 minutes**: PASS. Exit code 0, complete summary, one H20, `training_seconds=300.0`, `total_seconds=340.1`, unique every-fifth-plus-final evaluation records, and only `train.py` modified. (source: process exit; `run.log` L6-L75; git status)
- **`best_test_acc` exceeds baseline by at least 0.1 percentage points**: FAIL. 93.72% is 0.35 points below the 94.07% baseline and 0.45 points below the 94.17% threshold. Verification stopped on this necessary-condition failure. (source: `04-results.tsv`; `run.log` L66)

### Informational Metrics

Skipped under the verification protocol after the necessary metric condition failed; the observed values are retained in Run 1 Key Metrics for analysis.

## Errors & Dead Ends

None.

## Human Notes

> The user requested a fully offline, local workflow with no GitHub CLI or remote PR operations.
