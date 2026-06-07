# EXP-002: k=3 Width + Dynamic T_max + CutMix

## Execution

Overall Status & Info:
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-002.md
- **Plan**: plans/plan-002.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-002
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: failed

## Implementation Notes

### Summary
Widened to k=3 ({48,96,192}, 2.4M params), replaced CutOut with CutMix, and implemented dynamic T_max calibration from epoch 1 timing.

### Surprises & Discoveries
Epoch 1 takes ~21.6s due to torch.compile JIT overhead, but steady-state epochs are ~4.8s. The dynamic calibration used epoch 1 timing, causing it to estimate only 13 total epochs (T_max=10) when the actual total was 62.

### Decisions
None — followed the plan. The calibration bug was an unforeseen consequence of torch.compile's first-epoch overhead.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-28
- **Ended**: 2026-05-28

Description:
- k=3 ResNet-20 with dynamic T_max calibration and CutMix. Expected 35-45 epochs, got 62. T_max calibrated to 10 (from 21.6s first epoch) when it should have been ~57.

Observations:
- Calibration: 21.6s/ep → est 13 epochs, T_max=10 — WRONG. Actual: 62 epochs at ~4.8s/ep steady-state.
- best_test_acc=94.09% (marginal +0.06% over 94.03% baseline, below 0.1% threshold)
- final_test_acc=90.53% — 3.56% gap from best, worst gap yet. Cosine finished at ~epoch 15, LR at min for 47 epochs.

Key Metrics:
- best_test_acc: 94.09% (source: run.log)
- final_test_acc: 90.53% (source: run.log)
- num_epochs: 62 (source: run.log)
- num_params: 2,436,346 (source: run.log)

## Verification Results

### Conditions Checked

1. **Run completion**: exit code 0, best_test_acc present — **PASS**
2. **Time budget**: 300.0s <= 300 — **PASS**
3. **Accuracy improvement**: 94.09% < 94.13% (baseline 94.03% + 0.1%) — **FAIL**
4. Eval frequency: skipped — aborted after prior failure

## Errors & Dead Ends

### 2026-05-28 — Dynamic T_max calibration used epoch 1 timing (includes torch.compile overhead)
- Error: T_max calibrated to 10 based on 21.6s first epoch; actual steady-state was 4.8s/ep (62 epochs total)
- Root cause: torch.compile JIT compilation happens during epoch 1, inflating its timing ~4.5x vs steady-state
- Source: run.log calibration line
- Do NOT retry: never calibrate T_max from epoch 1 when torch.compile is active; use epoch 2+ or average of epochs 2-3

## Human Notes
