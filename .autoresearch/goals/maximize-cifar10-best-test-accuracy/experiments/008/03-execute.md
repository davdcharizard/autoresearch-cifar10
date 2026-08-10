# EXP-008: Width-2 Weight Decay 5e-4

## Execution

Overall Status & Info:
- **Created**: 2026-08-05
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-008
- **Commit**: (pending — committed on loop success)
- **Outcome**: failed (valid run, primary metric no-improvement)

## Implementation Notes

### Summary

Changed exactly one literal in `train.py`: coupled SGD weight decay from `1e-4` to `5e-4`. The accepted width-2 architecture, N1/M7 strong phase, 80% worker transition, weak hard-label tail, LR schedule, momentum, loss, seed, evaluator, and all timing/logging code remain unchanged.

### Surprises & Discoveries

- None. Compilation, Ruff, pre-commit, and the exact diff passed immediately.
- A disposable CPU check confirmed 1,073,962 parameters, `(2,10)` output, one optimizer group, LR 0.1, momentum 0.9, and weight decay 0.0005.

### Decisions

- Added no parameter grouping or norm logging. Applying canonical coupled decay to the existing parameter set is the reviewed experiment; additional reductions would change the fixed-time graph.
- Step retention will be recorded for attribution but cannot veto a goal-valid accuracy result because the scalar does not add operations.

## Experimental Adjustments

- None.

## Run Log

### Run 1

Metadata:
- **Job ID**: local training PID 2200260 (supervisor PID 2200256)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-05 14:09 UTC
- **Ended**: 2026-08-05 14:14 UTC

Description:
- One fixed-seed local H20 run of the accepted width-2 model with only coupled SGD decay raised to `5e-4`. The run preserves N1/M7 through 80%, the weak hard-label tail, and all 300-second mechanics. It tests whether plateau norm control carries improved generalization into the tail without compute loss. Formal improvement requires `best_test_acc >= 93.65%` against the 93.55% baseline.

Observations:
- The run completed with finite loss and no CUDA, OOM, DataLoader, traceback, NaN, or Inf signature. Step time stayed near 11-12 ms. (source: `run.log`, monitored tails/error scan)
- The pre-registered underfit signature was strong: epoch-55 switch accuracy/loss EMA were 81.29%/0.4148 versus EXP-007's 90.08%/0.2283. The first weak checkpoint was 91.63% versus 92.96%. (source: `run.log` transition; EXP-007 analysis)
- Tail train-loss EMA declined from 0.2119 to 0.0770 while accuracy rose from 91.63% to 93.38%. Test loss reached a 0.1975 minimum at epoch 67 and ended 0.0013 higher at 0.1988. (source: `run.log` epoch 56-69 progress/evaluations)
- Accuracy was still rising late: 93.34%, 93.38%, 93.38% over epochs 67-69, an endpoint/least-squares slope of about +0.02 points/epoch; best equaled final. The shorter 69-epoch run did not recover EXP-007's 93.55%. (source: `run.log` final evaluations)
- Exactly one switch occurred at 80.0% with eight workers stopped; 19 evaluator calls occurred at 19 unique epochs, terminal epoch 69. (source: `run.log` lifecycle/evaluation records)

Key Metrics:
- best_test_acc/final_test_acc: 93.38% at epoch 69 (source: `run.log` final summary)
- final_test_loss: 0.1988; minimum test loss 0.1975 at epoch 67 (source: `run.log` tail/summary)
- training/total/startup: 300.0/332.2/1.0 seconds (source: `run.log` final summary)
- peak_vram_mb: 598.7; num_epochs: 69; num_steps: 26,729; num_params: 1,073,962 (source: `run.log` final summary)
- step retention: 98.48% of EXP-007's 27,143; mean counted step time 11.22 ms (source: final summary and derived comparison)

## Verification Results

### Conditions Checked

- **Baseline/hardware/log preconditions — passed**: baseline 93.55% at `8faf0f3`; one idle H20; no stale log; one-literal diff. (source: pre-launch outputs)
- **Completion/summary integrity — passed**: exit 0; all ten fields exactly once and finite; 300.0 counted seconds, 332.2 total <600, and 1,073,962 parameters. (source: `run.log` final summary)
- **Lifecycle/evaluation integrity — passed**: one 80.0% switch, eight workers stopped, 19 unique evaluation epochs ending at epoch 69. (source: `run.log` lifecycle/evaluation records)
- **Primary metric — failed**: 93.38% is 0.17 points below the 93.55% baseline and 0.27 below the 93.65% acceptance threshold. Verification stopped on this necessary-condition failure. (source: `run.log` summary; baseline query)

### Informational Metrics

- Skipped as a formal verification section after primary failure; all summary values and the pre-registered underfit/tail diagnostics are preserved above for analysis.

## Errors & Dead Ends

- None.

## Human Notes

> None; autopilot execution.
