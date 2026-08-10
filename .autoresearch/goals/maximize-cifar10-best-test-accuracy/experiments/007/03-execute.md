# EXP-007: Width-2 ResNet-20

## Execution

Overall Status & Info:
- **Created**: 2026-08-05
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-007
- **Commit**: `8faf0f3`
- **Outcome**: completed

## Implementation Notes

### Summary

Changed only `train.py`. Added `WIDTH_MULTIPLIER = 2`, made the existing `ResNet` constructor accept a multiplier defaulting to 1, derived stem/stage/classifier channel dimensions from 16/32/64, and instantiated the training model with the declared multiplier. `BasicBlock`, Option-A stride/padding behavior, initialization, accepted N1/M7-to-weak loader lifecycle, optimizer, schedule, loss, seed, timing, and evaluator are untouched.

### Surprises & Discoveries

- Exact static checks matched the proposal: width 1 has 269,722 parameters, width 2 has 1,073,962, both output `(2,10)`, both contain nine residual blocks, and width-2 transition padding is 32/64.
- The repeated H20 benchmark was highly stable and produced a calibrated projection of 26,469 steps, slightly below the original 26,500 point-estimate gate but comfortably above the externally reviewed operational floor. This empirically validates Claude's plan-review concern that a 26,500 launch gate would have rejected a feasible experiment due to ordinary sub-percent timing variation.
- The disposable benchmark emitted a harmless PyTorch warning when converting the final differentiable loss tensor to a scalar for a finiteness assertion. Timings and candidate code were unaffected; future diagnostics should use `loss.detach().item()`.

### Decisions

- Proceeded at 26,469 calibrated steps because the final reviewed plan intentionally treats 23,000 as a margin-bearing operational floor and 26,000 as a post-run interpretation boundary. The preflight passed every stability, throughput, memory, shape, and parameter gate.
- Kept width 2 rather than substituting width 1.5. The experiment tests the reviewed high-ceiling capacity/update trade; any valid null routes through analysis rather than adaptive width selection.

## Experimental Adjustments

- None to the declared experiment. Pre-commit made one mechanical formatter wrap; candidate behavior is unchanged.

## Preflight Results

- Static: Python compilation, Ruff, pre-commit, and diff checks passed; only `train.py` is modified.
- Architecture: width1/width2 parameters 269,722/1,073,962; both outputs `(2,10)`; nine blocks; width-2 transition pads 32/64.
- Trial 1: width1 mean/median/p95 7.503/7.499/7.565 ms, 330.1 MB; width2 10.928/10.903/10.953 ms, 598.7 MB.
- Trial 2: width2 mean/median/p95 10.968/10.921/10.985 ms, 598.7 MB; width1 7.541/7.535/7.611 ms, 330.1 MB.
- Trial 3: width1 mean/median/p95 7.571/7.529/7.607 ms, 330.1 MB; width2 10.904/10.902/10.946 ms, 598.7 MB.
- Aggregated: control/candidate 7.541/10.928 ms; CV 0.452%/0.299%; ratio 1.4492; raw candidate 27,452 steps at 91.51 steps/s; calibrated 26,469 steps; peak 598.7 MB. All final reviewed gates passed.

## Run Log

### Run 1

Metadata:
- **Job ID**: local training PID 2188551 (supervisor PID 2188547)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-05 13:29 UTC
- **Ended**: 2026-08-05 13:35 UTC

Description:
- One fixed-seed local H20 run of width-2 post-activation ResNet-20 under the complete accepted EXP-004 data and optimizer recipe. The model has 1,073,962 parameters and is expected to complete about 26.5k updates in 300 counted seconds, with N1/M7 through 80% and approximately 13-14 weak-tail epochs. The experiment tests the net fixed-time value of 3.98x capacity against roughly 31% fewer updates. Improvement requires `best_test_acc >= 92.40%` against the 92.30% baseline.

Observations:
- The run progressed continuously with finite loss and no CUDA, OOM, DataLoader, traceback, NaN, or Inf signature. Synchronized steps remained near 11 ms. (source: `run.log`, monitored tails and error scan)
- The final strong checkpoint at epoch 56 was 90.08% with nearest debiased train-loss EMA 0.2283. This is 5.48 points above EXP-004's 84.60% strong checkpoint, direct evidence that width materially improved fitting/representation under N1/M7. (source: `run.log`, transition region; EXP-004 analysis)
- Exactly one switch occurred at 80.0% with all eight workers stopped. The first weak epoch reached 92.96%, then 93.16%, 93.25%, 93.25%, 93.39%, 93.36%, 93.46%, 93.49%, 93.43%, 93.43%, 93.55%, 93.51%, 93.48%, 93.51%, and 93.49% through epoch 71. (source: `run.log`, epoch 57-71 evaluations)
- Best accuracy occurred at epoch 67, four epochs before termination. The final gap was 0.06 points. The least-squares slope over epochs 69-71 was approximately +0.005 points/epoch, effectively flat at the evaluator's 0.01-point resolution rather than still clearly rising. (source: `run.log`, final three evaluations)
- There were 21 evaluator calls at 21 unique epochs; terminal evaluation matched summary epoch 71. (source: `run.log`, all evaluation records)

Key Metrics:
- best_test_acc: 93.55% at epoch 67 (source: `run.log` evaluation trajectory/final summary)
- final_test_acc: 93.49% (source: `run.log` final summary)
- final_test_loss: 0.2196 (source: `run.log` final summary)
- training_seconds: 300.0; total_seconds: 333.0; startup_seconds: 1.0 (source: `run.log` final summary)
- peak_vram_mb: 598.7 (source: `run.log` final summary)
- num_epochs: 71; num_steps: 27,143; num_params: 1,073,962 (source: `run.log` final summary)
- exposure: 70.76% of EXP-004's 38,358 steps, 3,474,304 presented samples, and 11.05 ms mean counted time/step (source: `run.log` summary and derived comparison)

## Verification Results

### Conditions Checked

- **Baseline/hardware/log preconditions — passed**: baseline 92.30% at `11f8469`; exactly one idle H20 with 97,871 MiB; no stale run log. (source: pre-launch command outputs)
- **Completion and summary integrity — passed**: exit 0; all ten unique finite fields; 300.0 counted seconds, 333.0 total <600, and 1,073,962 parameters. (source: `run.log` final summary)
- **Lifecycle/evaluation integrity — passed**: one `randaugment->base` switch at 80.0%, eight workers stopped, 21 unique evaluation epochs, and terminal epoch 71. (source: `run.log` switch/evaluation records)
- **Primary metric — passed**: 93.55% is 1.25 percentage points above the 92.30% moving baseline and 1.15 points above the 92.40% acceptance threshold. (source: `run.log` final summary; baseline query)

### Informational Metrics

- final_test_acc 93.49%; final_test_loss 0.2196; training/total/startup 300.0/333.0/1.0s; peak VRAM 598.7 MB; 71 epochs; 27,143 steps; 1,073,962 parameters. (source: `run.log` final summary)
- Strong/tail trajectory: epoch 56 strong 90.08% at loss EMA 0.2283; first weak 92.96%; best 93.55% at epoch 67; final 93.49%; last-three slope approximately +0.005 points/epoch. (source: `run.log` transition and tail records)

## Errors & Dead Ends

- None in candidate implementation or preflight. The disposable benchmark's scalar-conversion warning is documented under Surprises & Discoveries and did not constitute a failed gate.

## Human Notes

> None; autopilot execution.
