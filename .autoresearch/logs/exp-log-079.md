# EXP-079: Short CutMix Probability Ramp

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-079.md
- **Plan**: plans/plan-079.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-079
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: no-improvement

## Implementation Notes

### Summary

EXP-079 implements the approved short CutMix probability ramp in `train.py` only. It adds `CUTMIX_PROB_START = 0.25`, `CUTMIX_PROB_RAMP_STEPS = 1000`, and `current_cutmix_prob(step)`, then uses the scheduled probability for the existing CutMix Bernoulli sample. The long-run CutMix anchor remains `CUTMIX_PROB=0.5`, and the existing CutMix box, lambda, endpoint loss, clean loss, architecture, optimizer, schedule, transforms, and evaluation code are unchanged.

### Surprises & Discoveries

No implementation surprises. The existing training loop already has the pre-update `step` counter available at the exact CutMix sampling point, so the ramp can be applied without changing optimizer or scheduler order.

### Decisions

The ramp is computed before the batch update from the current completed-step count: step 0 uses probability 0.25, step 1000 and later use probability 0.5. This matches the plan and keeps the full anchor behavior active long before the first LR drop at step 21000.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground session 66783 on GPU0
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 23:02 UTC
- **Ended**: 2026-06-09 23:09 UTC

Description:
- This run tests whether a short early CutMix probability ramp improves the current probabilistic CutMix anchor. It preserves all long-run anchor settings and changes only early CutMix sampling probability from 0.25 toward 0.5 over the first 1000 optimizer steps. The active baseline is 94.11%, so the required improvement threshold is 94.21%.

Observations:
- Preflight scope check passed: `git diff --name-only` listed only `train.py`.
- Preflight syntax check passed: `python3 -m py_compile train.py` exited 0.
- Preflight style check passed: `uv run ruff check train.py` reported `All checks passed!`.
- GPU availability check found GPU0 idle and GPU1 partially active; Run 1 selected GPU0 with `CUDA_VISIBLE_DEVICES=0`.
- Startup confirmed `Device: cuda`, `ResNet-20 | params: 822,790`, unchanged `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`, and `CutMix prob ramp: 0.25 -> 0.5 over 1000 steps`.
- Early training output is being written normally; by step 350 in epoch 1 there were no error signatures.
- The ramp endpoint at step 1000 was reached in epoch 3, after which the run should be using the full `CUTMIX_PROB=0.5` anchor. Early evaluations are healthy, reaching 85.03% best at epoch 13.
- Pre-drop training remained healthy with best 88.49% at epoch 39. The first LR drop was reached at step 21000 in epoch 54, switching to `lr: 0.0100` with about 139s remaining.
- Post-drop convergence is normal but currently below the improvement threshold: best reached 93.95% at epoch 67 and remained 93.95% through epoch 80. No error signatures found in `run.log`.
- The best late result was 94.09% at epoch 82, after which the run stayed below 94.21% through the fixed training budget. The run completed cleanly and reported final summary metrics.

Key Metrics:
- `best_test_acc`: 94.09%
- `final_test_acc`: 93.26%
- `final_test_loss`: 0.2697
- `training_seconds`: 300.0
- `total_seconds`: 394.1
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 104
- `num_steps`: 40,252
- `num_params`: 822,790
- Verdict: no-improvement. This is below the 94.11% baseline and below the 94.21% improvement threshold.

## Verification Results

### Conditions Checked
- Code-scope check passed: the only tracked code change was `train.py`.
- Syntax check passed: `python3 -m py_compile train.py` exited 0.
- Style check passed: `uv run ruff check train.py` reported `All checks passed!`.
- Startup markers confirmed `Device: cuda`, unchanged CutMix `alpha=1.0`, `prob=0.5`, `label smoothing=0.05`, and the ramp marker `0.25 -> 0.5 over 1000 steps`.
- Run completed cleanly and produced numeric final summary metrics in `run.log`.
- No traceback, CUDA, import, shape, NaN, non-finite, or timeout error signature was observed.
- Improvement threshold check failed: `best_test_acc=94.09%` is below the required `94.21%`.

### Informational Metrics
- Peak VRAM was 660.4 MB and the run completed 40,252 optimizer steps in the fixed 300s training budget, matching the current anchor's parameter count and expected local throughput.

## Errors & Dead Ends
- No infrastructure errors or crashes.
- Scientific dead end: the short CutMix probability ramp peaked at 94.09%, below both the 94.11% baseline and the 94.21% improvement threshold.

## Human Notes

> Autopilot execution; no human intervention during implementation.
