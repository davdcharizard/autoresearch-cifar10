# EXP-032: Reset Momentum at the 80% Objective Boundary

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-032
- **Commit**: (pending — committed on loop success)
- **Outcome**: no-improvement — valid 93.89% best missed the 94.25% gate

## Implementation Notes

### Summary

Added one `@torch.no_grad()` helper that requires and zeroes every live SGD momentum buffer in place. It is called exactly once after the accepted 80% switch evaluation, worker shutdown, and weak-loader construction, before the first weak update; the existing switch log reports the reset count. No model, scalar, data, LR, timer, or evaluator logic changed. Static quality and exact-scope checks passed.

### Surprises & Discoveries

None during tracked implementation. Both preregistered corpora already existed and matched their registered hashes during planning.

### Decisions

The copied-state controller trains one source through the immutable strong corpus and forks both arms from that exact boundary, so post-reset divergence is attributable to buffer clearing rather than separately trained CUDA trajectories. Absolute production steps are informational; exactly 19 evaluations are required for max-metric comparability.

## Experimental Adjustments

- None.

## Run Log

### Run 1

Metadata:
- **Job ID**: local timeout PID 2532457 (execution session 66454)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log
- **WandB**: N/A
- **Status**: completed, exit 0, no retry
- **Started**: 2026-08-06 19:49:48 UTC
- **Ended**: 2026-08-06 19:55:13 UTC

Description:
- Conditional one-H20 seed-42 run of a once-only full SGD momentum reset at the accepted 80% objective/LR boundary. Production requires exact copied-state recurrence and safety gates. Success requires 94.25% with exactly 19 evaluation looks and no valid-result rerun.

Observations:
- Copied-state preflight passed after one controller-only equality fix. All 59 buffers changed from aggregate norm 1.295493 to exact zero while model/BN/gradients/RNG/groups/logits remained unchanged. (source: `experiments/032/preflight-report.json`)
- Across 64 exact weak batches, first/max candidate-control update ratios were 0.532279/1.030338, maximum relative parameter update was 0.000348951, own-16-step-median max was 1.231899, terminal loss-EMA ratio was 1.000231, and no candidate-only concentration occurred. (source: `experiments/032/preflight-report.json`)
- Production switched once at 80.0%, stopped eight workers, reset exactly 59 buffers, and reported 10,791/21,675 strong CutMix batches (49.78%). Switch/first-weak accuracy was 89.15%/93.21%. (source: `run.log` switch and epoch 56-57 lines)
- The tail peaked at 93.89% in epoch 65 and finished at 93.84% with 0.2047 NLL. The run exited zero after 300.0 counted / 331.5 total seconds, 27,039 steps, 70 epochs, exactly 19 unique evaluations, and no error signal. (source: `run.log` final trajectory/summary)

Key Metrics:
- Preflight reset count 59; buffer norm 1.295493 -> 0; no concentration; all safety thresholds passed.
- `best_test_acc=93.89%`, delta -0.26 points vs 94.15 and -0.36 below gate; `final_test_acc=93.84%`, `final_test_loss=0.2047`.
- Exposure 27,039 steps (100.52% of EXP010); peak VRAM 598.7 MiB; best-final gap 0.05.

## Verification Results

### Conditions Checked

- **Scope/safety — pass:** exact `train.py` diff, 59-buffer state invariance/recurrence, exact corpora, finite trajectory, and all spike/concentration gates passed.
- **Completion/integrity — pass:** exit 0, finite ten-field summary, 300.0/331.5 seconds, one 80% switch, 8 expected workers stopped, 59 buffers reset, 49.78% CutMix, hard weak targets, 1,073,962 parameters, and no errors.
- **Evaluator — pass:** exactly 19 unique evaluation epochs including terminal, no repeated epoch.
- **Primary metric — fail:** 93.89% <94.25%; valid no-improvement, no rerun.

### Informational Metrics

- Final accuracy/loss 93.84%/0.2047; training/total/startup 300.0/331.5/1.0s; VRAM 598.7 MiB; epochs/steps/params 70/27,039/1,073,962.

## Errors & Dead Ends

### 2026-08-06 — Controller compared parameter tensors through dictionary equality
- Error: `RuntimeError: Boolean value of Tensor with more than one value is ambiguous` at the post-reset optimizer-group check.
- Root cause: Deep-copied `param_groups` equality invoked tensor elementwise comparison for the `params` list.
- Source: first `preflight_reset.py` invocation after the reset invariance checks.
- Do NOT retry: Compare non-parameter group scalars directly and parameter identities/order by `id`; do not change candidate code, corpus, or safety gates.

## Human Notes

> Autopilot requested; no execution-phase intervention.
