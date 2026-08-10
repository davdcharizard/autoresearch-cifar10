# EXP-001: Budget-Aligned Cosine SGD with Nesterov

## Execution

Overall Status & Info:
- **Created**: 2026-08-05
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-001
- **Commit**: (pending — committed on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Implemented the reviewed training policy only in `train.py`: a 15%-hold followed by elapsed-time cosine decay to `1e-4`, Nesterov SGD, persistent training-loader workers, and six budget checkpoints plus a terminal evaluation. Removed the unreachable fixed-step scheduler while preserving model, batch size, transforms, hard-label loss, seed, weight decay, maximum steps, evaluator, and fixed 300-second training budget. Static compilation, Ruff, pre-commit, diff, and tracked-scope checks pass.

### Surprises & Discoveries

The external plan review correctly challenged the unmeasured timeout margin. Local diagnostics found non-persistent training-loader epochs around 19 seconds and repeated fixed-evaluator passes around 17.3 seconds. Enabling persistent training workers reduced the second loader epoch to 1.025 seconds; the fixed evaluator remains untouched, so evaluation count was capped at seven.

### Decisions

Replaced the initial sparse-then-dense validation proposal with checkpoints at 20/40/60/70/80/90% plus termination. This covers the plan critic's 60%-80% blind spot while bounding measured evaluation overhead to roughly 121 seconds. Added `persistent_workers=True` as a measured runtime control; it preserves the augmentation distribution and fixed seed but changes the exact worker RNG stream. Kept Nesterov as selected, with a standard-momentum schedule run pre-registered as the next discriminator if this experiment fails.

After the `no-improvement` verdict, reverted only the known `train.py` diff and returned to the integration branch. Skipped broad `git clean -fd` because it would delete the required untracked CIFAR-10 cache; removed `run.log` directly after all evidence was persisted.

## Experimental Adjustments

- **Persist training-loader workers and cap evaluation at seven passes**: Diagnostics measured later loader epochs at 1.025s with persistence versus 18.975s without it, and evaluator passes at 17.271s. This addresses a concrete timeout risk without changing the fixed evaluator. (ref: planning diagnostics and `02-plan-review.md` concern 1)

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 2097486 (timeout supervisor PID 2097482)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-05 07:14:56 UTC
- **Ended**: 2026-08-05 07:21:14 UTC

Description:
- Run the reviewed time-aligned schedule on one NVIDIA H20 with all output redirected and a 600-second hard cutoff. The independent variable is the 15%-hold/cosine optimization policy with Nesterov; persistent workers and checkpointed evaluation protect the required wall limit. Expected outcome is `best_test_acc >= 91.77%`, a complete summary, approximately 300 counted training seconds, and seven unique evaluated epochs.

Observations:

- The local process exited `0` before the 600-second supervisor timeout and produced a primary metric, so execution reached verification. (source: local PID 2097486; `run.log` L20)
- Necessary Condition 1 failed: `best_test_acc=91.57%`, which is 0.10 percentage points below the `91.67%` baseline and 0.20 points below the required `91.77%`. Verification stopped immediately as required. (source: `run.log` L20)

Key Metrics:

- best_test_acc: 91.57% (source: `run.log` L20)

## Verification Results

### Conditions Checked

- **Accuracy improvement**: FAIL — `best_test_acc=91.57%`, required `>=91.77%` against baseline `91.67%`. (source: `run.log` L20)
- **Valid numeric summary**: skipped — verification aborted after Necessary Condition 1 failed.
- **Fixed training budget and total runtime**: skipped — verification aborted after Necessary Condition 1 failed.
- **Unique seven-epoch validation schedule**: skipped — verification aborted after Necessary Condition 1 failed.

### Informational Metrics

- Skipped because a necessary condition failed.

## Errors & Dead Ends

## Human Notes

> The user restored the Claude gateway account and explicitly required external adversarial reviews with no fallback. Both the idea and plan reviews then completed successfully through Claude Code.
