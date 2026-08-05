# EXP-015: Per-Example Mixup Strengths

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-015
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - local-only run
- **Outcome**: failed - valid run scored below the accuracy threshold

## Implementation Notes

### Summary

Changed the accepted batch-shared mixup coefficient into a length-`B` vector of independent `Beta(0.2, 0.2)` draws, broadcasting only for image interpolation. Added a stateless production `mixup_loss` helper for per-example paired cross-entropy and made both scored training and preflight call the same helper. No model, schedule, optimizer, augmentation, cutoff, seed, or evaluation behavior was changed.

### Surprises & Discoveries

`train.py` constructs its evaluator at module import, so an import-safe preflight cannot use a dummy that raises during construction. The evaluator guard instead permits the one dummy construction and raises on any evaluation call; it imports no dataset and cannot expose test metrics.

### Decisions

Kept ordinary random permutation rather than adding derangement, preserving single-variable attribution. The preflight harness lives under the ignored experiment directory and imports production mixing/loss helpers; only the accepted scalar reference is reimplemented locally.

## Experimental Adjustments

- **Clarified evaluator isolation**: allow module-level dummy construction but fail on `evaluate`, because production import constructs `Eval` eagerly. This preserves the no-test-data intent without modifying production initialization. (ref: implementation preflight design)

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 60326
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed (research no-improvement)
- **Started**: 2026-07-26 16:13:26 UTC
- **Ended**: 2026-07-26 16:19:19 UTC

Description:
- One fixed-seed local H20 run tests whether independent per-example alpha-0.2 mixup coefficients improve CIFAR-10 top-1 accuracy while retaining the accepted 65% mixup window and hard-label tail. Expected throughput is at least 95% of the batch-scalar path. Success requires a valid complete run and `best_test_acc >= 94.17%`.

Observations:
- Evaluator-free semantics passed after aligning the harness's cuDNN settings with production: beta mean 0.508997, variance 0.178111, and 253 unique values in the first 256-sample batch. Production `mixup_batch`/`mixup_loss`, constant-vector equivalence, and hard-label state equivalence all passed. (source: semantic preflight stdout)
- Scope and syntax gates passed: the only production diff is `train.py`, no non-ignored untracked files or root importable Python extras exist, and compilation/diff checks exit 0. (source: pre-score audit commands)
- Matched H20 throughput passed: accepted median 10.818908 ms, candidate median 10.775510 ms, retention 1.004027, projected 142.471489 passes, accepted CV 0.003892, candidate CV 0.005127. (source: throughput preflight stdout)
- Log capture was active immediately; the run reported CUDA, 691,674 parameters, 195 batches per epoch, finite loss 1.2382 at step 450, and approximately 11 ms steps. (source: `run.log` early output)
- Mixup disabled exactly once at epoch 92, step 17,745, 195.0 counted seconds and LR 0.0612. The hard-label tail completed without error. (source: `run.log` L42)
- The process exited 0 with one complete summary, 29 distinct evaluation epochs, no traceback/non-finite/OOM signature, 300.0 counted seconds, and 340.3 total seconds. (source: `run.log` L6-L75 and verification audit)

Key Metrics:
- best_test_acc: 93.79% at epoch 135, 0.28 points below the 94.07% accepted baseline and 0.38 below the 94.17% threshold. (source: `run.log` L60, L66)
- final_test_acc: 93.62% at epoch 143. (source: `run.log` L64, L67)
- final_test_loss: 0.2628 at epoch 143. (source: `run.log` L64, L68)
- exposure: 27,737 steps = 142.01344 dataset-equivalent passes. (source: `run.log` L74)
- timing: 300.0 training seconds, 340.3 total seconds, 1.0 startup seconds. (source: `run.log` L69-L71)
- resources/model: 1,094.0 MiB peak VRAM and 691,674 trainable parameters. (source: `run.log` L72, L75)

## Verification Results

### Conditions Checked

- **Run completion and integrity**: PASS - exit 0; 300.0 counted seconds; 340.3 total seconds under the 600-second limit; one H20; one transition at 195.0 seconds; 29 evaluations with no duplicate epoch; no error signatures. (source: `run.log` L6-L75 and local command exit)
- **Primary metric >= 94.17%**: FAIL - `best_test_acc=93.79%`, below the 94.07% baseline and 94.17% necessary threshold. Verification stopped on this necessary-condition failure. (source: `run.log` L66; results-index baseline query)
- **Remaining conditions**: skipped - aborted after primary metric failure; the production diff/scope had already passed the mandatory pre-score audit.

### Informational Metrics

- Skipped under the verification guard because the primary metric condition failed. Values remain preserved under Run 1 Key Metrics for analysis.

## Errors & Dead Ends

### 2026-07-26 - Preflight matched-initialization check failed
- Error: `AssertionError: conv1.weight` while comparing two seed-reset model states.
- Root cause: the first traceback line was initially attributed to initialization, but line-number inspection showed the initial states matched and the failure occurred after sequential hard-label updates; the harness had omitted production cuDNN determinism settings. CUDA initialization was also moved before matched construction to remove that possible confound.
- Source: semantic preflight traceback before any scored run.
- Do NOT retry: do not compare matched CUDA convolution updates without first enabling the production `cudnn.benchmark=True` and `cudnn.deterministic=True` settings.

### 2026-07-26 - Hard-step equivalence was nondeterministic
- Error: `AssertionError: conv1.weight` at the post-update state comparison after initialization equality had passed.
- Root cause: semantic preflight did not set the production cuDNN determinism flags before sequential matched model updates.
- Source: semantic preflight traceback at `preflight.py` post-update `assert_state_equal`, before any scored run.
- Do NOT retry: keep backend determinism aligned with production for bitwise hard-path equivalence checks.

## Human Notes

> Autopilot local-only execution; no intervention requested.
