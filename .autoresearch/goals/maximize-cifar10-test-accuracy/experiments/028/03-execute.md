# EXP-028: Freeze the High-Resolution Prefix for the Hard Tail

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-028
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - offline/local-only session
- **Outcome**: failed

## Implementation Notes

### Summary

Added a strict one-way prefix-freeze helper and extracted the post-iterator early-phase transition into a directly testable controller. The accepted model trains unchanged through the complete early mixup/RandAugment interval; only after the first eligible exhausted iterator does the controller disable RandAugment and freeze all stem/stage-1 parameters while leaving their forward path and BN buffers live.

### Surprises & Discoveries

The optimizer can safely retain frozen parameters and their momentum buffers: SGD skips parameters whose gradients are `None`, so no group reconstruction or state deletion is needed. Explicit iterator state was added because `budget_exhausted` alone distinguished the scored partial epoch but could not serve as an executable oracle for normal exhaustion.

### Decisions

The production helper validates parameter identities/counts and optimizer membership but does not clone weights or state; the ignored preflight owns those expensive equality checks. The verifier loads accepted source independently from `git show 67c8e98:train.py`, and exposure projection uses EXP-027's observed 16,770-step boundary plus 9,208-step post-boundary tail rather than uniform whole-run weighting.

## Experimental Adjustments

- None after plan review.

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 1337978 (timeout PID 1337977)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed (exit 0; verification failed)
- **Started**: 2026-07-26 21:59:15 UTC
- **Ended**: 2026-07-26 22:05:23 UTC

Description:
- One offline local H20 score of accepted EXP-027 with the stem and stage 1 frozen only for the post-boundary hard tail. It tests whether late prefix gradients are less valuable than additional upper-layer decisions while preserving the full inference representation. It launches only after independent accepted-oracle semantics and a boundary-calibrated 145-pass timing gate pass.

Observations:
- Semantic preflight passed against independent `git show 67c8e98:train.py`: exact accepted pre-boundary state/optimizer/RNG, real controller branches, 33,424/953,674 counts, fixed frozen values/momentum, live prefix BN buffers, and finite upper updates.
- Balanced hard-tail timing passed: accepted/frozen medians 11.213227/7.191277 ms, speed ratio 1.559282, projected 31,127.866 steps / 159.374673 passes; CVs 0.000414/0.003249.
- Final audit confirmed baseline 94.32 at `67c8e98`, one idle NVIDIA H20, local CIFAR-10, frozen `prepare.py`, clean syntax/diff, no stale log, and only tracked `train.py` modified.
- The sole score completed cleanly. Mixup stopped at step 16,551/195.0 s; RandAugment and prefix freezing occurred together after epoch 85 exhausted at step 16,575/195.3 s (lag 24 steps) with exact 33,424/953,674 counts.
- Frozen-tail throughput rose from about 22k to 35.5k images/s. All 32 evaluations were unique at epochs 5 through 160 in increments of five; no numerical, CUDA, worker, or source-integrity error appeared.

Key Metrics:
- `best_test_acc=93.99%`; `final_test_acc=93.92%`; `final_test_loss=0.2804`.
- `training_seconds=300.0`; `total_seconds=348.4`; `startup_seconds=1.1`.
- `num_epochs=160`; `num_steps=31,074`; `data_passes=159.09888`; `num_params=987,098`; `peak_vram_mb=1096.3`.

## Verification Results

### Conditions Checked

- PASS: one H20, local data, frozen evaluator/`prepare.py`, and only `train.py` changed in tracked production scope.
- PASS: independent accepted oracle, real controller/freeze semantics, fixed frozen state, live BN buffers, and timing projection 159.374673 passes.
- PASS: sole score exit 0, 300.0 counted / 348.4 total seconds, 987,098 original parameters, 159.09888 realized passes, exact transitions, and 32 unique accepted-cadence evaluations.
- FAIL: `best_test_acc=93.99% <94.42%`; verification stopped at the first failed accuracy condition.
- SKIPPED after prior failure: `final_test_acc >=94.42%`; observed final was 93.92%.

### Informational Metrics

- Not formally collected after the necessary-condition failure. Observed final loss 0.2804 was 0.0281 worse than accepted 0.2523; best-final gap was 0.07 points.

## Errors & Dead Ends

### 2026-07-26 - Semantic harness omitted production cuDNN determinism
- Error: `AssertionError: pre_boundary_model.conv1.weight` after equal accepted/candidate logits, loss, and RNG.
- Root cause: sequential convolution backward calls were compared without enabling the production `torch.backends.cudnn.deterministic=True` setting in the verifier.
- Source: first EXP-028 semantic preflight traceback at `preflight.py:181`.
- Do NOT retry: do not compare independent CUDA training updates without reproducing the scored cuDNN benchmark/deterministic flags.

## Human Notes

> Autopilot run; no intervention requested.
