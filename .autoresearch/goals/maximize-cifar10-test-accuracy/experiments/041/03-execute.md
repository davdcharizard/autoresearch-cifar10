# EXP-041: Training-Only Direct-Path Auxiliary Cross-Entropy

## Execution

Overall Status & Info:
- **Created**: 2026-07-27
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-041
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - offline/local-only run
- **Outcome**: failed - valid normal-exposure metric miss

## Implementation Notes

### Summary

Added an explicit default-false dual-logit mode to the accepted model. The main logit call remains first and default inference returns it through the accepted single-classifier path; training opts into a second shared-classifier call on the raw pooled feature. Both the existing batch-shared mixup loss and hard-label loss now compute main/direct CE from the same inputs and targets and combine them as the exact convex objective `0.9 * main + 0.1 * direct`.

### Surprises & Discoveries

The inline training structure required duplicating only CE evaluation, not data mixing or model state. The main path already contains the raw pooled feature through its identity residual, so the new objective is a coupled gradient constraint rather than supervision of a previously disconnected representation.

### Decisions

The forward computes main logits before checking the opt-in flag, preserving accepted default kernel ordering. Dual mode calls the same classifier a second time instead of concatenating features, preventing a GEMM-shape numerical confound. The loss uses a convex 90/10 blend rather than `main + 0.1 * direct`, so nominal CE scale stays one; its 10% reduction of pooled-head data gradient relative to unchanged decay is an intentional preregistered part of the treatment.

## Experimental Adjustments

- None.

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local foreground process)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-27 04:37 UTC
- **Ended**: 2026-07-27 04:44 UTC

Description:
- Sole fixed-seed score of the always-on shared-classifier 90/10 main/direct objective against accepted 94.48%. It launches only after source/state/default-inference, loss/gradient/Nesterov, RNG/temporal, and full-body timing gates pass. Success requires best accuracy at least 94.58%; endpoint/loss and >=127-pass exposure govern interpretation but cannot rescue a primary miss.

Observations:
- Semantic preflight passed after one verifier-only tolerance correction. Default CPU/CUDA inference remained accepted; CUDA main/direct logit RMS difference was `0.14055` with 81.25% synthetic argmax agreement. Main/direct gradient cosines were `0.976-0.989`, maximum relative L2 decomposition error was `0.00113`, and all fresh/preseeded all-parameter update oracles were within `5.96e-8` for parameters and `3.73e-9` for buffers (source: semantic preflight stdout, 2026-07-27).
- Counterbalanced complete-body timing passed: retention `0.984301`, projected exposure `128.258` passes, maximum CV `0.006554`, and candidate peak `610.18 MiB` (source: timing preflight stdout, 2026-07-27).
- Launch output confirmed CUDA, 1,003,482 parameters, a 300-second budget, and 195 batches per epoch. Mixup stopped at step 16,103/195.0 seconds and RandAugment stopped after the epoch-83 iterator exhausted at step 16,185/196.0 seconds. The run produced 26 unique every-fifth plus final evaluations (source: `run.log` L1-L60).

Key Metrics:
- `best_test_acc`: `94.26%` versus `94.58%` threshold and `94.48%` baseline; final accuracy `94.26%` (source: `run.log` L62-L63).
- `final_test_loss`: `0.2529` versus accepted `0.2456` (source: `run.log` L64).
- Exposure: `25,105` steps = `128.53760` CIFAR-10 passes across 129 epochs (source: `run.log` L69-L70).
- Counted/wall/startup time: `300.0/343.5/1.2s`; peak VRAM: `1096.4 MiB`; parameters: `1,003,482` (source: `run.log` L65-L71).

## Verification Results

### Conditions Checked

- **Completion/resource contract - PASS**: exit code 0; CUDA H20; finite summary; `300.0s` counted and `343.5s` wall (<600); correct temporal transitions; 26 unique once-per-epoch evaluations; 1,003,482 parameters; 128.53760 passes (source: `run.log` L1-L71).
- **Primary metric improvement - FAIL**: best `94.26%` is `0.22` points below baseline `94.48%` and `0.32` below required `94.58%` (source: `run.log` L62).
- **Corroboration - skipped after necessary metric failure**: observed final `94.26%` and loss `0.2529` remain in Run 1 metrics but are not separately certified (source: `run.log` L63-L64).

### Informational Metrics

- Skipped under the verification procedure after the primary-metric necessary condition failed; raw values remain inline in Run 1.

## Errors & Dead Ends

### 2026-07-27 - Independent CUDA convolution-gradient tolerance too strict
- Error: `torch.testing.assert_close` failed with maximum absolute gradient difference `1.2893e-4` across three cloned FP32 backward graphs before timing.
- Root cause: Algebraically equivalent separate main/direct/combined CUDA convolution backward reductions round in different orders; source, logits, and production objective had already passed their exact checks.
- Source: semantic preflight traceback in `gradient_fixture`, before timing/scoring.
- Do NOT retry: print all gradient errors before assertions and use the measured `rtol=3e-3, atol=2e-4` only for this independent FP32 loss-gradient decomposition; retain exact source/graph and per-parameter Nesterov oracles.

## Human Notes

> User requested uninterrupted autopilot and offline/local-only execution.
