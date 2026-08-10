# EXP-006: Shared-Budget CutMix and Manifold Mixup

## Execution

Overall Status & Info:
- **Created**: 2026-08-05
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-006
- **Base Node**: 004
- **Commit**: `56c3ce3`
- **Outcome**: failed - valid run missed the required metric threshold

## Implementation Notes

### Summary

Implemented the approved hybrid policy in `train.py` only. The parent seed-42 gate and full selected-batch CutMix specification are consumed before private routing, seed 43 assigns 75% of selected batches to unchanged CutMix and 25% to block-2/block-4 manifold mixing, and seed 44 supplies exact four-uniform `Beta(2,2)` coefficients and manifold permutations. `PreActWideResNet.forward` now accepts optional hidden-mix arguments, applies one out-of-place interpolation at the chosen boundary, and restores channels-last layout. Final audit lines expose routing, boundaries, discarded specs, both lambda moments, and the unchanged SAM schedule.

### Surprises & Discoveries

- The preregistered boundary counts `6329/6299` specifically require a seed-43 `randint(2)` boundary draw; using a second thresholded uniform would produce `6318/6310`. The implementation uses the former, matching the registered simulator and plan.
- Hidden advanced indexing does not guarantee channels-last strides, so the mixed activation is explicitly made channels-last contiguous before entering the next residual block.
- GPU smoke measurement found only 1.0431x median manifold-step latency versus clean, leaving the 24,000-step feasibility target plausible.

### Decisions

- Kept the inherited `cutmix_batch` fixed-spec interface and added `sample_cutmix_spec`; this lets every selected hybrid batch advance the parent streams before routing while preserving exact inherited patch behavior.
- Kept all policy and lambda statistics as CPU scalars to avoid charged-loop device synchronization. Only the private manifold permutation is generated on CUDA.
- The actual parent source from commit `1a8d0de` was executed in a non-main namespace for parity testing, as required by Claude's adversarial review.

## Experimental Adjustments

- None. The implementation and fixed simulation values match the Claude-reviewed preregistration.

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 474337 (timeout wrapper; execution session 47373)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/tree-v0-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-05 14:13:18 UTC
- **Ended**: 2026-08-05 14:20:54 UTC

Description:
- One fixed-seed, physical-GPU-0 run of the Claude-reviewed shared-budget CutMix/manifold policy. The run retains EXP-004's total early mixing gate and clean-tail periodic SAM while replacing 25% of selected CutMix batches with one-pass hidden interpolation. Expected `best_test_acc` is 95.55-95.80%, with acceptance at the Decimal-derived 95.50% threshold. Output is captured completely in `run.log`; intermediate accuracy cannot change execution.

Observations:
- Pre-run static checks passed: `py_compile`, Ruff, and `git diff --check` all exited 0; only tracked `train.py` differs from parent `1a8d0de`.
- Exact policy replay passed with selected/CutMix/manifold/boundary2/boundary4 counts `49769/37141/12628/6329/6299`; lambda mean `0.500106625` and mean minimum lambda `0.312155088` matched preregistration.
- Actual-parent CutMix parity passed for 256 shared steps and 129 selected batches, including transformed outputs and bitwise CPU/CUDA generator states; global RNG isolation passed.
- Forward semantics passed for default parity, boundaries 2 and 4, channels-last layout, two-source gradients, invalid arguments, and CutMix clipped/zero-area regression.
- GPU-0 BF16 integration passed for clean, CutMix, both manifold boundaries, and SAM. Drop-path consumed exactly six draws, SAM perturbation norm was `0.050000`, BatchNorm updated once, and manifold/clean median latency ratio was `1.0431`.
- Run 1 initialized successfully on CUDA with 2,748,890 parameters, the exact registered hybrid-policy config, a 300-second budget, and 195 batches per epoch (source: `run.log` startup lines).
- Run 1 exited 0 with no traceback, OOM, nonfinite value, hidden-mix assertion, or mix/SAM overlap. The early policy stopped exactly at progress 0.75 and late mixing remained zero (source: `run.log` lines 270-273).
- All protocol checks passed before the primary metric was read. The single metric read returned `best_test_acc=95.41%`, below the preregistered `95.50%` threshold, so the valid run is final and no retry is allowed (source: `run.log` line 275).

Key Metrics:
- `best_test_acc`: 95.41%; `final_test_acc`: 95.41%; `final_test_loss`: 0.1749 (source: `run.log` lines 275-277).
- Runtime: 300.0 charged seconds, 455.8 total seconds, 1.2 startup seconds; peak VRAM 1,190.5 MiB (source: `run.log` lines 278-281).
- Work: 132 epochs/evaluations, 25,644 optimizer steps, 2,748,890 parameters (source: `run.log` lines 269, 282-284; evaluator count checked programmatically as 132).
- Policy: 10,257 selected and 10,412 clean of 20,669 eligible; 7,696 CutMix and 2,561 manifold batches; boundaries 2/4 were 1,324/1,237; lambda mean 0.494529 and mean minimum lambda 0.308301 (source: `run.log` lines 270-272).
- SAM: 2,488 of 4,975 eligible batches, ratio 0.5001, first step 20,670 at progress 0.7500 (source: `run.log` line 273).

## Verification Results

### Conditions Checked

- Pre-launch parent and hardware: pass. Node 004 is extendable with metric `95.40` and commit `1a8d0de`; threshold is exactly `95.50`; physical GPU 0 is NVIDIA H20 with `97871 MiB`.
- Pre-launch implementation verification: pass, with detailed evidence in Run 1 observations.
- Runtime/protocol integrity: pass. Exit 0; exact registered configuration; policy frequencies, lambda moments, cutoff, and SAM cadence all within bounds; 300.0 charged seconds and 455.8 total seconds; 25,644 steps; 132 evaluations for 132 epochs; unchanged parameter count; complete unique summary (source: `run.log` lines 270-284 and protocol parser output).
- Necessary metric improvement: fail. Parent `95.40%` plus `0.10` requires `95.50%`; observed `95.41%`, a `+0.01`-point delta (source: tree node 004 and `run.log` line 275).

### Informational Metrics

- Final test accuracy/loss: 95.41% / 0.1749, versus parent 95.40% / 0.1654.
- Steps: 25,644 versus parent 25,560; total runtime: 455.8s versus parent 457.3s; peak VRAM: 1,190.5 MiB versus parent 1,190.5 MiB.

## Errors & Dead Ends

- None.

## Human Notes

> User requires physical GPU 0 and Claude adversarial review with no fallback. Claude reviewed both the experiment idea and executable plan successfully.
