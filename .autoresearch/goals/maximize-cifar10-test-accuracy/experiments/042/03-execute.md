# EXP-042: Exact-Neutral Centered Content-Attention Pooling

## Execution

Overall Status & Info:
- **Created**: 2026-07-27
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-042
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - offline/local-only run
- **Outcome**: failed - valid normal-exposure metric miss

## Implementation Notes

### Summary

Appended one zero-initialized bias-free 1x1 scorer after all accepted model/head initialization. The final forward preserves accepted global average pooling, then adds a centered temperature-one softmax weighted correction over the same final 8x8 feature map before the exact pooled residual MLP and classifier. The scorer adds 128 parameters; training loop, loss, optimizer construction, data, timing controls, and evaluator remain untouched.

### Surprises & Discoveries

The accepted final spatial extent is exactly 8x8, making uniform `1/64` exactly representable in FP32. Expressing content attention as an accepted GAP plus a centered correction preserves the original reduction bytes at zero query, unlike a direct weighted sum whose reduction order could differ.

### Decisions

The scorer constructor is isolated in a restoring CPU RNG fork and overwritten with exact zeros, so no new seed or retained random value exists. It stays in the generic matrix-decay optimizer group; decay is exactly zero on its initial state and becomes ordinary after the first data-gradient update. No temperature, residual scale, bias, or positional state is added.

## Experimental Adjustments

- None.

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local foreground process)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-27 05:17 UTC
- **Ended**: 2026-07-27 05:24 UTC

Description:
- Sole fixed-seed score of exact-neutral single-query content-attention pooling against accepted 94.48%. It launches only after source/state/identity, analytic-gradient/Nesterov, RNG/temporal, and full-body timing gates pass. Success requires best accuracy at least 94.58%; endpoint/loss and >=127-pass exposure govern interpretation but cannot rescue a primary miss.

Observations:
- Semantic preflight passed directly. CPU/CUDA zero-query attention mass was exact, centered correction was bitwise zero, accepted logits/common gradients were exact, scorer gradient norms were `0.2530/0.2458` in mixup/hard fixtures, covariance/mean-CE oracle errors were <=`2.22e-16`, and all update errors were <=`1.19e-7`. Fresh-step attention remained broad at `63.77/63.997` effective sites (source: semantic preflight stdout, 2026-07-27).
- Counterbalanced complete-body timing passed: retention `0.980831`, projected exposure `127.806` passes, maximum CV `0.006587`, and candidate peak `614.92 MiB` (source: timing preflight stdout, 2026-07-27).
- Launch confirmed CUDA, 1,003,610 parameters, 300 seconds, and 195 batches/epoch. Mixup stopped at step 15,982/195.0s and RandAugment after the epoch-82 iterator exhausted at step 15,990/195.1s. The run produced 26 unique every-fifth plus final evaluations (source: `run.log` L1-L60).

Key Metrics:
- `best_test_acc`: `93.80%` versus `94.58%` threshold and `94.48%` baseline; final `93.80%` (source: `run.log` L62-L63).
- `final_test_loss`: `0.2787` versus accepted `0.2456` (source: `run.log` L64).
- Exposure: `24,987` steps = `127.93344` passes across 129 epochs (source: `run.log` L69-L70).
- Counted/wall/startup: `300.0/346.1/1.1s`; peak VRAM `1096.4 MiB`; parameters `1,003,610` (source: `run.log` L65-L71).

## Verification Results

### Conditions Checked

- **Completion/resource contract - PASS**: exit 0; CUDA H20; finite summary; `300.0s` counted, `346.1s` wall; correct transitions; 26 unique evaluations; 1,003,610 params; 127.93344 passes (source: `run.log` L1-L71).
- **Primary metric improvement - FAIL**: best `93.80%` is `0.68` below baseline and `0.78` below required `94.58%` (source: `run.log` L62).
- **Corroboration - skipped after necessary metric failure**: final `93.80%` and loss `0.2787` remain recorded but are not separately certified (source: `run.log` L63-L64).

### Informational Metrics

- Skipped under the verification procedure after primary failure; raw values remain inline above.

## Errors & Dead Ends

- None.

## Human Notes

> User requested uninterrupted autopilot and offline/local-only execution.
