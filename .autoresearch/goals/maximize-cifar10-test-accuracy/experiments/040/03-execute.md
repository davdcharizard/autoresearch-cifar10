# EXP-040: Equalize Effective Classifier Row Norms

## Execution

Overall Status & Info:
- **Created**: 2026-07-27
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-040
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - offline/local-only run
- **Outcome**: failed - valid normal-exposure metric miss

## Implementation Notes

### Summary

Replaced only the accepted final affine call with a differentiable effective weight whose ten rows share the raw RMS row norm while preserving raw classifier Frobenius norm at each state. Raw `fc.weight`/bias parameters, initialized bytes, state keys, optimizer membership, accepted decay coefficient, pooled representation, and all training/evaluation controls remain unchanged.

### Surprises & Discoveries

The accepted seed-42 classifier begins with a 6.96% population CV in row norms and a 1.2725 max/min ratio, so the map is a measurable rather than cosmetic intervention. Its instantaneous Frobenius invariant does not imply an accepted training trajectory because tangential conditioning and shared radial gradients change.

### Decisions

Production deliberately has no epsilon or clamp: fixed semantic/update fixtures must show row norms safely nonzero, and a future scored collapse is handled by the existing finite-loss failure path. The analytic gradient oracle uses a fixed upstream tensor independent of raw weight; only the instantaneous coupled-decay contribution is treated as radial because historical Nesterov state can be non-radial.

## Experimental Adjustments

- None.

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local foreground process)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-27 04:01 UTC
- **Ended**: 2026-07-27 04:07 UTC

Description:
- Sole fixed-seed score of the Frobenius-preserving equal-row-norm classifier against accepted 94.48%. It launches only after independent geometry, analytic gradient, transformed-forward, Nesterov, RNG, cadence, and full-body timing gates pass. Success requires best accuracy at least 94.58%; endpoint/loss and >=127-pass exposure govern interpretation but cannot rescue a primary miss.

Observations:
- Semantic preflight passed after two verifier-only corrections: initial row CV `0.0696447`, raw/effective Frobenius `4.5009542`, effective row norm `1.4233266`, relative weight delta `0.0695184`, logit delta RMS `0.04551`, five of 32 synthetic argmax changes, and analytic-gradient maximum error `4.44e-16`; fresh/preseeded update oracles passed (source: semantic preflight stdout, 2026-07-27).
- Counterbalanced timing passed narrowly: retention `0.978220`, projected exposure `127.466` passes, maximum CV `0.006957`, and candidate peak `610.16 MiB` (source: timing preflight stdout, 2026-07-27).
- Launch output confirmed CUDA, 1,003,482 parameters, a 300-second budget, and 195 batches per epoch (source: `run.log` L1-L4).
- Mixup stopped at step 15,947 and 195.0 seconds; RandAugment stopped only after the epoch-82 iterator exhausted at step 15,990 and 195.5 seconds. The run produced 26 unique every-fifth plus final evaluations (source: `run.log` L5-L60).

Key Metrics:
- `best_test_acc`: `93.91%` versus `94.58%` threshold and `94.48%` baseline; final accuracy `93.87%` (source: `run.log` L62-L63).
- `final_test_loss`: `0.2622` versus accepted `0.2456` (source: `run.log` L64).
- Exposure: `24,910` steps = `127.53920` CIFAR-10 passes across 128 epochs (source: `run.log` L69-L70).
- Counted/wall time: `300.0/344.7s`; peak VRAM: `1096.4 MiB`; parameters: `1,003,482` (source: `run.log` L65-L71).

## Verification Results

### Conditions Checked

- **Completion/resource contract - PASS**: exit code 0; CUDA H20; finite summary; `300.0s` counted and `344.7s` wall (<600); correct temporal transitions; 26 unique once-per-epoch evaluations; 1,003,482 parameters; 127.53920 passes (source: `run.log` L1-L71).
- **Primary metric improvement - FAIL**: best `93.91%` is `0.57` points below baseline `94.48%` and `0.67` below required `94.58%` (source: `run.log` L62).
- **Corroboration - skipped after necessary metric failure**: observed final `93.87%` and loss `0.2622` remain in Run 1 metrics but are not separately certified (source: `run.log` L63-L64).

### Informational Metrics

- Skipped under the verification procedure after the primary-metric necessary condition failed; raw values remain inline in Run 1.

## Errors & Dead Ends

### 2026-07-27 - Scalar tensor passed to `full_like`
- Error: `TypeError: full_like(): argument 'fill_value' must be Number, not Tensor` before timing.
- Root cause: The disposable invariant assertion used the tensor-valued RMS as a fill scalar; the measured production geometry was correct.
- Source: semantic preflight traceback in `geometry_checks`, before timing/scoring.
- Do NOT retry: compare row norms to `rms.expand_as(effective_rows)` without changing production geometry.

### 2026-07-27 - Independent FP32 norm ordering exceeded generic tolerance
- Error: independent full logits differed in 4/320 elements, maximum absolute `1.9558e-7`, versus `atol=1e-7` before timing.
- Root cause: Production `torch.linalg.vector_norm` and independent square/sum reductions are algebraically equal but round FP32 in a different order.
- Source: semantic preflight independent full-logit check, before timing/scoring.
- Do NOT retry: use the measured `rtol=2e-5, atol=2e-7` only for this independent full-logit comparison; keep geometry/float64 checks strict.

## Human Notes

> User requested uninterrupted autopilot and offline/local-only execution.
