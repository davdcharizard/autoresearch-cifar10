# EXP-031: End-to-End FP32 Channels-Last Training

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-031
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - local-only run
- **Outcome**: failed - preregistered semantic bound exceeded

## Implementation Notes

### Summary

Converted the accepted WRN to channels-last after direct seed-42 construction, requested channels-last during every pinned input H2D transfer, and added an idempotent conversion at `WideResNet.forward` entry for the frozen evaluator's contiguous NCHW batches. The sole production audit is derived at step zero from the actual conv weight and post-mixup tensor layout, strides, and dtypes.

### Surprises & Discoveries

The first proposal's simple startup label could have claimed channels-last without observing the scored tensor path. Plan review also identified the evaluator's final 16-example batch and the need to separate logical initialization equality, bounded cross-layout numerical differences, and exact candidate-layout replay.

### Decisions

Timing uses three paired replicate speedups with reciprocal time-phase rates; every pair must exceed 1.02 rather than allowing a 2% claim under larger window noise. Candidate replay snapshots the complete post-warm model/optimizer/RNG/mode state without changing cuDNN caches or flags between replays.

## Experimental Adjustments

- **Strengthened material-speed evidence**: Every paired reciprocal-rate replicate, not only a ratio of medians, must clear 1.02; arm CV is capped at 2%. (ref: `02-plan-review.md` concern 1)
- **Made evaluator/layout evidence executable**: Added batch-16 semantic coverage and runtime tensor-derived scored audit. (ref: `02-plan-review.md` concerns 2-3)
- **Localized numerical gates**: Per-named-tensor gradient/update/BN bounds replace aggregate vector and argmax requirements. (ref: `02-plan-review.md` concerns 4-5)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A - score not launched
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: not submitted - semantic abort
- **Started**: N/A
- **Ended**: 2026-07-26 23:31:19 UTC

Description:
- This will be the sole fixed-seed score only if end-to-end channels-last passes semantic isolation and demonstrates at least 1.02x complete-body speed in every balanced pair. The treatment preserves FP32 storage, full gradients, accepted hyperparameters/data decisions, and evaluator logic while changing legal cuDNN layout kernels. The primary threshold is 94.42%; final accuracy and loss are mechanism corroboration only.

Observations:
- Static audit passed on the isolated branch: `train.py` is the only tracked modification, `prepare.py` is unchanged, compilation succeeds, local CIFAR-10 is present, and one idle H20 is available. (source: preflight setup output)
- The semantic harness passed construction, logical state, layout, input/mixup, and candidate evaluator-path checks up to the independent accepted-logit comparison. At batch 256, NHWC versus accepted NCHW exceeded the fixed `rtol=2e-4, atol=2e-5` bound, so timing and scoring were prohibited. (source: semantic preflight traceback at `preflight.py:313`)

Key Metrics:
- accepted/candidate eval mismatches: 1,326 / 2,560 logits (51.8%) outside tolerance (source: semantic preflight traceback)
- maximum absolute logit difference: 0.0008890629 versus allowed absolute tolerance 0.00002 (source: semantic preflight traceback)
- maximum relative logit difference: 0.0555356 versus allowed relative tolerance 0.0002 (source: semantic preflight traceback)
- scored runs: 0; throughput runs: 0; `run.log`: not created

## Verification Results

### Conditions Checked

- **Static scope/environment - PASS**: one idle H20, local data, accepted baseline 94.32, only `train.py` tracked, frozen `prepare.py`, and successful compilation. (source: setup commands)
- **Semantic cross-layout bound - FAIL**: independent accepted/candidate logits exceeded the fixed tolerance at batch 256, with max absolute difference 0.0008890629 and 51.8% mismatched elements. (source: semantic preflight traceback at `preflight.py:313`)
- **Candidate replay and per-tensor update bounds - SKIPPED**: aborted at the earlier fixed cross-layout condition.
- **Material throughput - SKIPPED**: semantic qualification is mandatory before timing.
- **Primary objective - SKIPPED**: the sole score was not launched.

### Informational Metrics

- No score or throughput metrics were produced; the fixed semantic bound failed before those phases.

## Errors & Dead Ends

### 2026-07-26 - Channels-last exceeds fixed cross-layout numerical bound
- Error: `Tensor-likes are not close` with max absolute logit difference 0.0008890629 and max relative difference 0.0555356 at batch 256.
- Root cause: Legal deterministic NHWC cuDNN convolution accumulation diverges from the accepted NCHW path beyond the preregistered semantic tolerance, despite logical initialization/layout compatibility.
- Source: semantic preflight, `preflight.py:313`
- Do NOT retry: Do not loosen the fixed tolerance or rescue with alternate layout placement, cuDNN/TF32 flags, precision, compilation, fusion, batch, or LR changes.

## Human Notes

> Autopilot local-only execution; no user intervention requested.
