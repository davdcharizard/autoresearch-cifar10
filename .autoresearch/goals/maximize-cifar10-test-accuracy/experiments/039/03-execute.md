# EXP-039: Rephase Cosine Across the Hard-Label Tail

## Execution

Overall Status & Info:
- **Created**: 2026-07-27
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-039
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - offline/local-only run
- **Outcome**: failed - valid normal-exposure metric miss

## Implementation Notes

### Summary

Changed only `learning_rate()` so its returned values remain accepted before the 65% mixup boundary, then use the accepted boundary LR as the start of a second cosine across the remaining 35% to the accepted 0.002 endpoint. The implementation derives every anchor from existing constants and does not add momentum state, a new hyperparameter, model work, or runtime transition logic.

### Surprises & Discoveries

The accepted schedule function admits a direct isolated replacement. The first real hard-label step will normally occur slightly after 65%, so its LR must be checked against the formula at observed pre-step progress rather than against the exact synthetic 65% anchor.

### Decisions

The preflight uses 50% progress for complete mixup timing and 75% for hard-tail timing, ensuring the latter executes the changed schedule. It independently verifies coupled Nesterov updates because larger LR scales both data-gradient and weight-decay contributions; no unique causal attribution is claimed.

## Experimental Adjustments

- None.

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local foreground process)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-27 03:29 UTC
- **Ended**: 2026-07-27 03:35 UTC

Description:
- Sole fixed-seed scored run of the continuous 65%-anchored tail cosine rephase against the accepted 94.48% baseline. It launches only if source, schedule, update, RNG, cadence, and full-body timing gates pass. Success requires best accuracy at least 94.58%; endpoint/loss and >=127-pass exposure govern interpretation but cannot rescue a primary miss.

Observations:
- Semantic preflight passed after two verifier-only corrections: exact returned-value equality through 65%, boundary gap `2.998e-13`, post-warmup monotonicity, tail areas `0.00793445/0.01106563` (ratio `1.3946300912`), unchanged model/RNG/momentum, and independent fresh/preseeded Nesterov oracles (source: semantic preflight stdout, 2026-07-27).
- Counterbalanced timing passed at explicit 50%/75% schedule points: retention `0.999378`, projected exposure `130.223` passes, maximum CV `0.002810`, and candidate peak `610.16 MiB` (source: timing preflight stdout, 2026-07-27).
- Launch output confirmed CUDA, 1,003,482 parameters, a 300-second budget, and 195 batches per epoch (source: `run.log` L1-L4).
- Mixup stopped at step 16,494 and 195.0 seconds with LR 0.0612; RandAugment stopped only after the epoch-85 iterator exhausted at step 16,575 and 195.9 seconds (source: `run.log` L42-L43).
- Logged post-boundary LRs follow the candidate curve at rounded progress, including 0.0444 at 77.5%, 0.0103 at 91.5%, and 0.0020 near 100%; 27 unique evaluations occurred every fifth epoch plus final epoch 132 (source: `run.log` L44-L62).

Key Metrics:
- `best_test_acc`: `93.98%` versus `94.58%` threshold and `94.48%` baseline; final accuracy also `93.98%` (source: `run.log` L64-L65).
- `final_test_loss`: `0.2661` versus accepted `0.2456` (source: `run.log` L66).
- Exposure: `25,628` steps = `131.21536` CIFAR-10 passes across 132 epochs (source: `run.log` L71-L72).
- Counted/wall time: `300.0/341.9s`; peak VRAM: `1096.4 MiB`; parameters: `1,003,482` (source: `run.log` L67-L73).

## Verification Results

### Conditions Checked

- **Completion/resource contract - PASS**: exit code 0; CUDA H20; finite summary; `300.0s` counted and `341.9s` wall (<600); correct temporal transitions; 27 unique once-per-epoch evaluations; 1,003,482 parameters; 131.21536 passes (source: `run.log` L1-L73).
- **Primary metric improvement - FAIL**: best `93.98%` is `0.50` points below baseline `94.48%` and `0.60` below required `94.58%` (source: `run.log` L64).
- **Corroboration - skipped after necessary metric failure**: observed final `93.98%` and loss `0.2661` remain in Run 1 metrics but are not separately certified (source: `run.log` L65-L66).

### Informational Metrics

- Skipped under the verification procedure after the primary-metric necessary condition failed; raw values remain inline in Run 1.

## Errors & Dead Ends

### 2026-07-27 - Preflight monotonicity included intentional warmup
- Error: `AssertionError: maximum_rise <= 1e-15` before timing.
- Root cause: The disposable verifier incorrectly required global non-increase even though the accepted 0-5% warmup intentionally rises; production schedule behavior was correct.
- Source: semantic preflight traceback at `preflight.py` schedule check, before timing/scoring.
- Do NOT retry: do not apply a global monotonicity assertion; gate non-increase only from the 5% peak onward.

### 2026-07-27 - Rounded analytic ratio incompatible with tolerance
- Error: `AssertionError: abs(ratio - 1.39463009) <= 1e-10` before timing.
- Root cause: The reviewed plan rounded the analytic ratio to eight decimal places while requiring a tighter absolute tolerance; the exact independently calculated ratio is `1.3946300912086436`.
- Source: semantic preflight traceback at `preflight.py` tail-area check, before timing/scoring.
- Do NOT retry: use the full analytic float64 reference when enforcing the declared `1e-10` tolerance.

## Human Notes

> User requested uninterrupted autopilot and offline/local-only execution.
