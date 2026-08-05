# EXP-047: Post-GAP Feature Mixup Replacement

## Execution

Overall Status & Info:
- **Created**: 2026-07-27
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-047
- **Commit**: (pending - committed on loop success)
- **PR**: N/A (offline local run)
- **Outcome**: failed - valid normal-exposure metric miss

## Implementation Notes

### Summary

Replaced early pixel blending with the same Beta/permutation draw used to blend the actual post-GAP128-vector immediately before the accepted residual MLP. Added a default-`None` forward argument so hard/evaluation behavior is unchanged. The ignored preflight binds actual pooled and MLP-input tensors, checks forward/Jacobian/update/RNG controls, and counterbalances complete early/hard timing.

### Surprises & Discoveries

Because GAP is functional, ordinary module hooks cannot observe its unblended output. The harness temporarily wraps the candidate module's actual adaptive-pooling function, retains that returned tensor, and restores the function immediately; a pooled-head pre-hook independently captures the scored blended tensor.
Initial signature patching matched `PreActBlock.forward` rather than `WideResNet.forward`; static diff review caught and corrected it before any preflight or scored execution.

### Decisions

The fixed semantic fixture uses coefficient0.3 and a permutation containing two3-cycles so neither weight swaps nor inverse-permutation mistakes can pass. The score is interpreted only as the complete clean-spatial-BN plus post-GAP interpolation replacement.

## Experimental Adjustments

- None.

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID1451506 (launcher1451499)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-27 08:45:07 UTC
- **Ended**: 2026-07-27 08:52:34 UTC

Description:
- One fixed-seed local H20 score of early post-GAP feature mixup replacing accepted input mixup. It launches only after exact state/default-path/feature-algebra/Jacobian/update/RNG and exposure gates pass. Success requires best accuracy at least94.58% and at least127 realized passes; a valid result is never rerun.

Observations:
- Semantic gate passed: default path exact; actual pooled-tensor Jacobian error `0`; fixed feature loss `2.259622`; head Jensen gap `0.104502`; pair cosine `0.970949`; mixed/unmixed norm ratio `0.975275`; exact draw/RNG alignment; fresh/preseeded parameter errors `2.98e-8` and momentum errors `1.49e-8` (source: semantic preflight stdout, 2026-07-27).
- Timing gate passed all16 windows: arm CVs `0.20-0.69%`, ratio CVs `0.47%/0.79%`, retentions `[0.991861,0.998059,0.995083,0.984240]`, median projected exposure `129.453342` passes, and candidate peak `607.167MiB` (source: timing preflight stdout, 2026-07-27).
- Sole score launched with exact command and produced CUDA,1,003,482 params,300-second budget, and195 batches/epoch (source: `run.log` L1-L4).
- Sole score exited zero with one finite summary. Mixup stopped once at step16,237/195.0s and RandAugment after exhausted epoch84 at step16,380/196.6s;27 evaluations were unique every-fifth plus final131 and no error appeared (source: `run.log` L1-L73).
- Best and final accuracy were both94.20% with loss0.2619 at normal exposure, rejecting the exact bundled replacement rather than systems feasibility (source: `run.log` L64-L73).

Key Metrics:
- `best_test_acc`: `94.20%`,0.28 below baseline and0.38 below threshold (source: `run.log` L64).
- `final_test_acc` / `final_test_loss`: `94.20%` / `0.2619` versus accepted94.45% /0.2456 (source: `run.log` L65-L66; EXP036 report).
- Exposure:25,409 steps=`130.09408` passes across131 epochs (source: `run.log` L71-L72).
- Counted/wall/startup:300.0/342.5/1.1s; peak1,094.4MiB; params1,003,482 (source: `run.log` L67-L73).

## Verification Results

### Conditions Checked

- **Completion/resource contract - PASS**: exit0, one H20, finite summary,300.0s counted/342.5s wall, correct transitions,27 unique evaluations,1,003,482 params,130.09408 passes, no errors.
- **Primary metric improvement - FAIL**: best94.20% is below baseline94.48% and required94.58%.
- **Bundled hypothesis - FAIL**: normal exposure cleared127 but accuracy did not; exact post-GAP replacement is rejected.
- **Corroboration - skipped after metric failure**: final94.20% and loss0.2619 remain descriptive.

### Informational Metrics

- Skipped under fail-fast primary verification; raw metrics remain above.

## Errors & Dead Ends

- None.

## Human Notes

> Autopilot requested; user asleep. Offline/local only.
