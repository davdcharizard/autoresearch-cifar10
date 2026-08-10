# Report EXP-020: Isolated PyTorch Nesterov Momentum
- **Created**: 2026-08-06

## Goal

Increase CIFAR-10 `best_test_acc (%)` above the 94.15% moving baseline at `7c1e7d8`. A valid improvement required at least 94.25% under the fixed one-H20, seed-42, 300-counted-second and ten-minute wall protocol.

## Idea & Hypothesis

Add only `nesterov=True` to accepted momentum SGD, preserving model, data, optimizer scalars, schedule, timer, evaluator, and lifecycle. Mandatory external Claude review selected Nesterov because it shares ordinary momentum's steady constant-gradient scale, avoids PNM's sustained LR mismatch, and cleanly resolves EXP-001's confound. The hypothesis predicted 94.30% best accuracy, at least 99% exposure, and switch fit near 89.73%.

## Approach

The production word diff added only `nesterov=True` to the single PyTorch SGD call. Direct deterministic diagnostics matched installed PyTorch 2.9.1 to manual FP32 recurrence across four changing-gradient steps, coupled decay, and an LR change; first buffers were equal and the pre-storage first-direction ratio was 1.899999990. A seed-42 production-distribution corpus of 200 strong and 64 weak batches was persisted before either arm at SHA-256 `49b367ebf14f4ab9d7dc78e49407e532fe821d127e0d6ecbe15fcab5e5f06647`, with exactly 100 CutMix batches and clean worker shutdown.

## Execution

The first direct model diagnostic exposed ordinary non-bitwise CUDA backward gradients across separate identical models. A deterministic controller retry initially required the documented cuBLAS workspace environment, then passed completely with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. These were controller-only corrections; production defaults and the one-keyword intervention never changed.

The paired production-like safety path then trained both optimizers for all 264 persisted batches. State stayed finite, the representative Nesterov buffer/parameter recurrence had zero error, optimizer RNG was neutral, first replay loss was 1.104710x, maximum update spike was 1.736714x, and candidate loss EMA remained below control in both phases. Nevertheless, at strong step 11 Nesterov predicted class 2 for 124/128 samples while control's top class held 83/128, crossing the pre-registered candidate-only 95% concentration veto. Evidence was serialized before assertion. Timing and production were skipped; no `run.log` or test metric was produced.

## Results

- **Primary metric**: `NaN` (baseline: `94.15%`; no accuracy run)
- **Observations**: Nesterov's exact recurrence and overall short-run optimization were healthy, with strong final/max loss-EMA ratios 0.960829/0.976076 and weak ratios 0.964297/0.983864. The failure was a single early but decisive class-concentration event, not non-finite state, sustained loss divergence, transition shock, excessive update spike, or implementation mismatch.
- **Analysis**: The event is consistent with the required stronger current-gradient transient changing early class geometry even though steady scale matches ordinary momentum. It may have recovered later, as the lower loss and finite trajectory suggest, but the reviewed protocol deliberately treated candidate-only greater-than-95% concentration as a gross-instability veto. Relaxing it after observing step 11, adding warmup/clipping, or lowering LR would define a new intervention. This exact no-warmup operating point is therefore unproven and blocked, while Nesterov more broadly is not disproven.
- **Key Learning**: Nesterov hit 96.875% one-class predictions at step 11 despite finite state and lower loss; this exact no-warmup point is unsafe.

## Verification

- **Conditions**: Baseline/scope, literal one-keyword source, installed/manual recurrence, target/RNG semantics, immutable corpus, and lifecycle passed. Paired safety failed the candidate-only concentration condition; timing, exposure, production, evaluation-count, and accuracy conditions were not reached.
- **Review Notes**: The failure evidence is replayable and tied to the immutable corpus digest, unlike EXP019. The direct first-buffer equality was established under deterministic diagnostics; normal paired CUDA backwards produced minor pre-step differences, but that does not explain or invalidate the much larger step-11 histogram divergence. No test evaluation or metric gaming occurred.
- **Verdict**: invalid
- **Verdict Basis**: A registered safety gate blocked the scored run, leaving only partial production-distribution evidence and no `best_test_acc`.

## Unexplored Avenues

- A separately reviewed Nesterov experiment with explicit first-step warmup or smaller plateau LR could address the transient, but it would no longer be the clean one-keyword test and would confound the accepted schedule.
- A matched-effective-scale PNM variant could isolate optimizer-shaped noise better than paper-default PNM, but scale matching, initialization, and coupled decay require a new derivation and adversarial review.
- The concentration threshold itself may be overly conservative for transient finite optimizers; changing it must happen prospectively as a goal-wide protocol decision, never as an EXP020 rescue.

## Next Steps

- **High confidence**: retire exact no-warmup `nesterov=True` at LR 0.1 under the current safety contract; do not rerun it.
- **Medium confidence**: seek a mechanism that preserves the accepted optimizer path and changes representation or data without identity-oriented underfit.
- **Medium confidence**: if revisiting optimizer noise, derive a scale-matched PNM-style update before proposing code rather than importing paper defaults.

## Exit Action Results

- None defined.
