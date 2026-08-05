# EXP-034: Batch 512 With Fully Scaled LR

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-034
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - local-only run
- **Outcome**: failed - stable throughput feasibility miss

## Implementation Notes

### Summary

Changed exactly four production constants: batch size 512, peak/floor LR 0.4/0.004, and image-equivalent 32,000 maximum steps. The ignored verifier independently checks source/construction semantics, traces batch-512 worker crop/flip and clean-tail replay, measures accepted/candidate complete GPU bodies once, atomically publishes a provenance-bound timing payload, and uses that payload for balanced real-loader wall checks.

### Surprises & Discoveries

Plan review corrected the loader projection to allocate active and hard epoch counts from their separate counted seconds and step costs; 65%/35% are time fractions, not epoch fractions. It also corrected the source-faithful batch-512 transition lag to 1-97 steps and required an atomic single-write timing artifact.

### Decisions

The accepted loader control uses the current source-identical transform at batch 256 because dynamically executed accepted transform classes are not forkserver-importable. The semantic worker oracle compares separately reset batch-512 arms after complete active-epoch consumption and does not claim post-iterator identity with accepted batch 256.

## Experimental Adjustments

- **Strengthened phase and provenance accounting**: Adopted phase-specific epoch counts, exclusive atomic timing publication, exact batch-512 worker tracing, corrected transition lag, and fixed-protocol-only interpretation for best-only wins. (ref: `02-plan-review.md`)
- **Compared transform class names across module oracles**: Dynamic accepted and imported candidate classes have distinct Python identities despite identical audited source. (ref: first semantic preflight error)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A - score not launched
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: Not submitted
- **Started**: N/A
- **Ended**: 2026-07-27 01:06:00 UTC

Description:
- This will be the sole local fixed-seed batch-512 score only if semantic, complete-body GPU, and contemporaneous loader gates pass. It tests the indivisible `512, 0.4 -> 0.004, 32000-cap` operating point while retaining every accepted model, full-gradient, augmentation, optimizer-shape, schedule, seed, and evaluator choice. Primary success is `best_test_acc >=94.42%`; exposure and endpoint metrics are mechanism corroboration only.

Observations:
- Static audit passed: exactly four production constants changed, compilation succeeds, local CIFAR-10 and one idle H20 are available, `prepare.py` is frozen, and neither `run.log` nor `throughput.json` exists. (source: setup commands)
- Semantic preflight passed after the verifier-only namespace fix: exact four-line scope, initial model/construction RNG and optimizer-group semantics, doubled LR curve, image-equivalent cap, finite batch-512 mixup/hard updates, 987,098 parameters, 97 batches/49,664 examples, and exact active/tail worker replay across 49,664 samples per phase. (source: semantic preflight stdout)
- Balanced complete-body timing was stable but failed the fixed material-gain gate. Batch 512 improved weighted image rate from 22,217.0 to 23,571.4 images/s, only 1.06096x, projecting 141.115 passes and 13,780.8 updates versus required 1.10x/146.308/14,287. Per plan, timing was not repeated, no passing payload was published, loader timing was skipped, and the score was not launched. (source: throughput preflight stdout)

Key Metrics:
- **Semantic peak allocation**: 1,984.41 MiB; **parameters**: 987,098.
- **Batch-512 worker replay**: 49,664 active and 49,664 inactive samples; exact clean tail.
- **Accepted medians**: mixup 11.6676 ms; hard 11.2630 ms.
- **Batch-512 medians**: mixup 21.8616 ms; hard 21.4654 ms.
- **Image-rate ratio**: 1.0609597; **projected passes**: 141.11544; **projected updates**: 13,780.81.
- **Timing CV range**: 0.0869%-0.7125%; **peak allocation**: 1,984.41 MiB.
- **Scored runs**: 0; `run.log` was never created and `throughput.json` was not published after the failed gate.

## Verification Results

### Conditions Checked

- **Static and semantic qualification**: PASS - exact scope/construction/LR/cap/update/worker semantics with safe memory.
- **Complete-body stability**: PASS - every accepted/candidate mixup/hard CV <=0.7125%.
- **Material H20 image-rate gain**: FAIL - 1.06096x is below 1.10x; projected 141.115 passes and 13,780.8 steps also miss 146.308 and 14,287.
- **Loader/wall qualification**: SKIPPED after stable throughput failure.
- **Sole score and primary metric**: SKIPPED by preregistered abort criterion.

### Informational Metrics

- Batch 512 nearly doubled step time (mixup 1.8737x; hard 1.9059x) while doubling images, yielding only a 6.10% weighted image-rate gain.

## Errors & Dead Ends

### 2026-07-26 - Cross-module transform type objects cannot be identical
- Error: `AssertionError` comparing accepted/candidate transform type tuples.
- Root cause: the accepted oracle is dynamically executed in `accepted_train`, so its `EarlyRandAugment` class object cannot equal the imported candidate class object even with identical source and order.
- Source: first semantic preflight traceback at `experiments/034/preflight.py:239`.
- Do NOT retry: compare ordered class names and independently audit class source across dynamic module namespaces.

### 2026-07-26 - Batch 512 misses the material image-rate gate
- Error: `AssertionError: 1.06095967071561` at the fixed `retention >= 1.10` gate.
- Root cause: doubling batch size increased mixup/hard step medians to 21.86/21.47 ms, too close to a full 2x cost to compensate for roughly halved decision cadence.
- Source: sole throughput preflight raw payload and traceback.
- Do NOT retry: do not repeat timing, lower the gate, publish a passing payload, run loader timing/score, or repair with adjacent batch/LR/floor/momentum/warmup.

## Human Notes

> Autopilot local-only execution; no user intervention requested.
