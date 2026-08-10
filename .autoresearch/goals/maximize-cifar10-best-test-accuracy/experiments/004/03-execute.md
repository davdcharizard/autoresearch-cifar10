# EXP-004: Plateau-Only Conservative RandAugment

## Execution

Overall Status & Info:
- **Created**: 2026-08-05
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-004
- **Commit**: 11f8469
- **Outcome**: completed

## Implementation Notes

### Summary

Changed only `train.py`: added weak and one-operation magnitude-7 RandAugment transform pipelines, factored forkserver loader creation, and added an explicit strong-to-weak switch at the first batch crossing 80% counted training. The switch ends the epoch immediately, evaluates once, deterministically shuts down all eight persistent workers, clears iterator references, collects, and builds a weak crop/flip loader for the low-LR tail. Hard labels, model, seed, optimizer, schedule, and evaluator remain unchanged.

### Surprises & Discoveries

External Claude plan review correctly identified that `del` plus garbage collection was not a deterministic persistent-worker shutdown. The implementation uses the current PyTorch iterator's `_shutdown_workers()` and verifies all Process objects are no longer alive. A same-process forkserver preflight confirmed this lifecycle path works and measured a 2.612-second transition, far below the conservative 30-second limit.

### Decisions

Made forkserver context explicit in both training and preflight to match Python 3.14 behavior. The crossing batch computes LR from pre-batch progress and therefore remains at `lr=0.1`; an immediate post-batch break prevents any subsequent RandAugment batch from running at low LR. The fixed seed remains 42, but RandAugment necessarily changes subsequent worker RNG draws, so a threshold result will be treated as protocol-valid without strong causal attribution.

## Experimental Adjustments

- **Deterministic worker shutdown**: Replaced planned reference deletion alone with explicit iterator shutdown and worker liveness assertion. (ref: `02-plan-review.md` concerns 2-4)
- **Measured transition gate**: Preflight reproduced strong-to-weak teardown/rebuild in one process rather than estimating it. (ref: `02-plan-review.md` concern 2)
- **Wider wall-time margin**: Raised strong-loader gate to 100 batches/s and limited projected total to 500 seconds. (ref: `02-plan-review.md` concern 5)

## Run Log

### Run 1

Metadata:
- **Job ID**: local training PID 2147520 (timeout supervisor PID 2147516)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-05 11:05 UTC
- **Ended**: 2026-08-05 11:11 UTC

Description:
- Run one-operation magnitude-7 RandAugment only during the accepted 80% high-LR exploration phase, then return to crop/flip for the low-LR tail. The moving baseline is 91.83%; success requires at least 91.93%. The full run is allowed only because the worker throughput and lifecycle gate passed with ample margin.

Observations:
- Control loader timed rates: 382.514, 376.698, and 329.341 batches/s; all epochs contained 390 batches and all eight workers exited. (source: `/tmp/exp004_loader_bench.py control` output, persisted before deletion)
- Strong RandAugment loader rates: 170.586, 165.512, and 175.790 batches/s; median 170.586, slowest 165.512 (97.0% of median). Transition-to-first-weak-batch took 2.612s, weak rate was 342.121 batches/s, and both eight-worker sets exited fully. (source: `/tmp/exp004_loader_bench.py transition` output, persisted before deletion)
- Projected full runtime was 338.612s using the reviewed phase-weighted formula, passing the 500-second gate; the measurement establishes feasibility only. (source: preflight calculation from persisted rates)
- Run emitted finite training output normally within 10 seconds; step 50 reported loss 2.0649 at `lr=0.1000`. (source: compact `run.log` tail at 2026-08-05 11:05 UTC)
- The strong-to-weak switch occurred exactly once at epoch 79 and 80.0% counted progress; all eight strong workers stopped. Training resumed with exactly eight weak workers and continued normally. (source: `run.log` augmentation-switch line; process-tree check after switch)
- Accuracy reached 92.30% at epoch 98 and ended at 92.23% on epoch 99. The process exited 0 with a complete summary in 340.7s total. (source: `run.log` L53-L66)

Key Metrics:
- best_test_acc: 92.30% (source: `run.log` L57)
- final_test_acc: 92.23%; final_test_loss: 0.2574 (source: `run.log` L58-L59)
- training_seconds: 300.0s; total_seconds: 340.7s; startup_seconds: 1.1s (source: `run.log` L60-L62)
- peak_vram_mb: 330.1 MB (source: `run.log` L63)
- num_epochs: 99; num_steps: 38,358; num_params: 269,722 (source: `run.log` L64-L66)

## Verification Results

### Conditions Checked

- **Accuracy improvement**: PASS — `92.30% >= 91.93%`; delta `+0.47` points over moving baseline 91.83%. (source: `run.log` L57)
- **Clean completion and numeric summary**: PASS — exit code 0 and all ten expected finite summary keys occurred exactly once. (source: process completion; `run.log` L57-L66)
- **Fixed budget and wall limit**: PASS — counted training 300.0s and total 340.7s, below 600s. (source: `run.log` L60-L61)
- **Evaluation and intervention integrity**: PASS — 25 unique evaluation epochs, terminal evaluation epoch 99 equals `num_epochs=99`, exactly one switch at 80.0%, and `num_params=269722`. (source: parsing commands; `run.log` switch line, L55, L64, L66)

### Informational Metrics

- final_test_acc: 92.23%; final_test_loss: 0.2574. (source: `run.log` L58-L59)
- training_seconds: 300.0s; total_seconds: 340.7s; startup_seconds: 1.1s. (source: `run.log` L60-L62)
- peak_vram_mb: 330.1 MB; num_epochs: 99; num_steps: 38,358; num_params: 269,722. (source: `run.log` L63-L66)
- augmentation switch: epoch 79 at 80.0%, eight workers stopped; post-switch process tree contained exactly eight weak workers. (source: `run.log` switch line; process-tree check)
- loader preflight: control 329.341-382.514 batches/s; strong 165.512-175.790; weak 342.121; transition 2.612s; projected total 338.612s. (source: persisted preflight outputs)

## Errors & Dead Ends

## Human Notes

> The user requires external Claude adversarial reviews with no fallback; both EXP-004 idea and plan reviews completed successfully.
