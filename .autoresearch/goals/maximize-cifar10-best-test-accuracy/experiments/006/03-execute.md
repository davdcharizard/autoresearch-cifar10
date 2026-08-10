# EXP-006: Plateau-Only Fixed-Square Cutout

## Execution

Overall Status & Info:
- **Created**: 2026-08-05
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-006
- **Commit**: (pending — committed on loop success)
- **Outcome**: failed (valid run, primary metric no-improvement)

## Implementation Notes

### Summary

Starting from accepted commit `11f8469`, changed only `train.py`. The plateau transform now removes PIL RandAugment and applies `RandomErasing` after tensor conversion and normalization with `p=1.0`, fixed 25% area, square aspect, normalized-zero fill, and in-place mutation. The live initial loader uses that transform; setup assertions prove its object identity, exact ordered transform types and eraser parameters, and absence of RandAugment without sampling data or consuming training RNG. Both phase predicates and the single switch provenance were mechanically renamed to Cutout while preserving the accepted 80% lifecycle and weak tail.

### Surprises & Discoveries

The first timed diagnostic was contaminated by a full CPU `torch.isfinite(inputs).all()` reduction on every prefetched batch, reducing reported loader rate to 67.6-77.8 batches/s. Restricting validation to the first batch of each timed epoch measured the loader itself at 247.5-256.2 batches/s. The candidate therefore has ample host headroom; the initial number was a diagnostic-cost artifact, not a Cutout throughput failure.

### Decisions

- Adopted Claude's plan-review recommendation to assert that the live loader dataset references the exact reviewed Cutout transform. Used structural assertions rather than fetching a proof batch, because fetching would advance loader/RNG state before training.
- Treat optimizer exposure above EXP-004 by more than 1.5% as part of Cutout's fixed-time computational benefit and an attribution caveat, not as an invalid run. The goal rewards fixed-time accuracy, while mechanism claims must separate cheaper loading from pure occlusion effects.
- Kept the canonical 16x16 mask despite shortened-schedule underfitting risk; no strength tuning or rerun is permitted inside EXP-006.

## Experimental Adjustments

- **Removed timed-loop validation overhead from the disposable loader diagnostic**: Validate finiteness and shapes on the first batch only so the feasibility measurement reflects data production rather than a CPU tensor scan. Candidate transform, loader settings, thresholds, and training code were unchanged. (ref: preflight observations and Errors & Dead Ends)

## Preflight Results

- Static checks: `py_compile` passed; Ruff 0.15.6 passed; pre-commit Ruff and formatter passed after one mechanical format pass; `git diff --check` passed.
- Transform semantics: 32/32 nonzero synthetic inputs received exactly 256 all-channel-zero pixels with a 16x16 bounding box; all off-mask values remained 1.0 and all outputs were finite.
- Model integrity: output shape `(2, 10)` and 269,722 parameters.
- Loader integrity: 390 batches/epoch; warmed Cutout rates 256.2, 247.5, and 250.1 batches/s (minimum 247.5 > 160 gate); eight Cutout workers stopped; Cutout-to-first-weak-batch transition 3.020s (<5s gate); eight weak workers stopped.

## Run Log

### Run 1

Metadata:
- **Job ID**: local training PID 2173253 (supervisor PID 2173249)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-05 12:29 UTC
- **Ended**: 2026-08-05 12:35 UTC

Description:
- One fixed-seed local H20 run of the reviewed Cutout recipe under the 300-second counted training and 600-second total supervisor. The first 80% uses one mean-valued 16x16 erased patch per crop/flip view at `lr=0.1`; the final 20% rebuilds the weak crop/flip loader and follows the accepted low-LR cosine tail. Expected validity signals are one clean eight-worker switch, at least 37,783 steps, and a complete numeric summary. Improvement requires `best_test_acc >= 92.40%` against the 92.30% moving baseline.

Observations:
- The run progressed continuously with finite loss and no CUDA, OOM, DataLoader, traceback, NaN, or Inf signature. (source: `run.log`, monitored tails and error-pattern scan)
- The last Cutout checkpoint was 83.15% at epoch 78 with nearest logged debiased loss EMA 0.5170; EXP-004's last strong checkpoint was 84.60%, consistent with stronger underfitting from masking 25% of every plateau view. (source: `run.log`, epoch 78 transition region)
- Exactly one switch occurred at epoch 78 and 80.0% progress with all eight Cutout workers stopped. The next weak epoch used the low-LR tail and reached 90.47%; subsequent accuracy rose monotonically in broad terms to 91.63% but did not approach the baseline. (source: `run.log`, switch and epoch 79-98 evaluation records)
- There were 25 evaluation records at 25 unique epochs, so the at-most-once-per-epoch constraint passed. (source: `run.log`, all `eval ep` records)

Key Metrics:
- best_test_acc: 91.63% (source: `run.log` final summary)
- final_test_acc: 91.63% (source: `run.log` final summary)
- final_test_loss: 0.2617 (source: `run.log` final summary)
- training_seconds: 300.0 (source: `run.log` final summary)
- total_seconds: 339.2 (source: `run.log` final summary)
- startup_seconds: 1.0 (source: `run.log` final summary)
- peak_vram_mb: 330.1 (source: `run.log` final summary)
- num_epochs: 98 (source: `run.log` final summary)
- num_steps: 38,028, which is 99.14% of EXP-004's 38,358 and above the 37,783 floor (source: `run.log` final summary)
- num_params: 269,722 (source: `run.log` final summary)

## Verification Results

### Conditions Checked

- **Baseline and hardware preconditions — passed**: baseline 92.30% at `11f8469`; one idle NVIDIA H20 with 97,871 MiB and no stale run log before launch. (source: pre-launch command outputs)
- **Completion and numeric summary — passed**: exit code 0; all ten finite fields present; training 300.0s, total 339.2s <600s, 38,028 steps >=37,783, and 269,722 parameters. (source: `run.log` final summary)
- **Lifecycle and evaluation integrity — passed**: one `cutout->base` switch at 80.0%, eight stopped workers, no RandAugment switch, and 25 unique evaluation epochs. (source: `run.log` switch/evaluation records)
- **Primary metric improvement — failed**: 91.63% is 0.67 percentage points below the 92.30% moving baseline and 0.77 points below the required 92.40% threshold. Verification stopped on this necessary-condition failure. (source: `run.log` final summary; results-index baseline query)

### Informational Metrics

- Skipped as a verification section after the primary necessary condition failed. All run values and pre-registered diagnostics are preserved above in Run 1 Key Metrics and Observations for analysis.

## Errors & Dead Ends

### 2026-08-05 — Disposable preflight could not import project module
- Error: `ModuleNotFoundError: No module named 'train'`
- Root cause: Executing a script located under `/tmp` makes `/tmp` the first import path; the project root was not on `sys.path`.
- Source: local preflight command before the full run.
- Do NOT retry: Do not invoke the disposable script without `PYTHONPATH=.` from the project root.

### 2026-08-05 — Timed loader diagnostic included tensor-scan overhead
- Error: `cutout_min_bps: 67.6` followed by the 160 batches/s feasibility assertion.
- Root cause: The diagnostic executed a full CPU finiteness reduction on every batch inside the timed loop, measuring validation work rather than loader throughput.
- Source: disposable preflight output before the full run.
- Do NOT retry: Do not perform per-batch tensor reductions inside timed loader benchmarks; validate the first batch and time ordinary consumption.

## Human Notes

> None; autopilot execution.
