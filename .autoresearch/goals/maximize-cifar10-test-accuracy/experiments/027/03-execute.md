# EXP-027: Extra Final Block Plus Early RandAugment

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-027
- **Commit**: 67c8e98
- **PR**: N/A - offline/local-only session
- **Outcome**: completed

## Implementation Notes

### Summary

Composed the exact direct `(2,2,3)` stage-depth model from EXP-011 with the fixed worker-safe early RandAugment implementation from EXP-026. The transform keeps an isolated per-worker RNG and flips its shared byte only after an exhausted epoch, while all accepted optimizer, schedule, mixup, seed, and evaluator behavior remains unchanged.

### Surprises & Discoveries

The EXP-026 ignored preflight can safely supply its already-tested marker, tail-replay, and loader-arm helpers after adapting the current stage-count argument; production code is not imported from an experiment artifact.

### Decisions

Acceptance requires both best and final accuracy at least 94.17 to corroborate the tiny margin at the predetermined endpoint. GPU timing verifies model-side exposure only; loader timing separately verifies worker/wall feasibility using EXP-011 anchors.

## Experimental Adjustments

- None after plan approval.

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 1331952 (timeout PID 1331951)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed (exit 0)
- **Started**: 2026-07-26 21:19:13 UTC
- **Ended**: 2026-07-26 21:25:05 UTC

Description:
- One offline local H20 score of the exact EXP-011 depth treatment composed with exact EXP-026 early RandAugment. It tests near-additive top-1 interaction while retaining at least 130 passes and the clean hard-label tail. It launches only after component-oracle, GPU-exposure, and loader-wall gates pass.

Observations:
- Semantic preflight passed: exact `[2,2,3]` topology, 987,098 parameters, byte-equal construction oracle, preserved RNG, and EXP-026 worker RNG/tail/cutoff semantics.
- GPU timing passed: accepted/candidate weighted steps were 13.135889/14.266856 ms, projecting 130.651247 passes; all window CVs were <=0.003139.
- Loader timing passed: base/composed medians were 2.793634/2.894016 s, CVs 0.002484/0.017627, and differential/absolute wall projections 351.951/426.298 s.
- Final audit confirmed baseline 94.07 at `eb08811`, one idle NVIDIA H20, local CIFAR-10, frozen `prepare.py`, clean syntax/diff, and only tracked `train.py` modified.
- The sole score completed without error. Mixup disabled at step 16,622/195.0 s and RandAugment disabled after epoch exhaustion at step 16,770/196.7 s (lag 148 steps).
- Evaluations occurred once at each accepted-cadence epoch 5 through 130 and once at final partial epoch 134; all 27 evaluation epochs were unique.

Key Metrics:
- `best_test_acc=94.32%`; `final_test_acc=94.22%`; `final_test_loss=0.2523`.
- `training_seconds=300.0`; `total_seconds=345.3`; `startup_seconds=1.1`.
- `num_epochs=134`; `num_steps=25,978`; `data_passes=133.00736`; `num_params=987,098`; `peak_vram_mb=1096.3`.

## Verification Results

### Conditions Checked

- PASS: one NVIDIA H20, local CIFAR-10, frozen evaluator/`prepare.py`, and only `train.py` changed in tracked production scope.
- PASS: semantic component oracles, balanced GPU timing (130.651 projected passes), and loader timing (351.951/426.298 s projections).
- PASS: sole score exit 0, 300.0 counted seconds, 345.3 total seconds, 987,098 parameters, and 133.00736 realized passes.
- PASS: unique once-per-epoch evaluations and exactly one ordered mixup/RandAugment transition with a 148-step lag.
- PASS: `best_test_acc=94.32% >=94.17%` and `final_test_acc=94.22% >=94.17%`.

### Informational Metrics

- Final loss 0.2523 is 0.0259 below EXP-011's 0.2782, supporting the intended early-invariance/deeper-model interaction.
- Peak VRAM was 1096.3 MiB; 134 epochs and 25,978 steps completed.

## Errors & Dead Ends

### 2026-07-26 — Dynamically loaded helper was not forkserver-pickleable
- Error: `_pickle.PicklingError: Can't pickle <class 'exp026_preflight.MarkerDataset'>: No module named 'exp026_preflight'`
- Root cause: the ignored preflight loaded EXP-026's helper with `exec_module` but did not register its import name in `sys.modules`, so forkserver children could not resolve the dataset class.
- Source: semantic preflight traceback at `experiments/027/preflight.py:112`.
- Do NOT retry: do not dynamically load multiprocessing helper classes without first registering the module under `spec.name`.

### 2026-07-26 — Forkserver child could not import registered helper
- Error: `ModuleNotFoundError: No module named 'exp026_preflight'` in all DataLoader children after parent-side registration.
- Root cause: registering a dynamic module only affects the parent; spawned forkserver children also require the helper's directory on `sys.path` to import it by pickle name.
- Source: semantic preflight retry traceback from forkserver workers.
- Do NOT retry: do not expose dynamically loaded classes to forkserver without both parent registration and an importable child path.

### 2026-07-26 — Synthetic helper name did not match its filename
- Error: forkserver children still raised `ModuleNotFoundError: No module named 'exp026_preflight'` with the helper directory on `sys.path`.
- Root cause: the physical helper is `preflight.py`; `exp026_preflight` is not a module the child interpreter can discover from that directory.
- Source: final semantic preflight retry traceback from forkserver workers.
- Do NOT retry: use the physical import name `preflight` when dynamically loading multiprocessing classes from this file.

## Human Notes

> Autopilot run; no intervention requested.
