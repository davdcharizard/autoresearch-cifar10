# EXP-030: Raise the Weak-Tail Start LR to 0.02

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-030
- **Commit**: (pending — committed on loop success)
- **Outcome**: no-improvement — valid 93.90% best missed the 94.25% gate

## Implementation Notes

### Summary

Created the experiment branch from the clean integration baseline and changed only `ANNEAL_START_LR` from 0.01 to 0.02 in `train.py`. The complete accepted 80% strong curriculum and every other model, optimizer, data, lifecycle, timing, and evaluator choice remain unchanged. Compile, Ruff, format, pre-commit, whitespace, and exact-diff checks passed before preflight.

### Surprises & Discoveries

The external plan review correctly noted that program equivalence before 80% is not bitwise trajectory equivalence to the historical EXP010 run because the benchmark does not force deterministic CUDA algorithms. It also identified that 19 evaluations and 26,898 steps are historical references, not invariant consequences of a fixed-time run; the plan now accepts normal count jitter while preserving evaluator-integrity gates.

### Decisions

The preflight binds its schedule audit to the actual `train.py` AST/source and extracts the real model class definitions without importing the module-level `Eval` object, so it cannot invoke test evaluation. Its copied-state 2x update check is treated only as arithmetic/scope validation. One unchanged retry is permitted only for a documented external infrastructure failure that prevents a valid summary; a valid completion is never rerun.

## Experimental Adjustments

- None. The approved single-scalar intervention was implemented literally.

## Run Log

### Run 1

Metadata:
- **Job ID**: local timeout PID 2522226 (execution session 20459)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log
- **WandB**: N/A
- **Status**: completed, exit 0, no retry
- **Started**: 2026-08-06 19:04:48 UTC
- **Ended**: 2026-08-06 19:10:22 UTC

Description:
- One local fixed-seed production run of the accepted width-2 N1/M7 plus p=0.5 CutMix recipe, changing only the weak-tail cosine start LR from 0.01 to 0.02. The hypothesis predicts preserved strong-phase behavior and more useful hard-label refinement after 80%, reaching at least 94.25% best test accuracy. All output is redirected to `run.log`; no valid completion will be rerun.

Observations:
- Code-quality and exact one-line scope checks passed. The no-test controller bound the audit to `train.py` assignment line 27 and sole load line 236; accepted/candidate LR was identical at 0.1 through 80%, candidate began at 0.02 immediately above 80%, remained monotone, and both ended at 0.0001. Initial state/RNG hashes matched, parameter count was 1,073,962, momentum buffers remained equal, and copied-state aggregate update ratio was 2.000000000 (tensor range 1.999894128-2.000066692). (source: `experiments/030/preflight.log`)
- Environment gate found exactly one idle NVIDIA H20 with 97,871 MiB, 0 MiB used, 0% utilization, no compute process, and no stale root run log. Untracked `data/` was preserved. (source: pre-launch `nvidia-smi` and worktree checks recorded 2026-08-06)
- The production process started successfully on CUDA, reported 1,073,962 parameters, the fixed 300-second budget, and 390 batches per epoch. (source: `run.log` initial lines)
- The unchanged strong phase evaluated at 83.26%, 86.89%, 86.37%, and 81.38% at the four early checkpoints, then recovered to 88.56% at the 80.0% switch. The switch stopped all eight workers and reported 10,652 CutMix batches among 21,408 strong batches (49.757%). (source: `run.log` L6-L15)
- The first weak checkpoint reached 93.08% at epoch 56. The candidate rose to 93.87% at epoch 62 and peaked at 93.90% at epoch 66, then regressed to 93.79% final with 0.2083 NLL. (source: `run.log` L17-L43)
- The run exited cleanly after 300.0 counted seconds and 331.8 total seconds, with 26,758 steps, 69 epochs, 19 unique evaluations, and no fatal/non-finite signal. It was not retried. (source: `run.log` L44-L54 and post-run integrity parser)

Key Metrics:
- `best_test_acc`: 93.90%, delta -0.25 points versus 94.15 baseline and -0.35 below the 94.25 gate (source: `run.log` L45; results index baseline query).
- `final_test_acc`: 93.79%; `final_test_loss`: 0.2083 (source: `run.log` L46-L47).
- Switch/first weak: 88.56% / 93.08%, respectively -1.17 / -0.08 points versus EXP010's 89.73% / 93.16% (source: `run.log` L14-L17; `experiments/010/04-analysis.md`).
- Exposure: 26,758 steps, 99.48% of EXP010's 26,898, above the 26,629 diagnostic; peak VRAM 598.7 MiB (source: `run.log` L51-L53).
- Timing: 300.0s counted, 331.8s total, 1.0s startup (source: `run.log` L48-L50).

## Verification Results

### Conditions Checked

- **Baseline/scope — pass:** baseline query returned 94.15 at `7c1e7d8`; only tracked `train.py` changed and its exact diff was the registered `ANNEAL_START_LR = 0.01 -> 0.02`. Untracked `data/` remained untouched.
- **Completion/summary — pass:** exit 0 and exactly one complete finite ten-field summary; 300.0 counted seconds and 331.8 total seconds satisfy the fixed budget and process bound. (source: `run.log` L44-L54)
- **Model/data/lifecycle — pass:** 1,073,962 parameters; one switch at 80.0%; eight stopped workers; 10,652/21,408 = 49.757% strong CutMix; successful weak one-dimensional target assertions; no error signal. (source: `run.log` L14-L17, L54)
- **Exposure/evaluator — pass:** 26,758 steps exceed the 26,629 diagnostic; 19 evaluation epochs were unique, included the four early checkpoints and terminal epoch 69, and no epoch was evaluated twice. (source: parsed `run.log`; summary L52-L53)
- **Primary metric — fail:** 93.90% is below the required 94.25%; this valid result is no-improvement and was not rerun. (source: `run.log` L45)

### Informational Metrics

- `final_test_acc=93.79%`, `final_test_loss=0.2083`, `training_seconds=300.0`, `total_seconds=331.8`, `startup_seconds=1.0`, `peak_vram_mb=598.7`, `num_epochs=69`, `num_steps=26758`, `num_params=1073962`. (source: `run.log` L46-L54)
- Best checkpoint: 93.90% at epoch 66; best-final gap 0.11 points. (source: `run.log` L37-L43)

## Errors & Dead Ends

- None.

## Human Notes

> Autopilot requested; no execution-phase intervention.
