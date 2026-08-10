# EXP-002: Front-Loaded Probabilistic CutMix

## Execution

Overall Status & Info:
- **Created**: 2026-08-05
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-002
- **Base Node**: 001
- **Commit**: a36dc09
- **Outcome**: completed

## Implementation Notes

### Summary

Added a dependency-free CutMix helper and a fixed early-phase gate to the EXP-001 training recipe. The implementation uses standard square-root geometry, one shared rectangle, safe source-patch cloning, clipped-area target weighting, and one model forward. All CutMix work occurs inside the existing charged timer. Applied/eligible counters and fixed configuration values are logged without changing the final summary keys.

### Surprises & Discoveries

Claude's adversarial plan review identified that new random draws could perturb the parent's global shuffle and drop-path streams. The implementation therefore uses dedicated CPU and CUDA generators, each seeded once with 42. The deterministic GPU smoke passed with a 484-pixel patch and adjusted original-target lambda 0.527344, and also confirmed the forced zero-area identity case.

### Decisions

The helper accepts optional lambda, center, and permutation only to make geometry and target orientation deterministically testable; the timed path draws them from dedicated generators. `Beta(1,1)` is implemented exactly as a uniform draw. The CutMix cutoff and drop-path annealing intentionally share the 75% boundary, so later analysis must not attribute final-quarter dynamics to either transition alone.

## Experimental Adjustments

None. The reviewed `prob=0.5`, `alpha=1.0`, `end=0.75`, and seed-42 dedicated RNG recipe is fixed before launch.

## Run Log

### Run 1

Metadata:
- **Job ID**: local timeout PID 3926363 (training PID 3926364; exec session 13924)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/tree-v0-gpt-5-6-sol/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-05 09:07:14 UTC
- **Ended**: 2026-08-05 09:15:11 UTC

Description:
- Run the EXP-001 WRN recipe with fixed front-loaded probabilistic CutMix once on physical GPU 0. The intervention targets the parent's measured train/test generalization gap while retaining its validated architecture and charged-time optimization. Success requires at least 94.72% `best_test_acc`, clean completion, and an auditable applied/eligible CutMix ratio near 0.5.

Observations:
- `run.log` started writing within 16 seconds. (source: absolute run log path above, checked 2026-08-05 09:07:30 UTC)
- The process exited 0 with no normalized NaN/Inf, traceback, CUDA, or memory-error match. The final CutMix exposure was 10,257 applied of 20,668 eligible batches (49.63%). (source: `run.log` L294)
- Best accuracy was 95.23% at epoch 143; the mandatory final epoch-144 evaluation was 95.19%. (source: `run.log` L291-L293)

Key Metrics:
- `best_test_acc`: 95.23% at epoch 143, +0.61 points vs parent EXP-001 (source: `run.log` L291, L296)
- `final_test_acc`: 95.19%; `final_test_loss`: 0.2044 (source: `run.log` L293, L297-L298)
- `training_seconds`: 300.0; `total_seconds`: 467.1 (source: `run.log` L299-L300)
- `num_steps`: 27,950 across 144 epochs; `num_params`: 2,748,890 (source: `run.log` L303-L305)
- `peak_vram_mb`: 1,178.9 MiB (source: `run.log` L302)

## Verification Results

### Conditions Checked

- **Parent-relative accuracy threshold - PASS**: parent EXP-001 is 94.62%, so the threshold is 94.72%; EXP-002 reached 95.23%, a +0.61-point gain. (source: `tree.sh show ... 001`; `run.log` L296)
- **Clean completion and fixed budget - PASS**: launch exited 0, all ten summary keys are present, charged training was 300.0 seconds, total runtime was 467.1 seconds, and normalized error monitoring found no match. (source: exec session 13924; `run.log` L295-L305)
- **CutMix mechanism audit - PASS**: startup config matches the reviewed fixed recipe; deterministic helper smoke passed; all CutMix work is between `t0` and CUDA synchronization; realized ratio was 0.4963. (source: `run.log` L3, L294; `train.py` L288-L337; smoke output in Implementation Notes)
- **Model and validation integrity - PASS**: parameter count stayed 2,748,890; 144 evaluations occurred for 144 epochs, including final epoch 144. (source: `run.log` L293, L303-L305)

### Informational Metrics

- `final_test_acc`: 95.19% (source: `run.log` L297)
- `final_test_loss`: 0.2044 (source: `run.log` L298)
- `training_seconds`: 300.0 s (source: `run.log` L299)
- `total_seconds`: 467.1 s (source: `run.log` L300)
- `startup_seconds`: 1.2 s (source: `run.log` L301)
- `peak_vram_mb`: 1,178.9 MiB (source: `run.log` L302)
- `num_epochs`: 144 (source: `run.log` L303)
- `num_steps`: 27,950 (source: `run.log` L304)
- `num_params`: 2,748,890 (source: `run.log` L305)
- `cutmix_exposure`: 10,257/20,668 = 0.4963 (source: `run.log` L294)

## Errors & Dead Ends

## Human Notes

> Autopilot session. User required Claude-only adversarial review; the refreshed Claude plan review completed successfully and was applied before execution.
