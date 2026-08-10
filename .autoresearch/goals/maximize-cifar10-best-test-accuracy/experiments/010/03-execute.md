# EXP-010: Conservative Plateau CutMix

## Execution

Overall Status & Info:
- **Created**: 2026-08-05
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-010
- **Commit**: 7c1e7d8
- **Outcome**: completed

## Implementation Notes

### Summary

Added a module-level torchvision v2 CutMix transform and forkserver-safe collator. The strong loader applies alpha-1 CutMix to a fixed 50% gate while preserving CPU RNG state around the new draws; the weak loader uses default collation. Strong mixed/hard formats are counted before the timed region, weak targets are asserted one-dimensional, and the existing switch line reports provenance.

### Surprises & Discoveries

No tracked dependency or evaluator change was needed: installed `F.cross_entropy` accepts both integer and probability targets, and the existing loader-rebuild boundary cleanly changes collation with the transform.

### Decisions

The mandatory Claude plan review added an integrated real-loader/model contention gate because isolated worker and GPU tests cannot expose joint starvation. The underfit checkpoint remains diagnostic rather than an abort because it appears only after roughly 80% of the counted budget and the hard tail is the proposed repair mechanism.

## Experimental Adjustments

- None after implementation; alpha and probability remain pre-registered at 1.0 and 0.5.

## Run Log

### Run 1

Metadata:
- **Job ID**: PID 2253502 (timeout supervisor 2253501; tool session 7244)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-06 03:20 UTC
- **Ended**: 2026-08-06 03:26 UTC

Description:
- One fixed-seed width-2 run will compose CutMix with half of N1/M7 plateau batches, then retain the accepted hard-label weak tail. It launches only after functional, RNG, worker, synchronized-loss, lifecycle, and joint-contention gates pass. The hypothesis predicts at least 93.65% with at least 97% of EXP-007 exposure.

Observations:
- Static compilation, Ruff, formatting, pre-commit, whitespace, and exact tracked-file scope checks passed.
- Functional checks passed: forced hard/mixed branches produced correct shapes/dtypes, probability rows summed exactly to one, observed pasted area matched donor target mass, both CE paths had finite gradients, and the CutMix collator restored CPU RNG byte-for-byte.
- Worker preflight passed at 199.89/195.39/179.51 warmed batches/s with 600/1,170 mixed batches (0.5128), eight strong workers stopped, and a 2.971-second transition to hard weak targets.
- H20 hard/soft medians were 10.823/10.829 ms, projecting 0.99979 retention and 27,137 steps. The 1,000-step real joint test took 11.420s versus 10.928s synchronized work (1.045x), waited only 0.330s for batches, mixed 461/1,000, stopped all eight workers, and peaked at 598.7 MB.
- Full-run startup succeeded on CUDA with 1,073,962 parameters and 390 batches per epoch. (source: `run.log` startup lines)
- The strong checkpoint reached 89.73%, safely above the 87.08% underfit marker, with 10,673/21,446 mixed batches (49.77%) and eight stopped workers. The first weak checkpoint reached 93.16%, then the tail rose to 94.15% at the final epoch. (source: `run.log` L14-L43)
- Process exit 0 produced all summary fields after 300.0 counted and 330.7 total seconds; no traceback, OOM, assertion, `nan`, or `inf` pattern was present. (source: process status; `run.log` L45-L54)

Key Metrics:
- best/final test accuracy: 94.15% / 94.15% @ epoch 69 (source: `run.log` L43, L45-L46)
- final test loss: 0.1934 (source: `run.log` L47)
- final train-loss EMA: 0.0459 @ step 26,850 (source: `run.log` final progress record)
- training/total/startup: 300.0s / 330.7s / 1.0s (source: `run.log` L48-L50)
- peak VRAM: 598.7 MB (source: `run.log` L51)
- exposure: 69 epochs / 26,898 steps / 1,073,962 parameters (source: `run.log` L52-L54)
- realized CutMix: 10,673/21,446 strong batches = 49.77% (source: `run.log` L15)

## Verification Results

### Conditions Checked

- **Primary accuracy**: passed. 94.15% >=93.65%; +0.60 points over the 93.55 baseline and +0.50 above the gate. (source: results-index baseline; `run.log` L45)
- **Completion/summary/timing**: passed. Exit 0, ten finite fields, 300.0 counted seconds, 330.7 total seconds. (source: process status; `run.log` L45-L54)
- **Hardware/scope/lifecycle**: passed. One idle H20, only `train.py` changed, one 80.0% switch, eight workers stopped, 19 evaluations on 19 unique epochs, no weak soft-target assertion. (source: preflight, diff, `run.log` L6-L43)
- **Mechanism integrity**: passed. Mixed fraction 49.77%; 26,898 steps retain 99.10% of EXP-007 and exceed the 26,329 floor; strong checkpoint 89.73% avoided the 87.08% underfit marker.

### Informational Metrics

- final_test_acc 94.15%, final_test_loss 0.1934, peak_vram_mb 598.7, num_epochs 69, num_steps 26,898, num_params 1,073,962. (source: `run.log` L46-L54)
- Preflight hard/soft medians 10.823/10.829 ms; projected retention 99.98%; joint wall/step ratio 1.045; worker rates 179.51-199.89 batches/s. (source: Run 1 observations)

## Errors & Dead Ends

### 2026-08-05 — Disposable worker benchmark lacked forkserver main guard
- Error: `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase.`
- Root cause: The first `/tmp` benchmark invoked a Python 3.14 forkserver DataLoader at module import time.
- Source: initial `/tmp/exp010_worker_bench.py` invocation before any experiment run.
- Do NOT retry: Wrap disposable multiprocessing harness entry points in `if __name__ == "__main__"`; the corrected identical benchmark passed.

## Human Notes

> Autopilot session; no execution-phase intervention.
