# EXP-008: Implementation-Audited Mild RandAugment in the Regularized Phase

## Execution

Overall Status & Info:
- **Created**: 2026-08-05
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-008
- **Base Node**: 004
- **Commit**: `9625222`
- **Outcome**: failed - preregistered loader feasibility gate rejected the fixed package before a metric run

## Implementation Notes

### Summary

Implemented a paired-view transform in `train.py` only. Each worker applies the exact parent crop/flip once, emits the parent-identical clean FP32 tensor plus an audited magnitude-5 RandAugment uint8 tensor, and isolates RandAugment draws through a worker-seed-keyed private torch generator. The main loop binds one charged-progress scalar, selects the augmented view only below 0.75, normalizes it on GPU, and otherwise selects the clean view. Existing CutMix, drop path, SAM, model, optimizer, evaluator, and summary remain unchanged; new config and exposure audits describe the fixed policy and boundary.

### Surprises & Discoveries

- Claude's plan review identified that the paper's magnitude 2 is not semantically portable to torchvision 0.24.1: on 32x32 inputs, translation truncates below one pixel and posterization stays at eight bits. Before any accuracy output, the package was fixed once at magnitude 5, the lowest audited mild bin with two-pixel translations and seven-bit posterization.
- The paired design is semantically exact but loader-bound. Even with the augmented view encoded as uint8 and normalized on GPU, its worst sustained throughput was 15,549.8 images/s. Claude later noted that the 30,720 absolute headroom floor was miscalibrated because the parent worst epoch also missed it; the stop remains robust because candidate best throughput was below parent worst and below raw early GPU demand.
- Two initial verification-harness failures were harness-only: the exception test initially ignored the intended parent crop/flip RNG consumption, and the debug loader unpacked the dataset's `(views, target)` nesting incorrectly. A third harness attempt used `torch.quantile` on too many pixels and was replaced with an exact 256-bin histogram. No implementation or policy value changed.

### Decisions

- Returning both views is necessary because prefetched workers cannot observe main-process wall-clock progress. Main-process selection is the only exact way to keep every SAM batch clean without an epoch-aligned approximation or loader flush.
- Augmented views remain uint8 through collation/pinning to reduce IPC; clean views stay on the exact parent CPU normalization path for bitwise parity.
- The fixed package is rejected before launch. No worker-count, operation, magnitude, phase, or representation change is allowed to rescue it, and no full GPU training run is launched. The planned total-runtime projection was not reached and is not claimed.

## Experimental Adjustments

- **Magnitude 2 -> 5 before metric execution**: deterministic torchvision operation-table inspection showed magnitude 2 was near-identity and not semantically equivalent to the cited paper. Magnitude 5 was locked before any accuracy run or training launch (ref: `02-plan-review.md`; `02-plan.md` Decision Lock).

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A - full training not launched
- **Log file(s)**: N/A - preflight command output was distilled below; no `run.log` created
- **WandB**: N/A
- **Status**: failed at preflight
- **Started**: 2026-08-05 17:18 UTC
- **Ended**: 2026-08-05 17:22 UTC

Description:
- Preflight the fixed paired-view magnitude-5 RandAugment implementation before any GPU metric run. The fixed plan required bitwise parent-clean parity, deterministic private worker streams, nondegenerate pixel effects, and worst-epoch loader throughput at least 1.20x the roughly 25,600-image/s early GPU consumption rate. Failure of this gate rejects the package without tuning or observing accuracy.

Observations:
- Static checks passed: `py_compile`, Ruff, and `git diff --check` all exited 0; only tracked `train.py` differs from parent `1a8d0de`.
- Zero-worker clean tensors and post-crop torch RNG matched the parent exactly. Python/NumPy states were unchanged; injected RandAugment failure advanced private state, restored parent-global torch state, and propagated.
- The resolved magnitude-5 table had 11 nonzero numeric operations, including 2.4169-pixel translation (integer application 2) and seven-bit posterization.
- Across 10,240 production-worker samples, 91.1914% changed at least one pixel; mean absolute uint8 delta was 14.0395 and p99 delta was 185. Eight workers produced eight distinct private seed keys.
- One complete 195-batch shuffled epoch matched parent targets and clean tensors by SHA-256 on every batch. A second candidate worker recreation replayed all clean, augmented, and target hashes exactly.
- Loader gate failed across five worker-recreated epochs. Parent epoch times were 1.444/1.514/1.461/1.693/1.677s; candidate times were 3.011/3.210/2.772/2.881/2.684s. Candidate sustained rates were 16,578.9/15,549.8/18,009.1/17,326.5/18,601.0 images/s, all below the 30,720 headroom floor. Candidate p90 batch inter-arrival was 78.819 ms versus parent 33.386 ms.

Key Metrics:
- Parent loader worst/median throughput: 29,485.7 / 32,969.6 images/s (source: five-epoch preflight, distilled above).
- Candidate loader worst/median throughput: 15,549.8 / 17,326.5 images/s (source: five-epoch preflight, distilled above).
- Required headroom / measured early consumption: 30,720 / 25,600 images/s (source: preregistered feasibility formula in `02-plan.md`).
- Candidate/declared-floor worst-rate ratio: 0.5062; parent/declared-floor worst-rate ratio: 0.9598, exposing gate miscalibration.
- Candidate/raw-demand worst-rate ratio: 0.6074; candidate best/raw-demand ratio: 0.7266. Candidate best was below parent worst, so feasibility result remains fail without relying on the flawed absolute floor.
- Primary metric: unavailable; no training or evaluation was launched.

## Verification Results

### Conditions Checked

- Scope/static/policy/RNG/clean-parent parity: pass; exact values and coverage are recorded in Run 1.
- Loader feasibility: fail. Worst candidate throughput 15,549.8 images/s is below the fixed 30,720 images/s headroom requirement.
- GPU integration, full-run integrity, and primary accuracy: skipped after the prior fixed-package feasibility failure.

### Informational Metrics

- No training metrics available.

## Errors & Dead Ends

### 2026-08-05 - Paired-view loader cannot feed the parent GPU path
- Error: `AssertionError: fixed paired-view loader fails preregistered 1.20x headroom gate`
- Root cause: the complete paired-view package, including augmentation, second-view materialization, serialization, IPC, and pinning, costs about 2.1x parent loader time; the preflight did not isolate those components.
- Source: Run 1 five-epoch preflight values above.
- Do NOT retry: do not tune RandAugment scalars, worker count, or cutoff on this node. A future augmentation design needs one-view phase control or a GPU-native batched implementation without losing the clean-tail invariant.

## Human Notes

> User requires physical GPU 0 and Claude adversarial review with no fallback. Claude completed the idea review; the first plan-review attempt timed out, and a second Claude Opus attempt completed. No other reviewer was used.
