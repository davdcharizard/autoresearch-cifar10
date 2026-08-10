# EXP-019: Balanced Mixup and CutMix Geometry

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-019
- **Commit**: (pending - committed on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Modified only tracked `train.py` from frontier commit `7c1e7d8`. The strong collator now uses one forked CPU RNG draw to choose 25% alpha-1 CutMix, 25% alpha-0.4 Mixup, or 50% hard batches and returns an integer provenance. The strong loop conditionally unpacks that three-tuple, validates target form from provenance, and counts all three geometries; the rebuilt weak loader retains its two-item hard-label contract. Model, optimizer, schedule, timer, evaluator, and total mixed-batch probability remain unchanged.

### Surprises & Discoveries

Torchvision 2.9.1 MixUp pairs each batch with `roll(1, 0)` and uses a torch-CPU Beta draw, matching the planned same-process RNG containment. External Claude plan review emphasized that this is a compound geometry-and-strength bet rather than an equal-regularization comparison; a valid miss will therefore not isolate alpha, split, and geometry effects.

The paired safety controller's first attempt asserted after a candidate-only class concentration above 95% but before serializing the step histogram. A serialization-only correction reran the same code and passed, but fresh-process forkserver scheduling produced a visibly different augmented trajectory (control/candidate loss EMAs changed), because the controller streamed real source batches instead of persisting them. The second result therefore cannot override the first pre-registered veto. Timing and production were blocked.

### Decisions

The old `targets.ndim == 2` heuristic was removed completely from geometry counting because both CutMix and Mixup produce `[B,10]` targets. Integer provenance is used for attribution, while target dimensionality and dtype are validation-only. The alpha-0.4 refinement was retained exactly as externally reviewed. The safety threshold was not relaxed, and the passing non-identical retry was treated as diagnostic only rather than a fallback authorization.

## Experimental Adjustments

- **Blocked production after the first collapse veto**: The corrected controller retry did not replay identical materialized augmented batches, so its pass could not erase the registered attempt-1 failure. (ref: Run 1 observations and Errors & Dead Ends)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A - production blocked by preflight
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log`
- **WandB**: N/A
- **Status**: failed preflight; production not launched
- **Started**: 2026-08-06
- **Ended**: 2026-08-06

Description:
- Sole seed-42 production run of the reviewed 50/25/25 hard/CutMix/Mixup strong-phase policy. It will run only if deterministic semantics, 20,000-collation proportions, paired 200-real-batch safety, worker lifecycle, and five-pair timing gates all pass. Expected success is `best_test_acc >=94.25%` with at least 26,629 steps and every fixed-budget integrity condition satisfied.

Observations:
- Static preflight passed: `py_compile`, Ruff 0.15.6, `git diff --check`, and source inspection all exit zero; only tracked `train.py` is modified and preserved untracked `data/` remains. (source: execution command output before production)
- Deterministic semantics passed: all categorical branches, accepted hard/CutMix bitwise equivalence, CPU/CUDA RNG restoration, alpha-0.4 pixel/target pairing, finite gradients/momentum, exact 1,073,962 parameters, and Beta central-mass checks. (source: `/tmp/exp019_semantics.py` output, `SEMANTICS_PASS`)
- The 20,000-collation forkserver gate passed with hard 10,059 (50.295%), CutMix 5,038 (25.190%), Mixup 4,903 (24.515%), total mixed 49.705%, eight strong workers stopped, 2.953-second weak rebuild, hard weak labels, and eight weak workers stopped. (source: `/tmp/exp019_loader_gate.py` output, `LOADER_PASS`)
- Paired safety attempt 1 stayed finite through 200 batches and ended with candidate/control loss-EMA ratio 0.981399, but asserted because at least one step had candidate concentration above 95% while control was at or below 95%. The assertion preceded histogram output. (source: `/tmp/exp019_safety_gate.py` attempt 1 traceback)
- A serialization-only rerun passed with 105 hard, 47 CutMix, 48 Mixup, zero mixed-decision mismatches, maximum/final loss-EMA ratios 1.026892/0.962090, and zero concentration events. Its different loss trajectory proves the real augmented source sequence was not identical across fresh forkserver processes, so it is not a valid replay or clearance. (source: `/tmp/exp019_safety_gate.py` attempt 2 `SAFETY_RESULT`)
- Timing, exposure, and production accuracy were skipped immediately after concluding that the original registered veto remained binding. No `run.log` was created.

Key Metrics:
- Static and semantic gates: **passed**.
- 20,000-collation/lifecycle gate: **passed** - 50.295% hard, 25.190% CutMix, 24.515% Mixup; weak rebuild 2.953s.
- Paired real-batch collapse gate: **failed** - attempt 1 triggered candidate-only concentration above 95%.
- Production accuracy: **not measured** - no production run was authorized.

## Verification Results

### Conditions Checked

- Baseline/scope/source: **passed** - baseline 94.15 at `7c1e7d8`, only tracked `train.py` modified, syntax/Ruff/diff clean.
- Deterministic semantics and target/RNG integrity: **passed** - `SEMANTICS_PASS`.
- Forkserver proportions and lifecycle: **passed** - all registered intervals and worker/weak-loader gates passed.
- Paired 200-real-batch safety: **failed** - candidate-only greater-than-95% class concentration on attempt 1.
- Timing/exposure: **skipped - aborted after safety failure**.
- Production completion/metric: **skipped - production not launched**.

### Informational Metrics

- No production metrics; `run.log` was never created.

## Errors & Dead Ends

### 2026-08-06 - Safety controller failed to serialize concentration evidence before assertion
- Error: `AssertionError: candidate_only_concentration == 0` after 200 paired real batches.
- Root cause: the registered candidate-only concentration veto fired, but the disposable controller asserted before printing the stored step and paired histograms.
- Source: `/tmp/exp019_safety_gate.py` attempt 1 traceback; final loss-EMA ratio 0.981399.
- Do NOT retry: do not place collapse assertions before result serialization; persist source batches and failure evidence first.

### 2026-08-06 - Fresh forkserver retry did not reproduce the augmented source trajectory
- Error: attempt 2 passed the veto but had different control/candidate loss EMAs from attempt 1.
- Root cause: the controller streamed transformed batches from fresh forkserver workers; process scheduling changed which worker RNG stream produced each batch, so seed 42 did not make the cross-process augmentation trajectory identical.
- Source: `/tmp/exp019_safety_gate.py` attempts 1 and 2 outputs.
- Do NOT retry: do not use a fresh-process pass to override this failure; any future replayable safety controller must persist the exact post-transform source batches or their full tensors before paired training.

## Human Notes

> Autopilot; no human intervention during execution.
