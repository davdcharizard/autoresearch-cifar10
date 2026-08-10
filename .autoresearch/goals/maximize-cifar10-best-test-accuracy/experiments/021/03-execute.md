# EXP-021: Deterministic Pool-First Option-A Shortcuts

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-021
- **Commit**: (pending — committed on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Created the experiment branch from the accepted `7c1e7d8` frontier and changed only `BasicBlock.forward` in `train.py`: the two runtime transition shortcuts now use an explicit fixed 2x2 stride-2 average pool before the unchanged Option-A zero channel pad. All model, data, optimizer, schedule, timer, evaluator, seed, and logging code remains accepted EXP-010 code. The initial static gates passed with 1,073,962 parameters, 116 state-dict entries, 59 optimizer tensors, identical reset-seed state/RNG, and clean Ruff/pre-commit output.

### Surprises & Discoveries

No production-code surprise occurred. The approved edit is structurally simpler than EXP-017 because it creates no module, parameter, buffer, constructor draw, or shortcut BatchNorm state. External plan review did identify two controller hazards before implementation: path-launched ignored scripts need an explicit project-root `sys.path`, and Python 3.14 forkserver creation must remain behind the main guard.

### Decisions

- Kept the pool arguments explicit (`kernel_size=2`, `stride=2`, zero padding, floor output, no pad contribution) so the reviewed spatial rule is mechanically auditable.
- Retained the following `F.pad` byte-for-byte, preserving original channel provenance and zero-filled new channels.
- Production exposure below 26,360 will be an attribution caveat rather than a validity veto; the pre-launch paired timing projection must still meet the floor.
- No evaluator cadence change is permitted. The timing projection must predict exactly 19 looks; an actual shortfall is recorded as a deflation caveat, while more than 19 is an integrity failure.

## Experimental Adjustments

- **Stopped before timing and production**: The immutable-corpus safety controller triggered the pre-registered candidate-only >95% class-concentration veto at steps 17 and 18. No corpus regeneration, threshold relaxation, timing run, or candidate variant was attempted. (ref: `preflight-safety.json` lines 30-84 and 110-124)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (production vetoed in local preflight)
- **Log file(s)**: `preflight-semantics.json`, `preflight-corpus.json`, `preflight-safety.json`, `preflight-safety-control.json`, `preflight-safety-candidate.json`; no `run.log` was created
- **WandB**: N/A
- **Status**: failed preflight
- **Started**: 2026-08-06 13:05 UTC
- **Ended**: 2026-08-06 13:07 UTC

Description:
- One fixed-seed production run of deterministic pool-first Option-A shortcuts on the accepted width-2/N1-M7/CutMix recipe, contingent on semantic, replayable safety, paired timing, exposure, and evaluator-parity gates. The most likely accuracy outcome is no-improvement, but the registered switch/first-weak/NLL signature can isolate whether pooling caused EXP-017's late-generalization harm. Formal improvement requires at least 94.25% with all protocol-integrity conditions intact.

Observations:
- Static and semantic gates passed: exactly two even-shape stride-2 transition pools, exact values/0.25 gradients/zero padding, both finite target paths, 1,073,962 parameters, and aligned construction state/RNG. (source: `preflight-semantics.json`)
- The immutable production corpus contained 200 batches, exactly 100 hard and 100 CutMix probability targets, SHA-256 `9b2801ac6d2d07f9bc5a5204e370db815f1b8bfaca8a1b9848a97091703388ea`; all eight worker PIDs stopped. (source: `preflight-corpus.json`)
- Candidate and control began from identical state hash `2c8e8a2f...a7052c`, identical RNG hash `81ef85d9...26d0`, the same corpus, and identical deterministic backend settings. All 200 steps stayed finite in both arms. (source: `preflight-safety.json` lines 15-24, 86-108, 110-121)
- Candidate concentrated 123/128 predictions in class 5 at step 17 and 128/128 at step 18; control was 113/128 and 112/128 on those exact batches. This crossed the registered candidate-only >95% veto twice. (source: `preflight-safety.json` lines 30-84)
- Candidate terminal loss EMA was lower, 2.11687 versus control 2.24978 (ratio 0.940923), showing the veto was a prediction-collapse transient rather than non-finite loss or broad optimizer failure. Timing and production were skipped. (source: `preflight-safety.json` lines 15-24, 99-108, 123-124)

Key Metrics:
- `best_test_acc`: NaN / not run (production vetoed)
- safety concentration: candidate 96.09375% vs control 88.28125% @ step 17; candidate 100% vs control 87.5% @ step 18 (source: `preflight-safety.json` lines 30-84)
- terminal loss EMA ratio: 0.9409227 candidate/control after 200 exact batches (source: `preflight-safety.json` line 124)
- maximum gradient norm: candidate 12.66018, control 14.21516; maximum update norm: candidate 1.70546, control 1.84540 (source: `preflight-safety.json` lines 15-24 and 99-108)

## Verification Results

### Conditions Checked

- Semantic integrity: passed (source: `preflight-semantics.json`).
- Replayable production-batch safety: failed candidate-only class-concentration condition at steps 17 and 18 (source: `preflight-safety.json` lines 30-84 and 110-124).
- Timing/exposure/inference/evaluation projection: skipped after safety failure.
- Fixed-budget completion and `best_test_acc>=94.25%`: skipped because production was not authorized.

### Informational Metrics

- Corpus: 200 batches, 100 hard/100 mixed, 315,301,733 bytes, SHA-256 `9b2801ac...388ea`; 8/8 workers stopped (source: `preflight-corpus.json`).
- Safety: all 200 steps finite in both arms; EMA ratio 0.940923; candidate-only concentration failures at steps 17 and 18 (source: `preflight-safety.json`).

## Errors & Dead Ends

### 2026-08-06 — Deterministic pool-first Option-A hit candidate-only class concentration
- Error: `SAFETY_GATE_FAIL`: candidate predicted class 5 for 123/128 examples at step 17 and 128/128 at step 18, while control remained below 95%.
- Root cause: The deterministic average-pooled transition shortcuts changed the early optimization trajectory toward a transient single-class basin on two exact production batches; state, RNG, corpus, backend, and finiteness controls all aligned.
- Source: `preflight-safety.json` lines 15-24, 30-84, and 99-124; detailed records in `preflight-safety-control.json` and `preflight-safety-candidate.json`.
- Do NOT retry: do not regenerate the corpus, relax the >95% candidate-only veto, run production, or test a pool-size/one-transition/projection rescue under EXP-021.

## Human Notes

> Autopilot continuation requested. External Claude idea and plan reviews both completed with exit code 0; no fallback reviewer was used.
