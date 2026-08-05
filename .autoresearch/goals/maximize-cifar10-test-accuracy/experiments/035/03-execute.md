# EXP-035: Weaker Alpha-0.1 Batch-Shared Mixup

## Execution

Overall Status & Info:
- **Created**: 2026-07-27
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-035
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - offline/local-only session
- **Outcome**: failed - valid result below margin

## Implementation Notes

### Summary

Changed only `MIXUP_ALPHA` from 0.2 to 0.1 in production. Added an ignored,
evaluator-free preflight that imports `67c8e98:train.py` independently, proves
the exact source scope and construction identity, validates the candidate Beta
law and replay semantics, checks hard-path and cutoff identity, and measures
counterbalanced full-step H20 exposure before allowing the sole score.

### Surprises & Discoveries

The production implementation already isolates the treatment to a single
constant. The fixed seed does not imply an accepted/candidate trajectory match:
concentration-dependent gamma rejection intentionally changes later CUDA draws.

### Decisions

The preflight treats candidate self-replay as the determinism requirement and
accepted/candidate hard-label replay as the exact identity requirement. It does
not restore or align CUDA state between alpha-0.2 and alpha-0.1 mixed steps.

## Experimental Adjustments

None.

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 22968
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed (exit 0)
- **Started**: 2026-07-27 01:28 UTC
- **Ended**: 2026-07-27 01:34 UTC

Description:
- The sole fixed-seed alpha-0.1 score will run locally on one H20 only after
  semantic and complete-body timing gates pass. It tests whether weaker early
  batch-shared interpolation improves the accepted deeper-plus-RandAugment
  recipe while retaining at least 130 projected passes. The primary success
  threshold is 94.42% best test accuracy.

Observations:

- Semantic/distribution preflight passed: alpha-0.1 mean 0.499486,
  variance 0.208358, central mass 0.120770, endpoint mass 0.813190,
  exact replay/hard-path checks, and 1,100.1 MiB peak all passed (source:
  local preflight stdout, 2026-07-27 01:27 UTC).
- Timing preflight passed with retention 1.003557, projected 133.4805 passes,
  hard-path ratio 0.999981, maximum CV 0.007114, and 1,096.3 MiB peak
  (source: local preflight stdout, 2026-07-27 01:28 UTC).
- The sole score completed with one finite summary and no numerical, CUDA,
  worker, or evaluator error. Mixup disabled at step 16,723/195.0 s and
  RandAugment disabled at the epoch-86 iterator boundary at step 16,770/195.5
  s, a valid 47-step lag (source: `run.log` lines 40-42).
- Evaluations occurred once at every fifth epoch through 130 plus final epoch
  134; all 27 evaluation epochs were unique (source: `run.log` lines 6-62).

Key Metrics:

- **Score**: best/final 93.72%/93.72%; best delta -0.60 points from the
  94.32% baseline and -0.70 from the 94.42% threshold; final delta -0.50
  points from accepted 94.22% (source: `run.log` lines 64-65).
- **Final loss**: 0.2770, +0.0247 versus accepted 0.2523; best-final gap
  0.00 points (source: `run.log` lines 64-66).
- **Execution**: 26,012 steps, 134 epochs, 133.18144 passes, 300.0 counted /
  342.8 total / 1.1 startup seconds (source: `run.log` lines 67-72).
- **Resources/model**: 1,096.3 MiB peak VRAM and 987,098 parameters (source:
  `run.log` lines 70-73).
- **Exposure projection**: 133.48049 passes projected versus 133.18144
  realized, a -0.29905-pass residual; realized exposure remained above the
  protected 130-pass regime.

## Verification Results

### Conditions Checked

- **Run integrity**: PASS - exit 0, one H20, one finite summary, 300.0 counted
  and 342.8 total seconds, 133.18144 passes, exact topology/scope, correct
  temporal transitions, and 27 unique source-faithful evaluations (source:
  `run.log`; final source audit).
- **Primary metric**: FAIL - `best_test_acc=93.72%` is 0.60 points below the
  94.32% baseline and 0.70 below required 94.42%. No rerun (source: `run.log`
  lines 64-65).
- **Mechanism corroboration**: FAIL - `final_test_acc=93.72%` is below the
  preregistered 94.32%, and final loss 0.2770 is worse than 0.2523 (source:
  `run.log` lines 65-66).

### Informational Metrics

- Skipped as a formal success-only collection after the primary condition
  failed; all execution values required for analysis are preserved above.

## Errors & Dead Ends

## Human Notes

> Autopilot session; no execution-time intervention requested.
