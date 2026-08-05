# EXP-019: Static First-Block Scale Plus Final SE

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-019
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - local-only run
- **Outcome**: failed - valid run scored below baseline and acceptance threshold

## Implementation Notes

### Summary

Added a no-decay, exact-one 128-channel residual scale to `layer3[0]` and the established exact-neutral ratio-16 SE gate to `layer3[1]`. The accepted WRN is initialized first; the final gate is then initialized from fixed seed 42 inside a CPU-only restored RNG fork. Only terminal parameter statistics observe the static scale.

### Surprises & Discoveries

Plan review identified that reusing EXP-017's near-positive seed 17017 was avoidable initialization selection. The implementation instead exactly matches EXP-018's fixed-seed-42 final gate, making the new first-block scale the only added mechanism relative to that scored treatment.

### Decisions

The static scale is stored as a 1D parameter so the accepted optimizer rule places it in the no-decay group. It begins at exact identity rather than the post-hoc 0.65 mean; terminal statistics distinguish whether it learned attenuation without changing the treatment.

## Experimental Adjustments

- **Use fixed seed 42 for final SE**: avoids known-score seed selection and preserves the no-reroll contract. (ref: `02-plan-review.md` concern 4)

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 60434
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed (research no-improvement)
- **Started**: 2026-07-26 18:04:43 UTC
- **Ended**: 2026-07-26 18:11:13 UTC

Description:
- One fixed-seed local H20 run will test whether cheap learned first-block attenuation restores the two-stage attention signal lost in EXP-018 while retaining its conditional final gate and exposure. Success requires at least 94.17% with accepted training and evaluation behavior unchanged. Terminal scale statistics are explanatory only; a valid score will not be rerun or tuned.

Observations:
- Production scope/compile audit passed: `train.py` is the sole changed tracked production file and the diff is 39 insertions. (source: local audit stdout)
- Semantic preflight passed exact hybrid placement, 693,986 parameters, fixed-seed-42 oracle, accepted common state/logits/RNG, shortcuts, optimizer groups, and gradient opening. (source: semantic preflight stdout)
- Matched timing passed: accepted/candidate weighted steps were 12.552926/12.777486 ms, retention 0.982425, worst CV 0.003860, and finite informational projection 120.21 passes. (source: throughput preflight stdout)
- The scored run completed once with exit 0 on one H20, exact 693,986 parameters, 28 unique evaluations, and no error signature. (source: `run.log` startup/final summary)
- Mixup disabled exactly once at epoch 89, step 17,174 and 195.0 seconds. (source: `run.log` transition line)
- The static vector learned strong attenuation: mean 0.674833, std 0.151918, range 0.427161-1.095004. This closely reproduces EXP-017 gate 0's 0.6468 global mean but does not recover its score. (source: `run.log` terminal scale summary)

Key Metrics:
- best_test_acc: 93.86% at final epoch 138, -0.21 from baseline and -0.31 below threshold. (source: `run.log` final evaluation/summary)
- final_test_acc/loss: 93.86% / 0.2348. (source: `run.log` final summary)
- timing/exposure: 300.0 training seconds, 340.3 total seconds, 26,758 steps = 137.00096 passes. (source: `run.log` final summary)
- resources/model: 1,094.0 MiB peak VRAM and 693,986 parameters. (source: `run.log` final summary)

## Verification Results

### Conditions Checked

- **Completion and integrity**: PASS - exit 0, one H20, correct count, 300.0 counted and 340.3 total seconds, one transition, 28 unique evaluations, finite scale summary, and no errors. (source: `run.log`)
- **Primary metric >=94.17%**: FAIL - best 93.86%, 0.21 below accepted baseline and 0.31 below threshold. Verification stopped on this necessary-condition failure. (source: `run.log` final summary and results index)
- **Remaining conditions**: skipped after metric failure; pre-score scope, semantics, throughput, and diff checks passed.

### Informational Metrics

- Skipped under the verification guard; run and scale values are preserved under Run 1 Key Metrics and Observations.

## Errors & Dead Ends

## Human Notes

> Autopilot local-only execution; no intervention requested.
