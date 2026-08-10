# Adversarial result review: EXP-011

**Reviewer**: Claude Opus via Claude Code 2.1.220
**Reviewed**: 2026-08-06, against the raw `run.log`, goal, plan, execution record, and `train.py` diff

## Verdict

> The result is trustworthy. `best_test_acc = 95.61` is a valid formal improvement over EXP-004 (95.40) under the +0.10 rule. No blocking concerns remain.

Claude classified EXP-011 as `improvement`: 95.61 clears the formal 95.50 threshold, while the 95.70 stretch target was not reached. No invalidity or mechanism-dose condition was met.

## Independent recomputation

- `EMA_HALF_LIFE_S=(1-0.75)*300/4=18.75`.
- First/last cadence steps are `20863=31*673` and `25792=31*832`, giving exactly `832-673+1=160` updates and 159 consecutive distances.
- First due cadence after SAM begins at step 20,858 is step 20,863. First/last progress, 74.7736-second span, 0.063025 oldest coefficient, interval/decay ranges, and exact 80/80 parity all reconcile.
- CutMix eligibility 20,857 and SAM eligibility `25,798-20,858+1=4,941` reconcile; 2,471 is exactly the number of even SAM steps in that range.
- There are 133 sequential epoch evaluations with no gaps: 106 live + 27 EMA. Epoch 107 contains the first EMA sample and is correctly the first EMA-source evaluation.
- 27 swaps produced 27 exact restore checks and zero restore, coverage, nonfinite, or RNG failures. The inventory matches 13 BatchNorm modules, 44 parameter tensors, and 2,748,890 elements.
- All fixed gates pass: 300.0 charged seconds, 447.9 total seconds, 25,798 steps, 160 EMA updates, finite nonzero distances, complete summary, and only `train.py` changed.
- Preflight and realized values reconcile: projected/realized steps 25,570/25,798, total seconds 458.2/447.9, and peak MiB 1,277.0/1,222.4.

## Non-blocking concerns

1. The supported claim is that the cadence-31 charged-time EMA configuration reached 95.61 versus the parent's 95.40 under the preregistered protocol. EMA-only tail evaluation, n=1 fixed-seed measurement, and 25,798 versus 25,560 realized steps do not support assigning the entire +0.21 to EMA alone.
2. Epochs 118-133 form a 95.44-95.61 plateau with mean 95.493125; the formal 95.61 best is a normal maximum within that plateau, while final accuracy is 95.46. Descendants should retain plateau and final metrics rather than interpreting 95.61 as a stable level.
3. The inherited timer excludes data-loader wait before `t0`. This applies to parent and candidate and does not invalidate this comparison, but future input-pipeline changes could exploit that inherited accounting.
4. An invalid final-audit run now prints a complete summary before raising, so descendants must check exit code and `ema_audit_failed`, not only grep the metric line.

## Corrections to review text

- Claude referred to transient files as already deleted; at review time `run.log`, the preflight log, and smoke harness still existed and were supplied or locally present. They are deleted only after this review and durable transcription.
- Claude described diff scope as unverifiable in one concern, but the prompt supplied `git diff 1a8d0de -- train.py`, and an independent post-run `git diff --name-only 1a8d0de` returned only `train.py`.

These corrections do not change the reviewer verdict.
