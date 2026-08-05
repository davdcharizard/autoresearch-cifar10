# Proposal: Zero-Floor Cosine for the Hard-Label Tail

## Summary

Change only `MIN_LR` from `0.002` to `0.0` in the accepted EXP-002 recipe.
Keep WRN-16-2, batch 256, peak LR 0.2, the 5% time warmup, Nesterov SGD,
selective weight decay, alpha-0.2 mixup through 65% counted time, and every-fifth-
epoch evaluation unchanged. This converts the existing bounded cosine into a
canonical cosine-to-zero schedule and tests whether residual SGD motion at the
end of the validated hard-label phase is preventing the model from settling.

This is an unusually clean fixed-budget intervention: it changes one scalar,
adds no operations, preserves the stochastic-operation sequence, and targets a
schedule parameter that has never been isolated. The prior `0.002` floor was a
reasonable conservative choice in the successful EXP-001 bundle, but it was
not itself validated. Zero is preferable to an arbitrary intermediate value
such as `0.0005`: it is an exact, interpretable endpoint and produces the
strongest falsifiable test of the residual-update hypothesis.

## Limiter Diagnosis

The accepted path has already found a strong architecture and regularization
balance. EXP-001's WRN-16-2 plus a time-aligned schedule reached 93.38%, and
EXP-002's early alpha-0.2 mixup followed by a 35% hard-label tail reached
94.07%. Three subsequent input-regularization variants regressed at normal
exposure: CutMix scored 93.72%, ending mixup at 50% scored 93.91%, and alpha-0.4
mixup scored 93.57% with higher test loss. Another nearby augmentation change
is therefore a weaker bet than refining the accepted optimization path.

EXP-002 finished at its best score, with final test loss 0.2432, after 27,735
steps and 141.9 dataset-equivalent passes. Its accuracy continued improving in
the hard-label tail. That supports two competing interpretations:

- the tail is still making useful low-LR progress, in which case retaining a
  nonzero floor helps and cosine-to-zero may freeze the model too early; or
- the representation and margins are already learned, while the 0.002 floor
  sustains late stochastic motion that prevents the final iterate from settling
  more precisely, in which case a zero endpoint should improve final loss and
  accuracy.

This experiment cleanly distinguishes those interpretations. It does not claim
that the accepted endpoint exhibits visible instability: final equaled best,
so the plausible gain is modest and the 0.10-point acceptance threshold is a
meaningful hurdle.

## How Much of the Trajectory Changes

For post-warmup progress `p`, let
`q = (p - 0.05) / 0.95` and
`c(q) = 0.5 * (1 + cos(pi * q))`. The accepted and proposed schedules are:

```text
accepted: 0.002 + 0.198 * c(q)
proposed:         0.200 * c(q)
difference: 0.002 * (1 - c(q))
```

Thus the change is negligible in the high-LR portion and grows smoothly toward
the endpoint:

| Counted progress | Regime | LR at floor 0.002 | LR at floor 0 | Reduction |
|---:|---|---:|---:|---:|
| 25% | mixup | 0.179125 | 0.178914 | 0.1% |
| 50% | mixup | 0.109175 | 0.108258 | 0.8% |
| 65% | hard-label switch | 0.061232 | 0.059830 | 2.3% |
| 75% | hard labels | 0.033949 | 0.032272 | 4.9% |
| 80% | hard labels | 0.022875 | 0.021086 | 7.8% |
| 90% | hard labels | 0.007364 | 0.005418 | 26.4% |
| 95% | hard labels | 0.003350 | 0.001364 | 59.3% |
| 99% | hard labels | 0.002054 | 0.000055 | 97.3% |
| 100% | hard labels | 0.002000 | 0.000000 | 100.0% |

The area under the LR curve falls only about 0.99% over the full 300 seconds,
but 7.9% over the 65-100% hard-label phase and 52.2% over the final 10%. At
EXP-002 throughput, the hard-label phase contains about 9,945 optimizer steps;
the final 10% contains about 2,774 steps and the final 5% about 1,387. The
proposal therefore preserves almost all representation learning and primarily
reduces the amplitude of the final several thousand Nesterov updates.

`MIN_LR` currently serves both as the cosine floor and the warmup start. The
one-line change consequently starts the first step at LR zero and uses a
`0 -> 0.2` rather than `0.002 -> 0.2` warmup during the first 15 counted seconds.
The absolute difference starts at 0.002, decreases linearly to zero by 5%, and
changes warmup LR area by about 1%. The first zero-LR SGD step still initializes
its momentum buffer but does not move parameters. This is a real secondary
effect, not an exactly late-only intervention, but it is tiny relative to the
peak and affects roughly one of 27,735 steps maximally. Adding a second
`WARMUP_START_LR` constant would preserve the early path more exactly at the
cost of a broader code change; the single scalar is preferable for EXP-006 as
long as the warmup effect is reported honestly.

## Falsifiable Hypothesis

After alpha-0.2 mixup shapes the representation and the first part of the
hard-label phase restores clean margins, annealing all the way to zero will
reduce unproductive late Nesterov motion and allow the endpoint to settle into
a slightly better classifier. With unchanged exposure, this should raise
`best_test_acc` from 94.07% to at least the required **94.17%**, with final
accuracy at or near the best and final test loss no higher than 0.2432.

The mechanism is falsified by a valid run below 94.17%. More specifically:

- If accuracy peaks earlier and then plateaus or regresses while the final LR
  approaches zero, the residual floor was supporting useful continued progress;
  do not tune a smaller nonzero floor based on the same run.
- If final test loss improves but accuracy remains below 94.17%, zero-floor
  annealing may improve confidence calibration but does not advance the primary
  objective and is still a no-improvement verdict.
- If both final accuracy and loss worsen at normal exposure, the hard-label tail
  is under-updating or the small warmup-start change is harmful; retain 0.002.
- If score and loss are essentially unchanged, the floor is not a current
  limiter and future loops should move to an orthogonal mechanism.

One fixed-seed result is sufficient for the loop's decision. Do not reroll or
repeat a near miss.

## Exact Code Scope and Determinism

Starting from accepted commit `eb08811`, change exactly one line in `train.py`:

```python
MIN_LR = 0.0
```

Do not modify the `learning_rate` function or any other constant. In
particular, preserve `LR = 0.2`, `WARMUP_FRACTION = 0.05`,
`MIXUP_ALPHA = 0.2`, `MIXUP_END_FRACTION = 0.65`, batch 256, the model,
optimizer, selective decay, transforms, loader, seed 42, evaluator, and logging.

Unlike changing beta concentration, dropout, or augmentation logic, this
change adds no random draw and changes no RNG-consuming branch. Mixup samples,
permutations, data-loader ordering, and crop/flip draw counts should follow the
same fixed-seed sequence as the accepted implementation. Parameter values
diverge immediately because the LR is different, as intended, but stochastic
operation order is unchanged. No claim of bitwise identity across separate CUDA
runs is necessary; the treatment is the deterministic schedule scalar rather
than a seed change.

The scalar arithmetic has no meaningful throughput or memory cost. Expected
exposure is near EXP-002's 27,735 steps and 141.9 passes, with the unchanged
691,674 parameters and roughly 1,094 MiB peak allocation. A material exposure
change would be an execution anomaly rather than part of the mechanism.

## Preflight and Full-Run Verification

No throughput benchmark is needed because the operation graph is unchanged.
Before the single scored run:

1. Confirm one NVIDIA H20, a clean accepted base, and a diff containing only
   `MIN_LR = 0.0`; confirm `prepare.py` is untouched.
2. Evaluate `learning_rate` at progress 0, 5%, 65%, 90%, 95%, and 100% in a
   non-training smoke check. Require approximately 0, 0.2, 0.05983, 0.005418,
   0.001364, and 0 respectively.
3. Remove stale `run.log`, then run exactly once with
   `timeout 600s uv run train.py > run.log 2>&1`.
4. Require exit code 0, one H20, finite loss, approximately 300 counted training
   seconds, no more than 600 total seconds, and a complete final summary.
5. Confirm mixup disables exactly once near 195 counted seconds and evaluation
   occurs at most once per epoch. The transition LR should be about 0.0598,
   rather than EXP-002's 0.0612.
6. Record best/final accuracy, final test loss, steps, epochs, passes, VRAM,
   parameters, transition time/LR, and final logged LR. Compare exposure against
   EXP-002 before interpreting the mechanism.

Success requires `best_test_acc >= 94.17%`; a lower score is not rescued by a
lower test loss or by proximity to the threshold.

## Risks

- **Premature freezing:** The accepted run was still improving during the
  hard-label phase. Although LR remains above 0.005 until 90% progress, the
  final 30 seconds receive about half the accepted LR area and may lose useful
  margin refinement.
- **Momentum does not eliminate the freeze:** Nesterov retains a direction in
  its momentum buffer, but the learning rate scales the applied update. As LR
  approaches zero, buffered motion is also suppressed.
- **Small available headroom:** The floor affects only the late trajectory and
  EXP-002 already ends stably at its best. The effect may be below the required
  0.10 points even if the mechanism is directionally correct.
- **Warmup coupling:** The one-line constant also lowers the warmup start. This
  effect is explicitly bounded above and should not be retroactively invoked to
  justify retrying a negative result with extra changes.
- **Sparse observation:** Every-fifth-epoch evaluation may not locate an exact
  transient peak, but changing cadence would violate clean comparison and is
  forbidden for this proposal.

## EXP-006 Recommendation

This is a **medium-confidence, high-cleanliness EXP-006 candidate**. It is more
attractive now than another mixup/CutMix variant because three such changes
have regressed, and it is more feasible than WRN-16-3 after that candidate's
local preflight projected only about 80.6 passes. It directly follows EXP-001's
unexplored recommendation to isolate the schedule and has essentially zero
engineering, throughput, VRAM, or constraint risk.

Its expected ceiling is nevertheless modest. Final-equals-best does not show a
noisy endpoint, and continued hard-label improvement is evidence that a
nonzero floor may still be productive. A credible low-overhead orthogonal
candidate with a stronger generalization mechanism can rank ahead of it. If
the EXP-006 selection prioritizes attribution, constraint safety, and learning
value after the width preflight failure, choose this proposal; if it prioritizes
maximum upside despite added implementation risk, weak early block dropout may
be the higher-variance alternative. Do not combine the floor change with
dropout, EMA, width, or a mixup timing change.

## Evidence

- `experiments/001/04-analysis.md`: WRN-16-2 and the bundled time-aligned
  `0.2 -> 0.002` cosine schedule improved 91.54% to 93.38%; the report explicitly
  leaves peak LR and decay-floor isolation unexplored.
- `experiments/002/04-analysis.md` and `03-execute.md`: the accepted alpha-0.2
  recipe reached 94.07%, final equaled best, final loss was 0.2432, mixup ended
  at 65% with LR 0.0612, and the run completed 27,735 steps / 141.9 passes.
- `03-experiment-learnings.md` and `04-results.tsv`: three later augmentation
  changes regressed at normal exposure, while the accepted time-cosine schedule
  remains the reusable pattern.
- `train.py`: `MIN_LR` currently controls optimizer initialization, warmup
  start, and the post-warmup cosine endpoint; schedule progress is based on the
  frozen 300-second counted training budget.
