# Report EXP-019: SWA with a constant-LR averaging tail (proper Stochastic Weight Averaging)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-019.md
- **Plan**: plans/plan-019.md
- **Log**: logs/exp-log-019.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%), higher-is-better, editing only `train.py` within the fixed 300s
training budget on a single H20. Baseline = **96.22%** (EXP-012, commit 6c417a4); success bar = **96.32%**
(+0.1pp). This experiment tested proper SWA — the one weight-averaging variant the goal-learnings explicitly
sanction (EXP-006 failed only because cosine-to-0 settled the iterate; SWA supplies the missing terminal-LR floor).

## Idea & Hypothesis
Chosen idea: replace the cosine-to-0 LR schedule with cosine→moderate-floor (0.05) over [5%, 75%] then a
CONSTANT 0.05 averaging tail for the final 25%; during the tail, average the weights once per epoch
(`torch.optim.swa_utils.AveragedModel`), recompute BN stats for the average (truncated 50-batch pass), and
evaluate the BN-recomputed SWA model (raw model still evaluated in the main phase → one eval/epoch). Selected
because SWA directly attacks the diagnosed binding constraint (generalization at fixed k=4 capacity) with
strong comparable-setting evidence (Izmailov 2018, ~0.5–1.3pp on CIFAR WRN/ResNet) at near-zero compute cost.
Hypothesis: the flat-region weight average lifts best_test_acc above 96.32 by generalizing better than the
single cosine-to-0 endpoint; main downside risk is a mild regression (~−0.3pp) if forgoing cosine-to-0
sharpening isn't recovered by averaging.

## Approach
Six train.py edits (no new deps — `swa_utils` is core torch): imported `AveragedModel`; added `SWA_START_FRAC=0.75`,
`SWA_LR=0.05`, `BN_RECOMPUTE_BATCHES=50`; rewrote `lr_at_fraction` (warmup → cosine PEAK_LR→SWA_LR over
[WARMUP_FRAC, SWA_START_FRAC] → constant SWA_LR, continuous at the join); added `recompute_bn()` (resets BN,
momentum=None cumulative average, 50 augmented training batches forward — channels_last+Cutout+bf16, matching
the training input distribution); constructed `swa_model = AveragedModel(model)` (eager, separate from the
compiled handle); and branched the per-epoch eval (tail → update_parameters + recompute_bn + eval SWA model;
main phase → eval raw model). Model input shape unchanged → torch.compile graphs unaffected. No deviations
from plan-019. Ruff clean; diff = train.py only.

## Execution
One run, no retries. Clean startup: `num_params 4,299,866` (UNCHANGED — SWA adds no params), clean compile, no
traceback, no NaN. LR schedule verified correct: decayed 0.20→0.05 then held EXACTLY `lr: 0.0500` constant
through the tail (ep 68–91) — the terminal-LR floor EXP-006 lacked. Tail fired at ep 68 (~75% of 91 epochs):
67 `[raw]` + 24 `[swa]` evals = 91 = num_epochs (one evaluate()/epoch, constraint satisfied). Throughput-neutral:
8ms/step, ~15,600 img/s, 91 epochs (identical to EXP-012's 91) — a fair same-budget test. Exited 0 in 421.1s < 600s.
peak VRAM 469.3 MB (≈ baseline 454 + ~17 MB SWA copy).

## Results
- **Primary metric**: best_test_acc = **95.97%** (baseline 96.22, delta **−0.25pp**, −0.26%) — a near-miss,
  below the 96.32 bar.
- **Observations**: The SWA MECHANISM ENGAGED EXACTLY AS THEORIZED. At the 0.05 floor the un-annealed raw
  iterate dropped to 91.83% (ep 67); the BN-recomputed weight average then recovered to 93.95% (ep 68) and
  climbed monotonically to 95.97% (ep 91) as snapshots accumulated — a ~+4pp lift over the raw iterate. The SWA
  model also produced the **LOWEST final_test_loss in the project: 0.1788** (vs baseline 0.195 / compiled-k4
  0.208), i.e. a genuinely flatter/better-calibrated minimum.
- **Analysis**: Hypothesis REFUTED on the metric, but instructively. SWA worked — it found a flatter, lower-loss
  solution — yet that did NOT convert to higher top-1 accuracy: it landed 0.25pp BELOW the cosine-to-0 endpoint.
  Two takeaways: (1) this VALIDATES the EXP-006 diagnosis — with a proper terminal-LR floor, weight averaging is
  no longer a no-op (it moved +4pp over the moving iterate and produced the project's lowest loss), so EXP-006's
  null was indeed caused by the cosine-to-0 schedule, not by averaging being useless here. (2) At this 300s/~91-epoch
  budget, the cosine-to-0 endpoint's final sharpening is worth slightly MORE top-1 accuracy than SWA's flat-region
  averaging — the trade nets −0.25pp. The lowest-loss-but-not-highest-acc pattern is a known SWA signature (flatter
  minima improve calibration/loss more reliably than top-1). The SWA curve was still inching up at ep 91
  (95.93→95.97 over the last epochs), suggesting it had nearly but not fully converged. This corroborates the
  standing diagnosis that the 96.22 plateau is generalization-bound at fixed k=4 capacity in 300s: even a
  flatter optimum doesn't beat the well-tuned cosine-to-0 recipe here.
- **Key Learning**: Proper SWA (constant-0.05-LR tail) engages correctly — recovers the un-annealed iterate
  (91.8→95.97) and yields the project's lowest test loss (0.1788) — but at the 300s/~91-ep budget it falls
  0.25pp short of the cosine-to-0 baseline: flat-region averaging trades slightly worse top-1 for better loss.

## Verification
- **Conditions**: Cond 1 (best_test_acc ≥ 96.32) **FAIL** (95.97, −0.25pp) — decisive. Cond 2 (clean completion
  < 600s, no traceback) PASS (recorded for completeness; total_seconds 421.1). Cond 3 (scope) PASS (train.py only,
  eval-count 91 == num_epochs, params 4,299,866 unchanged, no new deps — swa_utils is core torch, seed 42 intact).
- **Review Notes**: Results trustworthy — clean run, throughput-neutral fair test (params unchanged, 91 epochs ==
  EXP-012's 91), SWA path verified engaging (constant 0.0500 LR confirmed, monotone SWA-eval climb), scope intact.
  The elevated training loss (~0.87) in the tail is the expected constant-LR artifact, NOT divergence; judged on test acc.
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (primary metric did not clear the bar; −0.25pp below baseline).

## Unexplored Avenues
- **SWA hyperparameter tuning (near-miss refinement)**: only −0.25pp short and the SWA curve was still rising at
  ep 91. A LATER tail start (SWA_START_FRAC≈0.85, more cosine annealing toward a lower floor before a short
  averaging window) or a LOWER SWA_LR (≈0.02 — closer to annealed yet still moving) could let the average land
  on a sharper-yet-flat solution that exceeds the cosine-to-0 endpoint. Medium-effort, single-axis; the most
  promising remaining lead given how close this came.
- **Cyclic-LR SWA (SGDR-style sawtooth in the tail instead of constant 0.05)**: cyclic LR samples more diverse,
  more-separated minima for the average — the original SWA paper's stronger variant — possibly a better-spread
  average than the constant-LR tail. Untried; medium effort.
- **SWA-as-final-only (don't lose main-phase annealing)**: keep cosine-to-0 for the full budget but ALSO snapshot
  the last few epochs and average — but cosine-to-0 makes those snapshots ≈ the endpoint (the EXP-006 trap), so
  this needs at least a small terminal floor; effectively a milder version of the SWA_LR≈0.02 idea above.

## Next Steps
1. **SWA_LR / start-frac tuning** (medium confidence): this was a −0.25pp near-miss with the SWA curve still
   rising — sweep SWA_LR lower (≈0.02) and/or SWA_START_FRAC later (≈0.85) to trade less lost annealing for a
   sharper average. The single best-evidenced remaining lead.
2. **Cyclic-LR SWA tail** (low-medium confidence): replace the constant tail with a short cyclic LR to sample
   more diverse minima for the average (the paper's stronger variant). Reference knowledge/papers/swa.md.
3. **Accept the plateau** (medium confidence): ~12 axes now exhausted; even a flatter optimum (lowest project
   loss) didn't beat cosine-to-0. If SWA tuning (step 1) also fails, the 96.22 plateau is well-established as
   generalization-bound at fixed k=4 capacity in 300s. Reference project-insights High "generalization-bound at fixed capacity".

## Exit Action Results
- No exit actions defined for this goal — skipped.
