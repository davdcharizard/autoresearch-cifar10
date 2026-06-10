# Report EXP-016: LR-schedule micro-tuning — raise peak LR 0.2 → 0.3 on the TA+Cutout recipe
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-016.md
- **Plan**: plans/plan-016.md
- **Log**: logs/exp-log-016.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%), higher-is-better, editing only `train.py` within the fixed 300s
training budget on a single H20. Baseline = **96.22%** (EXP-012, commit 6c417a4); success bar = **96.32%**
(+0.1pp). This experiment tested whether the peak learning rate — never re-tuned on the current
TA+Cutout recipe — is suboptimally low.

## Idea & Hypothesis
Chosen idea: raise `PEAK_LR` 0.2 → 0.3 (keeping the 5% linear warmup → cosine-to-0 shape). Selected because
the LR schedule is the highest-value knob never tuned on the current recipe (only weight-decay was swept,
EXP-005, on the old recipe), and it is compute-free → a perfectly fair same-budget test (no throughput
confound, the issue that muddied EXP-015). Hypothesis: peak 0.2 was a pre-widen/pre-augmentation heuristic
(EXP-000), and the heavy TA+Cutout regularization should let a more aggressive peak explore more before
annealing → flatter, better-generalizing minimum → best_test_acc above 96.32 (expected ~96.3–96.5), with
final_test_loss ≤ 0.195. If acc ≤ baseline, the peak is already at/above optimal.

## Approach
Single `train.py` edit (line 23): `PEAK_LR = 0.2` → `0.3`, with an inline comment noting the retune.
`PEAK_LR` feeds both `lr_at_fraction` (the warmup+cosine schedule) and the optimizer's initial `lr`, so the
schedule amplitude scales while its shape is unchanged. All else inherited from the EXP-012 baseline (k=4,
batch 128, WARMUP_FRAC 0.05, Nesterov, WD 1e-4, LS 0.1, Cutout(16), TrivialAugment, torch.compile, seed 42).
No deviations from plan-016. Ruff clean; `git diff` = the single PEAK_LR line.

## Execution
One run, no retries. Clean startup: `num_params 4,299,866` (UNCHANGED vs baseline — correct for a
hyperparameter-only change), clean compile, no traceback, no NaN. LR scaled correctly: warmup ramped
0.109→0.164 by 2.7% done and **peaked at exactly 0.3000** before the cosine descent. Higher-LR early noise
as predicted (ep 1 test_acc 39.80% vs EXP-015's 44.42% at the old peak) — recovered normally, no divergence.
Ran 84 epochs / 32,620 steps, dt ~8–13ms, peak VRAM 453.8 MB. Exited 0 in 400.8s < 600s budget.

## Results
- **Primary metric**: best_test_acc = **95.77%** (baseline: 96.22, delta: **−0.45pp**, −0.47%)
- **Observations**: final_test_loss 0.2018 (> EXP-012's 0.195 — loss ROSE). 95.77 is *below* the compiled-k4
  null band (~95.92, EXP-007/008/010/011), i.e. the higher peak actively *hurt*, not merely "no gain." The
  84-epoch count is within run-to-run throughput jitter (goal-learnings High Importance) and the change is
  compute-neutral, so the regression is attributable to the LR, not a throughput confound.
- **Analysis**: Hypothesis REFUTED in the tested direction. Raising the peak to 0.3 overshot — under the
  fixed ~84-epoch budget the more aggressive LR explores too much and the cosine anneal cannot recover a
  better minimum in time (loss↑ AND acc↓ = mild underfit/over-exploration). This is informative: peak 0.2 is
  at or above the optimum for this recipe+budget, NOT below it. The LR axis is therefore not flat — it has a
  *sign*, pointing toward a LOWER peak (0.1–0.15), which is the immediate next probe.
- **Key Learning**: Raising peak LR (0.2→0.3) hurts on the TA+Cutout k=4 recipe (95.77, −0.45pp, loss↑); peak
  0.2 is at/above optimal within the 300s budget — the LR axis points toward a LOWER peak, not higher.

## Verification
- **Conditions**: Cond 1 (clean completion < 600s, no traceback) PASS; **Cond 2 (best_test_acc ≥ 96.32) FAIL** (95.77); Cond 3 (scope) skipped — not reached (scope was clean for the record: train.py only, single PEAK_LR line, eval-count 84 == num_epochs, params unchanged, seed 42 intact).
- **Review Notes**: Results confirmed trustworthy — clean run, compute-neutral fair test (params/throughput unchanged), LR confirmed peaking at 0.3000, scope intact. No parsing anomalies.
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (primary metric did not clear the bar; −0.45pp below baseline).

## Unexplored Avenues
- **LOWER peak LR (0.1 or 0.15)**: the direct, evidence-backed follow-up — this experiment established the LR
  axis has a sign pointing *down* (0.3 overshot; textbook batch-128 WRN peak is 0.1). High value as the next probe.
- **Warmup-fraction or schedule-shape tuning** (e.g. longer warmup, linear/step decay instead of cosine):
  a different LR-schedule lever, but lower priority than getting the peak magnitude right first.
- **Cosine final-LR floor** (anneal to a small ε instead of 0): only relevant if combined with a schedule that
  benefits from a terminal-LR floor; EMA/SWA already nulled with cosine-to-0 (EXP-006).

## Next Steps
1. **Lower peak LR to 0.15 (or 0.1)** on the TA+Cutout recipe — the sign-corrected LR probe this experiment
   motivates; compute-free, fair test. Confidence: medium (the axis now has a direction).
2. **CutMix** (regional label-mixing aug, GPU-vectorized) — the higher-ceiling fallback from brainstorm-016
   if the LR axis proves shallow in both directions; risk is the Mixup-cousin null + epoch-budget. Confidence: low.
3. If both the lower-peak LR probe and CutMix null, the **96.0/96.22 regime is generalization-bound** at fixed
   k=4 capacity in 300s — ~10 axes exhausted; declaring the plateau becomes the honest call. Confidence: medium.

## Exit Action Results
- No exit actions defined for this goal — skipped.
