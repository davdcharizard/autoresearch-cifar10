# Report EXP-017: LR-schedule micro-tuning — lower peak LR 0.2 → 0.15 (sign-corrected probe)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-017.md
- **Plan**: plans/plan-017.md
- **Log**: logs/exp-log-017.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%), higher-is-better, editing only `train.py` within the fixed 300s
training budget on a single H20. Baseline = **96.22%** (EXP-012, commit 6c417a4); success bar = **96.32%**
(+0.1pp). This experiment completes the LR-peak map by probing BELOW the baseline peak.

## Idea & Hypothesis
Chosen idea: lower `PEAK_LR` 0.2 → 0.15 (same 5% warmup → cosine-to-0). EXP-016 had established the LR optimum
is ≤ 0.2 (peak 0.3 regressed); 0.15 was the maximum-likelihood optimum location if it lay below 0.2 — a modest
step toward the textbook batch-128 WRN peak (0.1). Compute-free → a perfectly fair test. Hypothesis: a gentler
peak settles into a slightly better-generalizing minimum, lifting best_test_acc above 96.32; a null means peak
0.2 is already optimal and the LR-peak axis is settled.

## Approach
Single `train.py` edit (line 23): `PEAK_LR = 0.2` → `0.15`, inline comment noting the retune. `PEAK_LR` feeds
`lr_at_fraction` and the optimizer's initial `lr`; schedule shape unchanged, amplitude scales. All else inherited
from the EXP-012 baseline. No deviations from plan-017. Ruff clean; `git diff` = the single PEAK_LR line.

## Execution
One run, no retries. Clean startup: `num_params 4,299,866` (UNCHANGED — correct for a hyperparameter-only
change), clean compile, no traceback, no NaN. LR confirmed peaking at exactly 0.1500. Ran 77 epochs / 29,744
steps — the low end of the run-to-run throughput-jitter band (~65–77, goal-learnings High Importance). Exited 0
in 398.3s < 600s. peak VRAM 453.8 MB.

## Results
- **Primary metric**: best_test_acc = **95.58%** (baseline: 96.22, delta: **−0.64pp**, −0.67%)
- **Observations**: This is WORSE than both baseline (0.2 → 96.22) and EXP-016's higher peak (0.3 → 95.77).
  final_test_loss 0.2046 (> EXP-012's 0.195). The full LR-peak sweep: **0.15 → 95.58 | 0.2 → 96.22 | 0.3 → 95.77**
  — 0.2 is a clear interior optimum; both directions regress. (The 77-epoch count is on the low side this run; a
  lower LR + fewer epochs compounds mild under-progress, but the regression magnitude and the symmetric-falloff
  shape make peak-magnitude the dominant factor, not the epoch jitter.)
- **Analysis**: Hypothesis REFUTED. Combined with EXP-016, the LR-peak axis is now fully mapped and SETTLED: peak
  0.2 (the EXP-000 heuristic) is at/near the optimum for this recipe+budget; the curve falls off on both sides.
  The schedule was, in fact, already well-tuned despite never being explicitly swept on the current recipe. This
  closes the LR-peak axis and, with it, the last cheap "free" knob — the productive direction now is a genuinely
  new mechanism (CutMix) or accepting the plateau.
- **Key Learning**: Peak LR 0.2 is an interior optimum (0.15→95.58, 0.2→96.22, 0.3→95.77); the LR-peak axis is
  settled/well-tuned and closed — both directions regress.

## Verification
- **Conditions**: Cond 1 (clean completion < 600s, no traceback) PASS; **Cond 2 (best_test_acc ≥ 96.32) FAIL** (95.58); Cond 3 (scope) skipped — not reached (scope clean for the record: train.py only, single PEAK_LR line, eval-count 77 == num_epochs, params unchanged, seed 42 intact).
- **Review Notes**: Results confirmed trustworthy — clean run, compute-neutral fair test (params/throughput unchanged), LR confirmed peaking at 0.1500, scope intact. The symmetric falloff around 0.2 across three runs is internally consistent.
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (primary metric did not clear the bar; −0.64pp below baseline).

## Unexplored Avenues
- **LR-schedule SHAPE** (warmup fraction, decay form, cosine-with-restarts): a second-order LR lever still
  untouched — but with the peak magnitude now pinned at a clean optimum and the axis read as well-tuned, expected
  value is low. Likely not worth a loop.
- **CutMix** (regional label-mixing aug, GPU-vectorized): the standing higher-ceiling fallback — a genuinely new
  *mechanism*, not another knob on a settled axis. Risk is the Mixup-cousin null (EXP-011) + epoch-budget underfit.
- **Combined Mixup+CutMix with switching** (timm-style): stronger but more complex/risky; only if single CutMix shows life.

## Next Steps
1. **CutMix** (GPU-vectorized, per-batch, alongside Cutout) — pivot OFF the now-settled LR axis to the last
   well-evidenced untried mechanism. Confidence: low-medium (higher ceiling, but Mixup-cousin + epoch-budget risk).
2. If CutMix nulls, the **96.0/96.22 regime is generalization-bound** at fixed k=4 capacity in 300s — ~10 axes now
   exhausted (capacity, block-ordering, activation, attention, EMA/SWA, WD, more-epochs, aug-policy, aug-strength
   variants, LR-peak). Declaring the plateau becomes the honest scientific call. Confidence: medium-high.
3. (Low priority) LR-schedule shape (longer warmup / SGDR restarts) only if a reason emerges to revisit the LR axis.

## Exit Action Results
- No exit actions defined for this goal — skipped.
