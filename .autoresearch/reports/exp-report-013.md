# Report EXP-013: Reduce Cutout hole size 16→8px under the TA+compile recipe
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-013.md
- **Plan**: plans/plan-013.md
- **Log**: logs/exp-log-013.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%) under a fixed 300s budget on one H20, editing only `train.py`. Higher is
better. Baseline **96.22%** (EXP-012); success bar ≥ **96.32%**.

## Idea & Hypothesis
Chosen idea: on the EXP-012 baseline (TA + Cutout(16) + compile), reduce `CUTOUT_SIZE` 16→8. Rationale: the 16px hole
was tuned *before* TrivialAugment existed (EXP-002/003); adding TA raised total augmentation strength, so the optimal
occlusion size likely shifted down — a 16px hole stacked on TA might over-regularize and slow convergence at the 300s
budget. Hypothesis: a smaller 8px hole reduces over-regularization, improves the fit, and lifts `best_test_acc` above
96.32 with a corroborating drop in final_test_loss below 0.195.

## Approach
Single-constant `train.py` edit: `CUTOUT_SIZE = 16` → `8`, everything else identical to the EXP-012 baseline (k=4
WRN, TrivialAugmentWide, compile reduce-overhead, batch 128, peak LR 0.2 cosine, Nesterov, WD 1e-4, LS 0.1, seed 42).
Ruff clean; diff = the one CUTOUT_SIZE line. Parameter-free change → num_params unchanged (sanity check).

## Execution
One run, no retries/errors, clean compile, exit 0. Steady-state ~9ms/step (~15,000 img/s ≈ EXP-012; Cutout size has
no throughput effect) after epoch-1 warmup jitter. Fit **92 epochs** — a fair, fully-converged test (eval count 92 ==
num_epochs). Completed 410.3s, peak VRAM 453.8 MB, params 4,299,866 (unchanged).

## Results
- **Primary metric**: **95.92%** (baseline 96.22, delta **−0.30 pp**, −0.31%) — below the bar AND the baseline.
- **Observations**: **final_test_loss 0.2023 ROSE** vs EXP-012's 0.195 (and is even slightly above the original
  pre-TA 96.0 recipe's 0.204). Loss↑ AND acc↓ together — the clean signature of *under*-regularization. Late evals
  cluster 95.82–95.92 (ep 88–92), distinctly below EXP-012's 96.12–96.22 cluster — a real shift, not noise.
- **Analysis**: The hypothesis was **refuted in an informative direction**. Reducing the Cutout hole did NOT help —
  it *hurt* both fit quality (loss↑) and accuracy (−0.30pp). This means the 16px occlusion was NOT over-regularizing
  under TA; the EXP-012 TA+Cutout(16) recipe is already well-tuned, and Cutout's occlusion benefit is **orthogonal**
  to TA's photometric/geometric diversity (not redundant — TA does not subsume it). The model under the winning recipe
  is at/near the right regularization level, NOT over-regularized/underfit; lowering augmentation strength pushed it
  the wrong way. Consistent with the project pattern that this k=4 net at 300s is generalization-bound and sensitive
  to the regularization sweet spot.
- **Key Learning**: Under TrivialAugment, Cutout(16) is already near its sweet spot — shrinking it to 8px
  under-regularizes (95.92, loss 0.202>0.195); TA's diversity is orthogonal to occlusion, not a replacement for it.

## Verification
- **Conditions**: Cond 1 (clean completion in budget) PASS; Cond 2 (≥96.32) **FAIL** (95.92); Cond 3 skipped.
- **Review Notes**: Trustworthy — clean single run, frozen eval, seed 42, eval once/epoch (92==92), num_params
  UNCHANGED (4,299,866), diff = train.py one line. The negative is corroborated by the independent loss metric
  (test loss ↑) and a stable late-eval cluster, so it is a genuine regression, not a noise dip. No reward-hacking
  surface. −0.30pp is beyond the ~0.2pp noise band.
- **Verdict**: no-improvement
- **Verdict Basis**: valid, trustworthy, fair converged run; primary metric below the bar (cond 2 failed). Baseline
  holds at 96.22 (EXP-012).

## Unexplored Avenues
- **Larger Cutout under TA (e.g. 20px or two 16px holes)** — since *less* occlusion hurt, the sweet spot is ≥16px;
  *more* occlusion is the untested direction. But TA already adds strength, so >16px risks over-regularizing/underfit
  at the budget — low-medium, two-sided. A single 20px probe would bracket the sweet spot.
- **RandAugment(N=2) vs TA** — a stronger (two-op) auto-aug policy; the augmentation-strength axis is clearly live
  (TA gained, less-aug lost), so a stronger policy is the natural direction to probe MORE augmentation. Medium.
- **Complementary regularizer ON TOP of TA+Cutout(16)** — mild Mixup / stronger label smoothing now that the recipe
  is more strongly augmented; the EXP-011 Mixup null was on the *old* recipe. Low-medium.

## Next Steps
1. **RandAugment(num_ops=2, magnitude≈9) replacing TrivialAugment** — *medium confidence*; EXP-012 (TA gained) and
   EXP-013 (less-aug lost) together show the augmentation-STRENGTH axis is live and points toward MORE, not less,
   augmentation — a two-op policy is the cleanest next probe of that direction. *Best next experiment.*
2. **Larger Cutout (20px or 2×16 holes) under TA** — *low-medium*; brackets the occlusion sweet spot from above now
   that 8px (below) is known worse.
3. **Mild Mixup / stronger label smoothing stacked on TA+Cutout(16)** — *low-medium*; re-test a complementary
   regularizer on the stronger-aug recipe where the overfit/underfit balance differs from EXP-011's old recipe.

## Exit Action Results
- None defined for this goal — skipped.
