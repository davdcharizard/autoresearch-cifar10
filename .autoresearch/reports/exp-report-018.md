# Report EXP-018: CutMix (regional label-mixing aug), GPU-vectorized per batch, on the TA+Cutout recipe
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-018.md
- **Plan**: plans/plan-018.md
- **Log**: logs/exp-log-018.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%), higher-is-better, editing only `train.py` within the fixed 300s
training budget on a single H20. Baseline = **96.22%** (EXP-012, commit 6c417a4); success bar = **96.32%**
(+0.1pp). This experiment tested CutMix — the last well-evidenced untried augmentation mechanism.

## Idea & Hypothesis
Chosen idea: add CutMix (Yun et al. 2019) as a per-batch GPU op (p=0.5, α=1.0) on top of the TA+Cutout recipe —
paste a random area-(1−λ) box from a shuffled batch copy, mix labels by kept area, loss = λ·CE(y) + (1−λ)·CE(y_perm).
Selected because the augmentation-strength axis broke the plateau once (TA, EXP-012) and CutMix is the strongest
evidenced aug mechanism not yet tried (regional, label-mixing, distinct from photometric TA and occlusion Cutout).
Hypothesis: reduces the residual generalization gap and lifts best_test_acc above 96.32; main downside risk is
underfit within the short budget (CutMix wants 200–300 epochs).

## Approach
Three train.py edits (no new deps; α=1.0 ⇒ λ~Uniform, so `torch.rand` suffices): added `CUTMIX_PROB=0.5`/
`CUTMIX_ALPHA=1.0`; added a `cutmix_batch(x)` GPU helper (randperm + slice-paste, returns area-corrected λ); and a
per-batch coin-flip in the loop applying CutMix + the two-term soft-target loss (plain CE otherwise). Cutout and TA
kept. Model input shape unchanged → torch.compile CUDA graphs unaffected. No deviations from plan-018. Ruff clean.

## Execution
One run, no retries. Clean startup: `num_params 4,299,866` (UNCHANGED — aug-only), clean compile, no traceback, no
NaN. CutMix path ran correctly (ep 1 acc 33.64%, low as expected with soft targets; dt ~8–14ms). Ran only 71 epochs
/ 27,535 steps (low end of the throughput-jitter band). Test loss elevated throughout (0.25–0.30, soft-target
artifact). Exited 0 in 394.9s < 600s. peak VRAM 453.8 MB (unchanged).

## Results
- **Primary metric**: best_test_acc = **95.14%** (baseline: 96.22, delta: **−1.08pp**, −1.12%) — the largest
  augmentation regression in the project.
- **Observations**: final_test_loss 0.2774 (soft-target artifact — high by design, not a quality signal). Only 71
  epochs fit (CutMix's extra ops + run jitter). The accuracy fell well below even the compiled-k4 null band (~95.92).
- **Analysis**: Hypothesis REFUTED — and instructively so. CutMix is a strong label-mixing regularizer that warms
  up slowly (papers train 200–300 epochs); at our ~71-epoch budget it UNDERFITS, and stacked on an already
  regularization-saturated recipe (TA + Cutout + LS + WD) it actively hurt by −1.08pp. This is the SAME failure
  family as the weak-Mixup null (EXP-011), now confirmed with the strongest variant: label-mixing augmentation does
  not help this recipe/budget. Combined with EXP-012/013/014 (TA gains, policy saturated, Cutout co-tune saturated),
  the **augmentation axis is now settled** — TrivialAugment is the ceiling. The result strongly corroborates that
  the 96.22 plateau is generalization-bound at fixed k=4 capacity within 300s.
- **Key Learning**: CutMix (and label-mixing aug generally) UNDERFITS and regresses (−1.08pp) at the 300s/~71-epoch
  budget on the saturated TA+Cutout recipe — the augmentation axis is settled, TrivialAugment is the ceiling.

## Verification
- **Conditions**: Cond 1 (clean completion < 600s, no traceback) PASS; **Cond 2 (best_test_acc ≥ 96.32) FAIL** (95.14); Cond 3 (scope) skipped — not reached (scope clean for the record: train.py only, eval-count 71 == num_epochs, params unchanged, seed 42 intact, no new deps).
- **Review Notes**: Results trustworthy — clean run, throughput-neutral fair test (params unchanged), CutMix path verified executing, scope intact. The elevated test loss is the expected soft-target artifact (per the plan and the CutMix knowledge note), NOT a failure signal; the verdict rests on accuracy.
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (primary metric did not clear the bar; −1.08pp below baseline).

## Unexplored Avenues
- **CutMix at lower probability (p=0.25) or with a warmup-then-enable schedule**: could reduce the underfit, but the
  −1.08pp gap is large and the budget can't grow — low expected value.
- **CutMix WITHOUT Cutout (swap, not stack)**: would isolate CutMix from the stacking confound, but EXP-013 showed
  removing/reducing Cutout hurts, so this likely trades one regularizer for a worse-fitting one — low value.
- These are refinements of a mechanism that regressed substantially; none are promising within the 300s budget.

## Next Steps
1. **Declare the 96.22 plateau**: this was the last well-evidenced untried mechanism. ~11 axes are now exhausted
   (capacity, block-ordering, activation, attention, EMA/SWA, weight-decay, more-epochs, aug-policy, aug-strength
   variants, LR-peak, label-mixing aug). The model is generalization-bound at fixed k=4 capacity in 300s. Confidence: high.
2. **If continuing**: only genuinely-novel, low-confidence probes remain — e.g. test-time-augmentation is blocked
   (eval frozen); a different optimizer (AdamW) is low-ceiling on CIFAR convnets; batch-size scaling is blind and
   fights saturated epochs. Confidence: low that any clears +0.1pp.
3. Per-channel input std-normalization (code uses std=(1,1,1)) remains the one cheap untried probe, though BN almost
   certainly absorbs it (expected null). Confidence: low.

## Exit Action Results
- No exit actions defined for this goal — skipped.
