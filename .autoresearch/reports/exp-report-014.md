# Report EXP-014: RandAugment(2,9) replacing TrivialAugmentWide (keep Cutout(16) + compile)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-014.md
- **Plan**: plans/plan-014.md
- **Log**: logs/exp-log-014.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%) under a fixed 300s budget on one H20, editing only `train.py`. Higher is
better. Baseline **96.22%** (EXP-012); success bar ≥ **96.32%**.

## Idea & Hypothesis
Chosen idea: replace `TrivialAugmentWide()` with `RandAugment()` (torchvision defaults num_ops=2, magnitude=9) on the
EXP-012 recipe, keeping Cutout(16) + compile. Rationale: EXP-012 (adding TA gained +0.22) and EXP-013 (reducing aug
lost −0.30) established that the augmentation-strength axis is live and points toward MORE augmentation; RA(2,9)
applies two ops/image (vs TA's single op) and is the canonical strong CIFAR-WRN auto-aug. Hypothesis: more
augmentation per image lifts `best_test_acc` above 96.32 (~96.3–96.6%) with final_test_loss ≤ 0.195.

## Approach
Single-line `train.py` swap: `transforms.TrivialAugmentWide()` → `transforms.RandAugment()` (defaults: 2 ops at
magnitude 9). Everything else identical to the EXP-012 baseline (k=4 WRN, Cutout(16), compile reduce-overhead, batch
128, peak LR 0.2 cosine, Nesterov, WD 1e-4, LS 0.1, seed 42). Ruff clean; diff = the augmentation line + comment.
Parameter-free → num_params unchanged.

## Execution
One run, no retries/errors, clean compile, exit 0. Steady **8ms/step (~15,300 img/s) — identical to EXP-012**:
RandAugment (CPU PIL ops, no GPU sync) added no throughput cost. Fit **91 epochs** — a fair, fully-converged test
(eval count 91 == num_epochs). Completed 411.4s, peak VRAM 453.8 MB, params 4,299,866 (unchanged).

## Results
- **Primary metric**: **96.19%** (baseline 96.22, delta **−0.03 pp**, −0.03%) — below the bar; ≈ baseline (within noise).
- **Observations**: **final_test_loss 0.1972 ≈ EXP-012's 0.195** (within noise). RA(2,9) and TA produce essentially
  identical fit-quality AND accuracy. Late evals cluster 96.11–96.19 (ep 87–91), overlapping TA's 96.12–96.22 — a
  statistical tie, not a regression.
- **Analysis**: The hypothesis is **not supported, in an informative way**. RandAugment(2,9) ≈ TrivialAugment here:
  swapping the auto-augmentation *policy* (single-op uniform-strength TA vs two-op fixed-magnitude RA) does NOT move
  the metric. This matches the literature (TA ≈ tuned RA on CIFAR-WRN) and tells us the **auto-aug-policy axis is
  saturated** — both strong policies land at the same ~96.2 plateau with loss ~0.195. Combined with EXP-012 (adding
  *a* strong policy gained) and EXP-013 (reducing aug lost), the picture sharpens: *having* a strong diverse
  auto-aug matters a lot; *which* strong policy (or whether 1 vs 2 ops) does not. The two-op increment did not
  over-augment (loss flat, not raised) nor help — it's simply redundant with TA's single-op-but-uniform-strength
  diversity at this operating point.
- **Key Learning**: RandAugment(2,9) ≈ TrivialAugment (96.19 ≈ 96.22, loss 0.197 ≈ 0.195) — the auto-aug *policy*
  choice is saturated; once a strong diverse policy is present, swapping or adding ops doesn't move the metric.

## Verification
- **Conditions**: Cond 1 (clean completion in budget) PASS; Cond 2 (≥96.32) **FAIL** (96.19); Cond 3 skipped.
- **Review Notes**: Trustworthy — clean single run, frozen eval, seed 42, eval once/epoch (91==91), num_params
  UNCHANGED, diff = train.py one line, dt 8ms (no throughput confound). Result is a clean statistical tie with the
  TA baseline (loss + late-eval cluster both ≈ EXP-012), not a regression. No reward-hacking surface.
- **Verdict**: no-improvement
- **Verdict Basis**: valid, trustworthy, fair converged run; primary metric below the bar (cond 2 failed), ≈ baseline
  within noise. Baseline holds at 96.22 (EXP-012, TA).

## Unexplored Avenues
- **Larger Cutout (20px) under TA** — the occlusion-size sweet spot was bounded below by EXP-013 (8<16); >16 is still
  untested. A different mechanism (occlusion) than the now-saturated auto-aug *policy*. Low-medium (16 is textbook
  optimum; may over-regularize).
- **Mixup/CutMix stacked on TA+Cutout** — a third orthogonal augmentation mechanism (interpolation); EXP-011 Mixup
  null was on the old Cutout-only recipe. Low-medium (underfit risk from a 3rd aug at 300s).
- **Non-augmentation levers** — the augmentation axis (policy + Cutout-down) is largely mapped now; LR-schedule
  micro-tuning (peak LR, warmup) on the TA recipe is genuinely untried (only WD was swept, EXP-005, on the old recipe).

## Next Steps
1. **Pivot OFF auto-aug-policy tuning** — *strategic, high confidence*: EXP-014 shows policy choice is saturated;
   stop swapping/tuning auto-aug policies. The remaining augmentation move is a *different mechanism* (larger Cutout
   or Mixup/CutMix on top), not another policy.
2. **LR-schedule micro-tuning on the TA recipe (peak LR 0.2→0.15 or 0.25, or warmup fraction)** — *low-medium*; a
   genuinely untried optimization knob (only WD swept, on the old recipe), orthogonal to augmentation. *Best next
   experiment* for opening a new axis.
3. **Larger Cutout (20px) under TA** — *low-medium*; brackets the occlusion sweet spot above 16 (different mechanism
   than the saturated policy axis).

## Exit Action Results
- None defined for this goal — skipped.
