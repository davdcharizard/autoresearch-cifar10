**Prioritized Feedback**

1. **EXP-014 is the dominant context: capacity/epochs are no longer the limiter.** `project-insights.md` and `experiments/014/04-analysis.md` show compile bought +19 epochs and layer2-320 annealed at 143 epochs, yet accuracy tied/lost. This directly weakens Idea 2’s “more/better tail” premise and Idea 3’s “compile-funded capacity” premise. Improve by treating both as low-EV diagnostics, not headline win attempts.

2. **Idea 1 is the only candidate attacking a not-directly-falsified mechanism, but it overstates the augmentation lane.** EXP-008’s Cutout12+RandomErasing won +0.38, but EXP-011 showed a second strong input aug, CutMix, only tied, and EXP-009/008 notes warned RandAugment/AutoAugment/TrivialAugment can underfit at ~150 epochs. Improve by making the primary cells mild and comparative: baseline, policy replacing RandomErasing/Cutout, and only one conservative stack variant. Do not just add strong policy aug on top.

3. **Idea 1’s curriculum tail-off has a concrete implementation trap.** `train.py` uses `persistent_workers=True` in the DataLoader, so mutating `train_set.transform` from the main process late in training likely will not reach worker copies. Fix with a worker-visible/shared schedule, rebuild the DataLoader deliberately, or skip curriculum and use fixed low probability/strength.

4. **Idea 1’s “throughput-free” claim needs wall-time proof.** Policy aug runs before `ToTensor` on PIL/uint8 and can be CPU-heavy; DataLoader wait is excluded from `training_seconds`, so it can inflate `total_seconds` toward the 600s wall cap without reducing `num_epochs`. Pre-smoke worker throughput and make `total_seconds` a verdict metric.

5. **Idea 2 is clean but probably below the measurable effect size.** The current LR is already time-based and reaches zero (`train.py:282-290`), while EXP-014 showed extra low-LR epochs are worth only +0.03pp. If tested, pre-register it as a same-session micro-sweep and require a clear >0.1pp delta; otherwise it is likely noise-chasing.

6. **Idea 3 is not an unchanged failed retry, but it is very close to the exhausted within-architecture capacity axis.** First layer2 ReZero helped at a lower baseline, but later width and compile-funded capacity did not move the 96.3-96.5 ceiling. If run anyway, use it only with EXP-014’s validated compile recipe, record `num_epochs`, and smoke-test both `alpha.grad` and branch-conv gradients after one update.

7. **The cited EXP-015 proposal files are missing.** `experiments/015/proposals/idea-01.md` through `idea-03.md` are not present, so several evidence claims cannot be audited beyond the brainstorm text. Not fatal, but it lowers confidence.

**Scored Verdict**

- **Idea 1: Policy augmentation**
  - Evidence/reasoning: **6/10**. Supported by EXP-008’s augmentation win and a genuinely different transform class, but weakened by EXP-011 input-aug saturation and prior long-schedule underfit warnings.
  - Potential impact: **7/10**. Best plausible upside among the three if mild policy aug adds geometric/photometric diversity without over-regularizing.

- **Idea 2: Schedule reshaping**
  - Evidence/reasoning: **5/10**. Easy and untried, but its claimed mechanism overlaps with the anneal/extra-epoch axis EXP-014 largely falsified.
  - Potential impact: **3/10**. Likely sub-noise on an already time-annealed recipe.

- **Idea 3: Second layer2 ReZero block, compile-funded**
  - Evidence/reasoning: **4/10**. Exact depth-at-layer2 is untested, but the current diagnosis says same-architecture capacity is saturated.
  - Potential impact: **4/10**. Could clear +0.1pp only if depth behaves differently from width, but the post-EXP-014 prior is poor.

**Pick: Idea 1, policy-based augmentation, with strict mild/replacement design.** It wins because it is the only candidate that plausibly attacks the generalization ceiling through a mechanism not already directly falsified. Idea 2 is too small; Idea 3 is too aligned with the capacity path EXP-014 just closed.
