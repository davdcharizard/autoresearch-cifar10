# Report EXP-003: RandomErasing on the 4x-wide net
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-003.md
- **Plan**: plans/plan-003.md
- **Log**: logs/exp-log-003.md

## Goal

Maximize CIFAR-10 test accuracy (best_test_acc, %, higher is better) of train.py within the fixed 300s training budget (≤10 min total wall clock). Baseline at experiment start: **95.23%** @ bd0976e (EXP-001: 4x-wide ResNet-20 with time-keyed one-cycle recipe). Question tested: does occlusion-based augmentation convert the observed train/test gap into test accuracy at zero throughput cost?

## Idea & Hypothesis

Chosen over intermediate width 6x and torch.compile because it had the best evidence-to-risk ratio: Random Erasing (arXiv 1708.04896) reports +0.4–1.0pp on WRN-class CIFAR-10 models, EXP-001 showed an in-project generalization gap (final_test_loss 0.2447 with much lower train loss), the change is one line, and it is orthogonal to the capacity axis where EXP-002 just failed.

**Hypothesis**: Adding RandomErasing (p=0.5, scale (0.02, 0.4), random fill) to the train transform of the 4x-wide net raises best_test_acc from 95.23% to ≥95.5%, with throughput and epoch count essentially unchanged.

## Approach

Single transform appended to `train_tf` in train.py after Normalize (paper-standard placement, operates on tensors):

```python
transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value="random")
```

Architecture (WIDTH_MULT=4, 4.29M params) and the full EXP-001 recipe byte-identical otherwise. The eval transform lives in frozen prepare.py, so eval semantics are unchanged by construction. No deviations from plan; ruff clean; only train.py modified.

## Execution

Single run, no retries, no adjustments. Launched in background on GPU 0 (`CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`). Early signal matched prediction exactly: params 4,286,026 unchanged; epoch-1 test_acc 39.23% vs EXP-001's 39.20% (transform adds no early-training cost). Run completed cleanly in 399.5s total.

## Results

- **Primary metric**: best_test_acc = **96.06%** (baseline: 95.23, delta: **+0.83pp**, +0.87%)
- **Observations**:
  - num_epochs 114 and peak_vram_mb 1620.7 are *identical* to EXP-001 — a perfectly controlled comparison; the only delta is the transform.
  - final_test_loss fell 0.2447 → 0.2084 (−15%), directly confirming the gap-closing mechanism rather than a lucky best-epoch spike.
  - final = best (96.06 = 96.06): the one-cycle anneal still converges cleanly under the stronger augmentation; 114 epochs are sufficient — no undertraining signature.
  - The gain (+0.83pp) landed near the top of the published +0.4–1.0pp range despite only 114 epochs, likely because the 4x net was capacity-rich relative to augmented data complexity.
- **Analysis**: Hypothesis validated and exceeded (96.06 ≥ 95.5). The trajectory 91.97 → 93.16 (schedule) → 95.23 (capacity) → 96.06 (regularization) shows each loop attacked the then-binding constraint; regularization was the right axis after capacity, exactly as the EXP-002 failure suggested. Stronger augmentation may also shift the width-epoch optimum rightward: EXP-002's 8x failure was undertraining at 40 epochs, but with RandomErasing raising effective data complexity, wider nets have more signal per epoch to learn — 5–6x width on top of this recipe is now better-motivated than before.
- **Key Learning**: Once capacity is provisioned (4x width), occlusion regularization is nearly free accuracy — +0.83pp at zero throughput/VRAM cost — and it raises the ceiling for the next capacity step.

## Verification

- **Conditions**: all passed
  1. Clean completion within budget: total_seconds 399.5 ≤ 600 — PASS
  2. best_test_acc ≥ baseline + 0.1pp (≥ 95.33): 96.06 — PASS
  3. Eval at most once per epoch: 114 eval lines = 114 epochs — PASS
- **Review Notes**: results confirmed trustworthy — eval is the frozen prepare.py Eval class on the ground-truth test set; epoch count/VRAM identical to EXP-001 rules out throughput confounds; improvement came through the intended intervention class (training-input augmentation in train.py), not measurement gaming.
- **Verdict**: improvement
- **Verdict Basis**: all conditions passed + meaningful improvement (+0.83pp ≫ 0.1pp bar)

## Unexplored Avenues

- **Stronger/longer-schedule variants of erasing**: p=1.0, larger max area, or Cutout's fixed 16x16 patch — published results suggest p=0.5 random-area is near-optimal for CIFAR, but at higher width a stronger setting may pay.
- **Composing with width 5–6x**: RandomErasing raises effective data complexity, which strengthens the case for the next width step (the EXP-002 failure mode was epoch starvation *without* augmentation pressure; the optimum may have moved).
- **Other cheap augmentations**: TrivialAugmentWide / AutoAugment CIFAR policy (in torchvision, no new deps) — typically +0.5–1.0pp over flip+crop, partially overlapping with erasing's gain; risk is slower fitting at 114 epochs.
- **Mixup/CutMix**: stronger regularizers but interact with label smoothing and need loss changes; medium risk at this epoch count.

## Next Steps

1. **TrivialAugmentWide added to the current recipe** (high confidence of a clean signal either way; torchvision built-in, one line, zero throughput risk; literature gains are larger than erasing's and partially additive).
2. **Width 5–6x on top of the regularized recipe** (medium confidence: regularization shifts the width-epoch optimum rightward; 6x would get ~65–75 epochs, and final=best at 4x says epochs are not yet binding).
3. **torch.compile as an enabler for the width push** (low-medium confidence for accuracy directly; revisit if (2) shows undertraining).

## Exit Action Results
