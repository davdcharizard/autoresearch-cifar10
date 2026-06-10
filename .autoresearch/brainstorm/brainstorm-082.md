# Brainstorm EXP-082
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CutMix regularization** (`knowledge/papers/cutmix-regularization.md`)
  The current best anchor depends on regional image/label mixing, so the next experiment should preserve CutMix rather than retuning its already bracketed strength.
- **Torchvision RandomCrop padding modes** (`knowledge/references/torchvision-randomcrop-padding.md`)
  Reflection crop padding remains the validated spatial crop boundary behavior, and EXP-081 showed reducing its magnitude to 3 gives only a sub-threshold gain.
- **Wide residual networks** (`knowledge/papers/wide-residual-networks.md`)
  Capacity remains a plausible broad lever, but local fixed-budget evidence is more important than generic CIFAR guidance because architecture changes often trade accuracy for step budget.

No new external search was needed. The local 82-experiment trajectory gives more specific guidance than additional generic CIFAR augmentation papers for this next narrow probe.

## Experimental History Review

- Current best remains EXP-064 at `best_test_acc=94.11%` from commit `1119ff8`; the +0.10pp noise guard requires `best_test_acc >= 94.21%`.
- EXP-081 reached 94.18% with `RandomCrop` reflection padding 3, a valid near miss but below threshold. This supports the possibility of mild over-regularization but does not justify changing the crop anchor by itself.
- CutMix strength/timing is locally bracketed: probability 0.25/0.75, alpha 0.5/2.0, post-drop taper, clean warmup, and short probability ramp all missed threshold. Keep `CUTMIX_ALPHA=1.0` and `CUTMIX_PROB=0.5`.
- Label smoothing, startup LR, batch size, weight averaging, and scheduler-only changes are recurring failed families. The current anchor should preserve `label_smoothing=0.05`, `LR=0.1`, batch size 128, and the 21k first LR drop.
- Added augmentation is usually too strong: Cutout, ColorJitter, RandAugment, and AutoAugment underperformed. The remaining plausible spatial-regularization lane is not adding augmentation, but slightly weakening one existing no-overhead transform.
- Architecture/topology experiments remain weaker than the augmentation micro-tune lane, but a narrow final-stage width reduction is still a plausible fallback if spatial probes are exhausted.

## Candidate Ideas

### 1. Lower Horizontal Flip Probability
**Summary**: Replace the default `transforms.RandomHorizontalFlip()` with `transforms.RandomHorizontalFlip(p=0.4)`. Preserve reflection crop padding 4, CutMix alpha/probability/label smoothing, clean label smoothing, architecture, optimizer, LR schedule, batch size, seed, compile/channels-last, and validation cadence.

**Reasoning**: EXP-081 suggests that slightly weakening spatial augmentation can produce a near-threshold result, but crop padding 3 did not clear the noise guard. Horizontal flip probability is another no-overhead spatial augmentation strength knob that targets a different invariance from translation jitter. Reducing flip frequency modestly may reduce over-regularization while keeping the validated reflection crop and CutMix anchor intact.

**Sources**: `train.py` transform pipeline; `reports/exp-report-081.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `knowledge/papers/cutmix-regularization.md`

**Estimated Effort**: low

**Risk Assessment**: CIFAR horizontal flips are a strong default, so lowering the probability may weaken useful invariance and regress. The failure mode should be a clean no-improvement with little infrastructure risk.

### 2. Final Stage Width 104
**Summary**: Keep ResNet-20 depth and the first two stage widths unchanged, but reduce final stage width from 112 to 104: `STAGE_WIDTHS=(28, 56, 104)`. Preserve current CutMix, transforms, optimizer, LR schedule, batch size, seed, compile/channels-last, and validation cadence.

**Reasoning**: The 28/56/112 anchor is strong but may be close to the fixed-budget capacity/optimization boundary. A small layer3 reduction could increase useful step coverage and post-drop refinement while staying closer to the anchor than prior shallow-wide or topology rewrites.

**Sources**: `knowledge/papers/wide-residual-networks.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; EXP-062, EXP-077, EXP-078; `train.py` `STAGE_WIDTHS`

**Estimated Effort**: low

**Risk Assessment**: Local architecture changes usually underperform, and reducing final-stage width may lower the accuracy ceiling more than it improves optimization coverage. It is a reasonable fallback but weaker than the next spatial micro-tune.

### 3. Slightly Higher Horizontal Flip Probability
**Summary**: Replace `transforms.RandomHorizontalFlip()` with `transforms.RandomHorizontalFlip(p=0.6)`, preserving all other anchor settings.

**Reasoning**: If EXP-081's near miss was not over-regularization but simply random variation, slightly stronger flip invariance might improve generalization without adding new transforms or runtime overhead. This brackets the flip knob symmetrically around the default.

**Sources**: `train.py` transform pipeline; `reports/exp-report-081.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`

**Estimated Effort**: low

**Risk Assessment**: Because added regularization repeatedly underperforms on top of CutMix, increasing flip probability is less aligned with recent evidence than decreasing it. It should wait until the lower-probability side is tested.

## Idea Evaluation

Lower horizontal flip probability is the best next experiment. It follows the same narrow, no-overhead spatial-regularization logic that made EXP-081 informative, but it probes a different mechanism while preserving the validated reflection crop padding 4. It also avoids closed families: it does not retune CutMix strength or timing, alter label smoothing, change LR startup, add policy augmentation, or rewrite architecture.

Final-stage width 104 is feasible and could improve step coverage, but architecture changes are a locally weak family and the current goal needs only a small gain; lowering capacity risks giving away the accuracy ceiling. Slightly higher flip probability is easy, but recent evidence favors reducing regularization pressure rather than adding more.

## Chosen Idea
**Selected**: Lower Horizontal Flip Probability

**Why this idea**:
It is the cleanest remaining no-overhead spatial augmentation strength probe after the padding-3 near miss. The change is one line, leaves all validated anchor settings intact, and tests whether the CutMix recipe is marginally over-regularized by full default flip frequency.

**Hypothesis**:
If the current CutMix anchor is slightly over-regularized by always using the default 0.5 horizontal flip probability, then reducing flip probability to 0.4 will improve late clean accuracy enough to raise `best_test_acc` from 94.11% to at least 94.21%.
