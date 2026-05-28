# Report EXP-022: Reflect Padding + Cutout Replacing RandomErasing
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-022.md
- **Plan**: plans/plan-022.md
- **Log**: logs/exp-log-022.md

## Goal

Maximize CIFAR-10 test accuracy (best_test_acc, %, higher is better). Baseline: 96.46% (EXP-020). Threshold: >96.56% (baseline + 0.1pp). This experiment tested whether swapping augmentation components (reflect padding + Cutout) would improve data augmentation quality without changing regularization pressure.

## Idea & Hypothesis

Selected "Reflect Padding + Cutout Combined" over Cutout-alone and deeper architecture (NUM_BLOCKS=4). Both components are validated by airbench96's 96.05% recipe. Hypothesis: replacing zero-padding with reflect-padding in RandomCrop and replacing RandomErasing with Cutout(12px) would improve best_test_acc by +0.1-0.3pp (targeting 96.56-96.76%) by providing higher-quality augmented samples — reflect padding eliminates artificial zero borders, Cutout provides consistent fixed-size occlusion complementing TrivialAugmentWide better than RandomErasing's variable-size random-fill approach. Zero throughput cost expected.

## Approach

Three changes to `train.py`'s augmentation pipeline:
1. Added custom `Cutout` class (12×12 fixed zero-fill square, p=0.5) operating on normalized tensors, placed after `transforms.Normalize`
2. Changed `RandomCrop(32, padding=4)` to use `padding_mode='reflect'`
3. Replaced `RandomErasing(p=0.25, scale=(0.02, 0.2))` with `Cutout(size=12, p=0.5)`

No deviations from plan. Implementation was straightforward — augmentation pipeline swaps with no unexpected code interactions.

## Execution

Single run, no retries. Training completed normally: 99 epochs in 300.0s budget (~15-16ms/step), confirming zero throughput cost from the augmentation changes. Best accuracy reached at epoch 95 (96.53%), with slight decline through epoch 99 (96.33%). Peak VRAM 864.6 MB (unchanged from baseline).

## Results

- **Primary metric**: 96.53% (baseline: 96.46%, delta: +0.07pp, +0.07%)
- **Observations**: Zero throughput cost confirmed (99 epochs, same as baseline). Best accuracy at epoch 95, then declining — training may be slightly over-regularized in final epochs. Epoch progression: ep93 96.51%, ep94 96.44%, ep95 96.53%, ep96 96.42%, ep97-99 declining to 96.33%.
- **Analysis**: The +0.07pp gain is within noise margin and below the 0.1pp threshold. The hypothesis that reflect padding + Cutout would compound to a meaningful improvement was not confirmed. The augmentation-quality swap produced a marginal effect, suggesting that at 96.46% baseline, the augmentation pipeline (TrivialAugmentWide + occlusion) is already near-optimal — swapping between equivalent-strength occlusion methods (RandomErasing vs Cutout) yields negligible returns. The late-epoch decline (96.53% at ep95 → 96.33% at ep99) hints Cutout p=0.5 may be slightly stronger regularization than RandomErasing p=0.25, causing mild overshoot past the accuracy peak.
- **Key Learning**: At 96.46% accuracy, augmentation-quality swaps between comparable occlusion methods are in the noise floor; gains must come from capacity, optimization dynamics, or fundamentally different training signals.

## Verification

- **Conditions**: Condition 1 FAILED (96.53 ≤ 96.56); Conditions 2-3 PASSED
- **Review Notes**: Results confirmed trustworthy. All 10 summary fields present, 99 eval runs matching 99 epochs, metrics internally consistent. No parsing errors or stale output concerns.
- **Verdict**: no-improvement
- **Verdict Basis**: Primary metric (96.53%) did not exceed threshold (96.56% = baseline + 0.1pp). Condition 1 failure.

## Unexplored Avenues

- **Cutout with different size/probability**: 12px at p=0.5 was chosen from airbench96, but our model and training recipe differ substantially. Larger Cutout (16px, per original paper for WRN-28-10) or lower probability (p=0.25, matching prior RandomErasing rate) could shift the regularization balance. However, given the +0.07pp result, the upside ceiling for Cutout tuning is likely small.
- **Reflect padding alone (without Cutout swap)**: Isolating reflect padding's contribution would clarify whether the +0.07pp came from reflect padding, Cutout, or their interaction. But the expected effect of reflect padding alone on 4px padding of 32×32 images is minimal.
- **Cutout on GPU tensors via batch-level masking**: Current Cutout runs per-sample on CPU in the data pipeline. A batch-level GPU implementation could be more efficient and enable larger/more aggressive masking without throughput cost.

## Next Steps

1. **Deeper architecture (NUM_BLOCKS=4, ResNet-26)**: Adds ~33% more conv layers for increased capacity. Risk is ~25-30% throughput loss, but the model may be capacity-limited at 96.46%. (medium confidence)
2. **Scheduled EMA with lower β or cubic schedule**: EXP-014 showed β=0.999 too conservative for ~92 epochs. Lower β (0.995-0.998) or airbench96-style cubic scheduling could help. Need efficient implementation to minimize throughput cost. (medium confidence)
3. **Higher BN bias learning rate (airbench96 pattern)**: Separate BN bias param group with significantly higher LR. Novel technique from airbench96, untried in this project. (low confidence — airbench96's architecture differs substantially)
