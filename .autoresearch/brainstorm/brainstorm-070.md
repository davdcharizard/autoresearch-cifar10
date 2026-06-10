# Brainstorm EXP-070
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Project knowledge base** (`knowledge/README.md`)
  Prior saved knowledge covers CutMix, mixup, Cutout, RandAugment, stochastic depth, SE, downsampling tweaks, cosine schedules, and PyTorch throughput tools. No new external source was needed for EXP-070 because the next highest-signal gap is already visible in the local recipe: the current transform subtracts CIFAR channel means but uses unit standard deviations.

## Experimental History Review

- Current best remains EXP-064 at `best_test_acc=94.11%`, commit `1119ff8`, from `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, endpoint label smoothing 0.05, `WEIGHT_DECAY=2e-4`, reflection crop padding, and the step-21000 first LR drop.
- The active goal requires `best_test_acc >= 94.21%` to count as an improvement; ties and smaller gains are `no-improvement`.
- CutMix scalar and temporal strength are now locally bracketed: EXP-065/066 tested probability 0.25/0.75, EXP-067/068 tested alpha 0.5/2.0, and EXP-069 tested a post-drop probability taper to 0.25. All missed threshold.
- High-importance failures caution against isolated second-drop schedules, EMA/SWA, batch-size deviations, and label-smoothing deviations. Medium-importance failures caution against direct mixup, cosine LR variants, SE, residual BN down-scaling, LR scalar changes, BN/bias decay exceptions, and Cutout.
- Prior reports already identified standard CIFAR channel-std normalization as a distinct input-conditioning lever, but it has not been tried on the current CutMix anchor. The current code uses `std=(1,1,1)`, preserving channel centering while leaving per-channel scale unstandardized.

## Candidate Ideas

### 1. Standard CIFAR Channel-Std Normalization
**Summary**: Change the training normalization tuple from `std=(1,1,1)` to the standard CIFAR-10 per-channel standard deviations, while preserving the current means, CutMix anchor, architecture, optimizer, LR schedule, reflection padding, label smoothing, compile/channels-last path, and validation cadence.

**Reasoning**: The current transform subtracts per-channel CIFAR means but does not scale by channel standard deviation. Standard channel-std normalization is a distinct input-conditioning lever: it can change first-layer activation/gradient scaling without adding augmentation overhead or touching the evaluator. Unlike schedule, smoothing, batch-size, and CutMix-strength retunes, this gap has not been tested on the successful EXP-064 anchor. EXP-026 explicitly called it a medium-confidence next direction after momentum tuning failed.

**Sources**: `train.py` transform definition; `reports/exp-report-026.md`; `reports/exp-report-064.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: The scale change may be too large for the tuned `LR=0.1` recipe and could underperform despite BatchNorm after the first convolution. Worst case is a valid no-improvement run; code risk is low because the change is localized to the transform constants.

### 2. CIFAR AutoAugment on the CutMix Anchor
**Summary**: Add torchvision's CIFAR AutoAugment policy after crop/flip and before tensor conversion while keeping the EXP-064 CutMix and optimizer anchor unchanged.

**Reasoning**: CutMix is the only recent augmentation mechanism to improve the baseline. A targeted CIFAR policy might add complementary invariance without erasing pixels like Cutout or globally interpolating labels like mixup. EXP-064 listed AutoAugment as a lower-confidence follow-up after CutMix brackets were exhausted, which is now true.

**Sources**: `reports/exp-report-064.md`; `reports/exp-report-044.md`; `knowledge/papers/randaugment-augmentation.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: Prior RandAugment and ColorJitter probes underperformed, and policy augmentation can add CPU overhead or excessive regularization. Combining it with CutMix may over-regularize rather than improve.

### 3. Fan-Out Kaiming Initialization for Conv Layers
**Summary**: Change convolution initialization to explicit ReLU Kaiming normal with `mode="fan_out"` while leaving linear initialization, architecture, optimizer, transforms, CutMix, and schedule unchanged.

**Reasoning**: The local code uses default `init.kaiming_normal_(m.weight)`, which defaults to fan-in mode. Many residual CNN recipes use fan-out mode for convolutional layers to preserve backward variance. This is a narrow initialization-only probe that avoids the repeatedly failed augmentation/schedule/regularization families.

**Sources**: `train.py` `_weights_init`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; prior residual-initialization failures in EXP-028/051 as cautionary context.

**Estimated Effort**: low

**Risk Assessment**: Initialization-only changes can be noisy and may be dominated by the fixed seed and short training budget. Prior residual initialization tweaks failed, though this is less invasive than changing BN scale.

## Idea Evaluation

Standard CIFAR channel-std normalization has the best balance of evidence, novelty, and failure mode. It targets input conditioning with no extra CPU transform cost, no parameter-count change, no evaluation-harness change, and no new dependency. It also avoids directly retrying recurring failed families. The main risk is that the existing LR/BN setup is already tuned around unit std, but a single clean run will answer that cheaply.

CIFAR AutoAugment is more speculative. It is now admissible because CutMix brackets are exhausted, but prior policy and photometric augmentation failures lower the prior, and adding it on top of CutMix may simply over-regularize. Fan-out Kaiming initialization is clean and cheap, but the expected effect is smaller and prior residual-initialization probes were negative.

## Chosen Idea
**Selected**: Standard CIFAR Channel-Std Normalization

**Why this idea**:
It is the most distinct remaining one-file change with a plausible conditioning mechanism and minimal runtime risk. It preserves the validated CutMix anchor while testing whether the recipe's unusual unit-std input scaling is leaving accuracy on the table.

**Hypothesis**:
If unscaled channel variance is limiting optimization or first-layer conditioning, switching to standard CIFAR-10 channel standard deviations will improve `best_test_acc` from 94.11% to at least 94.21% without violating the fixed-budget or single-file constraints.
