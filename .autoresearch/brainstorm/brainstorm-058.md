# Brainstorm EXP-058
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Squeeze-and-Excitation Networks** (CVPR 2018, https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)
  The paper proposes a lightweight channel recalibration block that models channel interdependencies and reports broad accuracy gains for existing CNN architectures with modest added compute. This is relevant because the current local history has exhausted many scalar recipe tweaks while leaving channel-attention architecture changes untested.
- **Existing knowledge base** (`knowledge/README.md`)
  Existing entries cover CIFAR augmentation, mixup, cosine schedules, wide residual networks, stochastic depth, EMA, residual initialization, and PyTorch throughput. There is no existing SE/channel-attention entry, so EXP-058 fills a distinct architecture gap rather than repeating a known failed family.

## Experimental History Review

- Current best remains EXP-038 at `best_test_acc=93.97%`; because the goal requires at least +0.10 percentage points, EXP-058 must reach `94.07%` to count as an improvement.
- The current anchor is `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, full-run label smoothing 0.05, FP32 compile, channels-last, and once-per-epoch validation.
- Recent failures strongly discourage more isolated scalar recipe tweaks: EXP-039/041 bracketed weight decay below the anchor, EXP-040/043 bracketed LR, EXP-046/052 tested cosine schedules, EXP-036/053 bracketed batch size, and EXP-033/037/057 showed label-smoothing deviations fail.
- Direct regularizers and augmentations have also lagged: stochastic depth reached 93.40%, mild mixup 93.85%, RandAugment 93.83%, clean ColorJitter 93.49%, and post-drop hard labels 93.42%.
- Architecture space is less exhausted than recipe space. Width expansion beyond 28/56/112 is a recurring failure and projection shortcuts hurt, but lightweight feature recalibration inside the existing width/depth has not been tested.
- The active model's `BasicBlock` has a natural insertion point after `bn2(conv2)` and before residual addition. This can test channel attention without changing depth, shortcut type, augmentation, optimizer, schedule, or evaluation protocol.

## Candidate Ideas

### 1. Squeeze-and-Excitation BasicBlocks
**Summary**: Add a small `SEBlock` to every `BasicBlock`, inserted after `self.bn2(self.conv2(out))` and before the shortcut addition. Use global average pooling and two `1x1` convolutions or linear layers with a conservative reduction ratio such as 16, preserving all current optimizer, augmentation, schedule, loss, width, depth, compile, and channels-last settings.

**Reasoning**: The SE paper directly targets representational quality through channel-wise feature recalibration rather than more scalar regularization. This is a distinct architecture mechanism from failed width expansion and projection shortcuts: it keeps the 28/56/112 backbone and option-A shortcut but lets each residual branch emphasize useful channels. The compute and parameter increase should be small relative to a full width/depth increase, and the worst likely outcome is a valid no-improvement if overhead or overfitting outweighs the representational gain.

**Sources**: CVPR 2018 SE paper URL above; current `train.py` `BasicBlock`; goal learnings showing failed scalar recipe/regularizer families and untested channel attention.

**Estimated Effort**: medium

**Risk Assessment**: Added operations may reduce step budget or interact poorly with `torch.compile` on this small CNN. Extra parameters and gating can also overfit under the fixed 300s budget. The implementation is still contained to `train.py` and should fail as no-improvement rather than invalid if preflight passes.

### 2. Average-Pool Option-A Downsample Shortcut
**Summary**: Replace the current strided slicing shortcut in downsampling blocks with average pooling before zero-channel padding, preserving the option-A no-parameter shortcut shape and every other anchor setting.

**Reasoning**: Projection shortcuts previously hurt, but the current stride-2 shortcut discards three-quarters of spatial positions by slicing. Average pooling could preserve more local information without adding learned parameters or changing the residual branch. This probes transition quality while avoiding the already-failed learned projection family.

**Sources**: EXP-018 projection shortcut failure; current `train.py` option-A shortcut; goal learnings noting width and projection failures.

**Estimated Effort**: low

**Risk Assessment**: This may be too close to the projection/shortcut family that already underperformed, and average pooling may add overhead without improving accuracy. It also changes a sensitive ResNet-CIFAR convention with weaker external evidence than SE.

### 3. Tiny Classifier Head Dropout
**Summary**: Add very small dropout, such as `p=0.05`, after global average pooling and before `self.fc`, preserving the rest of the anchor.

**Reasoning**: This is a cheap final-head regularization probe that does not touch data loading or residual blocks. It could reduce classifier co-adaptation while preserving the validated backbone recipe.

**Sources**: Current `train.py` classifier head; recent regularizer failures EXP-054 and EXP-055 as cautionary evidence.

**Estimated Effort**: low

**Risk Assessment**: Recent isolated regularizers have consistently lagged the anchor, and the current recipe already uses full-run label smoothing and stronger weight decay. This is likely lower-upside than an architecture mechanism.

## Idea Evaluation

Squeeze-and-Excitation BasicBlocks have the strongest evidence and clearest new mechanism. The primary source argues for channel recalibration as a lightweight architectural unit that improves existing CNNs, and the local experiment history has not yet tested this family. It also avoids the high-confidence failed areas: scalar LR/decay/smoothing brackets, batch-size deviations, cosine tails, EMA, mixup, stochastic depth, and photometric augmentation.

Average-Pool Option-A Downsample Shortcut is simpler and preserves the no-parameter shortcut constraint, but the evidence is weaker and the local shortcut family already has one negative signal from EXP-018. It is better kept as a backup if channel attention is too slow or fails cleanly.

Tiny Classifier Head Dropout is the cheapest implementation, but it is another isolated regularizer in a regime where isolated regularizers have repeatedly underperformed. Its expected impact is lower than a residual-block representation change.

The lead candidate is therefore the SE block architecture probe: it is a meaningful `train.py`-only model change, supported by primary literature, and distinct from the repeated local no-improvement families.

## Chosen Idea
**Selected**: Squeeze-and-Excitation BasicBlocks

**Why this idea**:
It targets a still-open architecture mechanism, channel-wise feature recalibration, without changing the proven training anchors. Compared with more regularization or schedule tweaks, it has better external support and avoids the recurring local failure modes.

**Hypothesis**:
Adding lightweight SE gates to each residual block will improve CIFAR-10 representation quality enough to reach at least `94.07%` best test accuracy, while preserving a valid single-GPU run, the first LR drop, and the fixed evaluation protocol.
