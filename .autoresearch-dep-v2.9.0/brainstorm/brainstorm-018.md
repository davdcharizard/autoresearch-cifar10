# Brainstorm EXP-018
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Deep Networks with Stochastic Depth** (Huang et al. 2016, https://arxiv.org/abs/1603.09382)
  Stochastic depth randomly drops entire residual blocks during training with linearly increasing drop probability. Reports ~15% test error reduction on CIFAR-10 ResNets, acts as both regularizer and implicit ensemble. For shallow nets (9 blocks), survival probability p_L=0.8-0.9 at the deepest layer is appropriate (p_L=0.5 standard for 110+ layer nets). At test time, all layers are kept with deterministic output.

- **hlb-CIFAR10 speedrun** (tysam-code, GitHub)
  Achieves 95.79% in ~110s on A100. Uses TTA with horizontal flip averaging: `(model(ims) + model(ch.fliplr(ims))) / 2`. Also uses 12-pixel Cutout, batch 1024, architecture scaling. TTA is zero training cost.

- **cifar10-airbench** (KellerJordan, GitHub)
  Achieves 96% in 46.3s on A100. Uses 12-pixel Cutout (larger than standard RandomErasing), Muon optimizer, data filtering, architecture scaling. Key insight: larger occlusion regions may be more effective than our current RandomErasing(scale=(0.02, 0.2)).

- **PyTorch stochastic depth implementation** (torchvision.ops.stochastic_depth)
  Linear decay formula: survival_prob = 1 - (l/L) * (1 - p_L) where l is block index, L is total blocks. Implementation: multiply residual branch output by Bernoulli sample / survival_prob during training; identity at eval.

## Experimental History Review

- **Current best**: 95.57% (EXP-015, commit 626e9d1) — label smoothing 0.2 on top of WIDTH_MULT=4 + AMP + batch 256 + TrivialAugmentWide + RandomErasing + WD=5e-4
- **Trajectory** (8 improvements from 19 experiments): BASE 91.72 → EXP-001 92.29 → EXP-002 92.92 → EXP-003 93.33 → EXP-005 94.44 → EXP-007 94.82 → EXP-009 95.39 → EXP-015 95.57
- **Exhausted approaches**: SE blocks (count 2, intrinsic 9ms/step overhead), torch.compile (zero speedup on H20), BN momentum tuning (already converged at ~98 epochs), shifted LR drops (0.5/0.75 near-optimal)
- **Failed but not exhausted**: Full state_dict EMA (β=0.999 too conservative, goal-learnings says "lower β + efficient implementation may work"), Mixup (destabilizes polish phase; mWh scheduling could help), CutMix (over-regularized when stacked; lower α or replace instead of stack could work)
- **What hasn't been tried**: Stochastic depth / DropPath (structural regularizer), test-time augmentation, larger occlusion (Cutout-style), knowledge distillation, cosine annealing with correct T_max, gradient clipping, different optimizers (AdamW, Muon)
- **Key patterns**: wall-clock-fractional schedule at 0.5/0.75 is optimal; ~96-98 epochs in 300s at batch 256; label smoothing 0.2 composes cleanly with input-space augmentation; throughput is the binding constraint

## Candidate Ideas

### 1. Stochastic Depth (DropPath) on BasicBlock
**Summary**: Add stochastic depth to the 9 BasicBlocks of ResNet-20, with linearly decaying survival probability from 1.0 (first block) to p_L=0.9 (last block). During training, each block's residual branch is randomly dropped with probability 1 - p_l, and the output is scaled by 1/p_l. During eval, all blocks are active with no scaling. This is a structural regularizer orthogonal to all current input-space augmentation (TrivialAugmentWide, RandomErasing) and output regularization (label smoothing). Implementation: modify BasicBlock.forward() to apply dropout to the residual branch before addition; add a `drop_path_rate` parameter; assign linearly spaced rates in ResNet.__init__().

**Reasoning**: Stochastic depth is specifically designed for residual networks and has strong empirical support (Huang et al. 2016). It provides regularization through a completely different mechanism than our current stack — randomly shortening the effective network depth rather than augmenting inputs or smoothing labels. The implicit ensemble effect (training exponentially many sub-networks) is unique among our untried approaches. With p_L=0.9 (only 10% max drop rate), the regularization is mild enough to avoid the over-regularization trap that killed CutMix (EXP-010) and Mixup (EXP-017). Dropped blocks skip computation, providing a minor throughput improvement (~1-3% fewer FLOPs on average) that could yield 1-2 extra epochs in the 300s budget. EXP-017 report explicitly recommended this as next step #1.

**Sources**: Huang et al. 2016 "Deep Networks with Stochastic Depth"; torchvision.ops.stochastic_depth implementation; EXP-017 report § Next Steps; goal-learnings § Failed Approaches (SE blocks exhausted — stochastic depth is a different structural regularization axis)

**Estimated Effort**: low — ~20 lines of code change in BasicBlock and ResNet classes

**Risk Assessment**: With p_L=0.9, the regularization is very mild. Worst case: negligible effect (+0.0 to -0.05pp) since 9 blocks is quite shallow for stochastic depth (the technique was designed for 110+ layer nets). The implicit ensemble benefit may be limited with only 9 blocks producing 2^9=512 sub-networks. Risk of over-regularization is low given the mild drop rates. No throughput risk — dropped blocks are strictly cheaper.

### 2. Full State Dict EMA with β=0.995 and In-Place Updates
**Summary**: Maintain an exponential moving average of the full model state_dict (including BatchNorm buffers) with β=0.995, using in-place tensor operations instead of deepcopy for efficiency. After training completes, swap the EMA weights into the model for evaluation. Key differences from EXP-014 (β=0.999): (1) lower β=0.995 tracks the ~96-epoch trajectory better — effective averaging window ~200 steps vs ~1000 steps; (2) in-place `shadow[k].lerp_(param, 1-β)` avoids per-step state_dict() + deepcopy overhead that cost EXP-014 ~6 epochs.

**Reasoning**: Goal-learnings explicitly notes EMA is "not exhausted — lower β + efficient implementation may work." EXP-014's +0.05pp came despite two handicaps: β=0.999 was too conservative for ~92 epochs (averaging over too long a window smooths away recent improvements), and the naive implementation cost ~6 epochs of throughput. β=0.995 with ~96 epochs means the effective averaging window is ~200 updates — enough to smooth noise but responsive enough to track the learning trajectory through LR drops. In-place lerp on pre-cached tensor references eliminates the per-step state_dict() overhead entirely.

**Sources**: EXP-014 report; goal-learnings § Failed Approaches "Full state_dict EMA β=0.999" (explicitly says not exhausted); EXP-017 report § Next Steps recommendation #2

**Estimated Effort**: medium — EMA shadow initialization, per-step in-place update loop, post-training weight swap, careful handling of BN buffers

**Risk Assessment**: β=0.995 may still be too conservative or too aggressive — the optimal β depends on the noise level which is hard to predict. The throughput cost should be minimal with in-place ops (lerp on ~4.3M params is fast), but if it costs even 2-3 epochs, the smoothing benefit may not compensate. If β is wrong, worst case is a small regression similar to EXP-014 (+0.05pp to -0.1pp range). The approach is well-understood and implementation is straightforward.

### 3. Cutout with Larger Occlusion Region
**Summary**: Replace RandomErasing(p=0.25, scale=(0.02, 0.2)) with a Cutout-style augmentation using a larger fixed-size occlusion region (12×12 pixels, following cifar10-airbench). Cutout (DeVries & Taylor 2017) applies a fixed-size square mask at a random position, zeroing out the pixels. The key difference from current RandomErasing: (1) larger occlusion — 12×12 = 144 pixels = ~14% of image vs RandomErasing's variable 0.02-0.2 scale; (2) fixed square shape vs variable aspect ratio; (3) deterministic size for more consistent regularization signal.

**Reasoning**: cifar10-airbench achieves 96% using 12-pixel Cutout as a key component. Our current RandomErasing uses relatively small occlusion regions (scale 0.02-0.2 means erasing 2-20% of the image area with variable shape). Larger, fixed-size occlusion forces the model to rely on more distributed features rather than local patterns. This is a direct replacement (not stacking), avoiding the over-regularization trap from EXP-010. The technique is well-validated on CIFAR-10 specifically. Implementation is simple: `transforms.RandomErasing(p=1.0, scale=(0.14, 0.14), ratio=(1.0, 1.0), value=0)` approximates Cutout, or a custom transform for exact behavior.

**Sources**: cifar10-airbench (KellerJordan); DeVries & Taylor 2017 "Improved Regularization of Convolutional Neural Networks with Cutout"; EXP-017 analysis (RandomErasing > mild Mixup, but larger occlusion untried)

**Estimated Effort**: low — single line change in augmentation pipeline

**Risk Assessment**: Larger occlusion with p=1.0 (always applied) combined with TrivialAugmentWide may over-regularize — similar risk to EXP-010's CutMix failure. However, Cutout is simpler than CutMix (no cross-sample mixing, no label modification) and occlusion-based rather than interpolation-based, which aligns with the finding that per-sample occlusion works better than cross-sample methods (EXP-017). Risk is moderate: if too aggressive, accuracy could drop 0.1-0.3pp.

## Idea Evaluation

**Evidence strength**: Stochastic depth has the strongest theoretical and empirical backing (Huang et al. 2016, peer-reviewed, specifically for ResNets). EMA has project-specific evidence that the idea is "not exhausted" but the prior attempt was weak (+0.05pp). Cutout has strong external evidence from cifar10-airbench but it's a different architecture/setup.

**Mechanism clarity**: Stochastic depth has the clearest causal mechanism — implicit ensemble of 2^9 sub-networks + structural regularization orthogonal to all existing techniques. EMA smooths weight noise in the final model — clear mechanism but magnitude depends on β tuning. Cutout forces distributed feature learning — clear but overlaps with existing RandomErasing's mechanism.

**Expected impact**: EMA with correct β has the highest ceiling if it works (EXP-014 showed +0.05pp with wrong β; correct β + efficient implementation could yield +0.1-0.3pp). Stochastic depth is likely modest for 9 blocks (+0.05-0.15pp range). Cutout is uncertain — could be +0.1-0.2pp or could over-regularize.

**Risk profile**: Stochastic depth has the safest failure mode — with p_L=0.9, worst case is negligible effect, unlikely to hurt. EMA is also safe — worst case is small throughput cost for no gain. Cutout has moderate risk of over-regularization.

**Feasibility**: Stochastic depth and Cutout are both low effort. EMA is medium effort but well-understood.

Stochastic depth offers the best risk-adjusted profile: orthogonal regularization mechanism, safe failure mode, low implementation effort, potential minor throughput gain, and it was the #1 recommended next step from EXP-017. While EMA has a higher ceiling if β is tuned correctly, the uncertainty around the right β makes it riskier for a single experiment. Stochastic depth's mechanism is more predictable.

## Chosen Idea
**Selected**: Stochastic Depth (DropPath) on BasicBlock

**Why this idea**:
Stochastic depth provides a regularization mechanism completely orthogonal to everything in the current training pipeline (input augmentation, label smoothing, weight decay). It has the strongest evidence base (peer-reviewed, specifically designed for ResNets), the safest failure mode (mild p_L=0.9 cannot over-regularize), and was the consensus #1 next step from the prior experiment. The low implementation effort and potential minor throughput bonus (from dropped blocks) make it the best risk-adjusted choice.

**Hypothesis**:
Adding stochastic depth with p_L=0.9 (maximum 10% drop probability at the deepest block) will improve best_test_acc by 0.1-0.2pp to ≥95.67%, through structural regularization and implicit ensemble effects that are orthogonal to the existing input-space augmentation and output regularization. The mild drop rates will avoid over-regularization, and the reduced computation from dropped blocks will provide a minor throughput improvement (~1-2 extra training iterations per epoch).
