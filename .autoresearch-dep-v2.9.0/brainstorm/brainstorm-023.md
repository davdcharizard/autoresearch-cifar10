# Brainstorm EXP-023
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **airbench96 source code** (https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench96.py)
  Key EMA technique: "Lookahead" with cubic decay schedule `alpha = 0.95^5 * (step/total_steps)^3`, updates every 5 steps, interpolates ALL parameters (including BN buffers via state_dict). Decay starts near-zero and ramps cubically — minimal early interference, strong late averaging. Final update uses decay=1.0 (full copy). This is fundamentally different from EXP-014's constant β=0.999 which over-smoothed early training.

- **airbench96 BN bias 64x LR** (https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench96.py)
  Separate param group for BN biases with `lr_biases = lr * 64.0` (bias_scaler). BN scale is fixed at 1.0 (not trainable), only bias is trained with massively higher LR. Novel technique that accelerates BN adaptation. Zero throughput cost — just a param group change.

- **airbench96 architecture and training** (https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench96.py)
  3 ConvGroup blocks, widths 128→384→512, 10 conv layers total, Nesterov momentum 0.85, WD 0.012, batch 1024, 37 epochs. BN momentum 0.6, fixed scale=1.0. LR: 10% warmup, linear decay to zero. Deterministic augmentation (alternating flip, 4px translate, 12px cutout).

- **Wide ResNet research** (Zagoruyko & Komodakis, 2016; DeepWiki summary)
  Width > depth beyond a certain point for CIFAR-10. WRN-28-10 optimal. Diminishing returns from depth after ~28 layers. WRN-40-8 loses to WRN-22-8, confirming width matters more. However, throughput constraints at 300s budget limit how wide/deep we can go.

## Experimental History Review

**Current best**: 96.46% (EXP-020, cosine LR schedule on WIDTH_MULT=4 ResNet-20)

**Improvement trajectory** (10 successful experiments): BASE 91.72 → width-2x 92.29 → augmentation 92.92 → WD 93.33 → AMP 94.44 → width-4x 94.82 → batch-256 95.39 → label-smoothing 95.57 → TTA 95.91 → cosine-LR 96.46

**Exhausted approaches** (regularization stack saturated):
- CutMix, Mixup, DropPath, Cutout-swap all failed — adding/swapping regularization at 96.46% is in the noise floor (EXP-010, 017, 018, 022)
- SE blocks: ~9ms/step intrinsic overhead regardless of implementation (EXP-011, 012) — exhausted
- Pre-activation blocks: ~6% throughput regression from lost cuDNN fusion (EXP-021)
- torch.compile: zero speedup on H20 for this model (EXP-008)
- BN momentum 0.5: +0.02pp, noise (EXP-016)
- LR schedule shifting: earlier drops hurt (EXP-006)

**Partially explored (idea not exhausted)**:
- EMA: Parameter-only EMA failed due to BN mismatch (EXP-013). Full state_dict EMA with constant β=0.999 yielded only +0.05pp (EXP-014) — but β was too conservative for ~92 epochs and per-step overhead cost ~6 epochs. A scheduled EMA with cubic decay (as in airbench96) is fundamentally different and untried.
- Deeper architecture: NUM_BLOCKS=4 (ResNet-26) untried. +33% more conv layers for increased capacity.
- BN bias high-LR: Completely untried. airbench96-validated technique.

**Key patterns to exploit**:
- Cosine decay to ~0 LR is validated as default schedule (EXP-020)
- ~99 epochs complete in 300s budget at current config
- Throughput-to-accuracy conversion remains strong — each additional epoch contributes meaningfully
- Output-distribution regularization (label smoothing) composes cleanly with input augmentation

## Candidate Ideas

### 1. Cubic-Scheduled EMA (Lookahead)
**Summary**: Implement airbench96-style Lookahead EMA with cubic decay schedule. Every 5 training steps, interpolate the EMA shadow toward the current model weights using `decay = 0.95^5 * (step/total_steps)^3`. Use full state_dict (not just parameters) to include BN running stats. At training end, copy EMA weights fully (decay=1.0). This differs from EXP-014's constant β=0.999 in three critical ways: (1) cubic ramp means near-zero smoothing early, strong smoothing late; (2) updates every 5 steps instead of every step, reducing overhead; (3) the schedule is tuned for short training regimes.

**Reasoning**: EXP-014 showed full state_dict EMA is mechanically correct but β=0.999 is too conservative for ~92 epochs — the learning in goal-learnings says "β must be tuned to epoch count (lower β for shorter training)." The cubic schedule directly addresses this: decay starts near-zero (no interference during high-LR exploration) and ramps to ~0.77 by end of training (strong averaging during polish phase). airbench96 achieves 96.05% in 37 epochs with this technique as one component. The every-5-steps update cadence reduces per-step overhead from ~1ms (EXP-014) to ~0.2ms equivalent.

**Sources**: airbench96 source code (Lookahead implementation), EXP-014 report (full state_dict EMA +0.05pp with constant β), goal-learnings § Failed Approaches (EMA entries)

**Estimated Effort**: low — ~30 lines of code for the EMA class and schedule integration

**Risk Assessment**: Main risk is throughput cost: even with every-5-steps updates, state_dict copy/lerp on a 4.3M parameter model may cost 1-2 epochs over 300s. If the smoothing benefit is <0.1pp (as with EXP-014), the net effect could be zero or negative. Mitigation: the cubic schedule should produce much stronger late-training smoothing than constant β.

### 2. Deeper Architecture (NUM_BLOCKS=4, ResNet-26)
**Summary**: Increase NUM_BLOCKS from 3 to 4, creating a ResNet-26 (6×4+2=26 layers). This adds one BasicBlock per stage (3 stages × 1 extra block = 3 extra blocks, 6 extra conv layers). At WIDTH_MULT=4, this increases parameters from ~4.3M to ~5.7M (+33%). The model gains representational capacity through depth while maintaining the same width structure.

**Reasoning**: The model may be capacity-limited at 96.46%. Width scaling from 1x to 4x produced large gains (91.72→94.82 across several experiments). Depth is the next capacity dimension to explore. Wide ResNet research confirms width > depth, but our model is already fairly wide (channels 64/128/256) — adding depth should provide complementary capacity. EXP-022 report's Next Steps lists this as medium confidence. The risk is throughput: ~25-30% more FLOPs means ~74-80 epochs instead of 99.

**Sources**: Wide ResNet (Zagoruyko & Komodakis, 2016), EXP-022 report § Next Steps, goal-learnings § Patterns (throughput-to-accuracy)

**Estimated Effort**: low — single hyperparameter change (NUM_BLOCKS = 4)

**Risk Assessment**: ~25-30% throughput loss is the primary concern. At 99 epochs → ~74 epochs, the cosine schedule would need ESTIMATED_EPOCHS adjusted. The capacity gain may not compensate for fewer training iterations in the 300s budget. Pre-activation ResNet showed that throughput regression directly hurts accuracy (EXP-021: -0.23pp from 6% throughput loss). A 25% throughput loss is much larger.

### 3. BN Bias 64x LR (Separate Param Group)
**Summary**: Create a separate optimizer param group for BatchNorm bias parameters with 64× the base learning rate (LR_biases = 0.2 × 64 = 12.8). Keep BN weight (scale) at normal LR. This accelerates BN's learned shift (bias) adaptation while the scale parameter trains normally. Zero additional computation — just a param group configuration change.

**Reasoning**: airbench96 uses this technique (`bias_scaler = 64.0`) as part of its 96.05% recipe. The rationale is that BN biases need to adapt quickly to shifting feature distributions during training — a higher LR lets them track distribution shifts faster, improving optimization dynamics especially during warmup and LR transitions. This is completely untried in this project. EXP-016 showed BN momentum tuning is ineffective at ~98 epochs, but BN bias LR is a different mechanism — it affects the learned parameters, not the running statistics.

**Sources**: airbench96 source code (bias_scaler=64.0), EXP-016 (BN momentum noise — different mechanism)

**Estimated Effort**: low — ~10 lines to create separate param groups

**Risk Assessment**: The main risk is that 64x was tuned for airbench96's very different architecture (batch 1024, Nesterov 0.85, WD 0.012, 37 epochs, different widths). Our training regime differs substantially: batch 256, SGD momentum 0.9, WD 5e-4, 99 epochs. The optimal bias LR multiplier may be very different. A 64x multiplier with our LR=0.2 gives bias_LR=12.8, which is extremely high and could cause instability. The technique might need significant tuning of the multiplier for our setup.

## Idea Evaluation

**Evidence strength**: Cubic EMA has the strongest evidence — EXP-014 already proved full state_dict EMA works mechanically in this codebase (+0.05pp), and the specific failure mode (β too conservative) has a clear fix (cubic schedule). The airbench96 recipe validates the cubic schedule in a comparable CIFAR-10 context. BN bias 64x LR has airbench96 validation but in a very different training regime — the optimal multiplier is uncertain. Deeper architecture has general Wide ResNet evidence but no direct evidence for our throughput-constrained setting.

**Mechanism clarity**: Cubic EMA has the clearest mechanism: minimal smoothing during exploration (cubic starts near-zero), strong smoothing during polish (ramps to ~0.77). This directly targets the late-training noise that limits accuracy. BN bias 64x LR's mechanism is less clear for our setup — faster bias adaptation helps, but 64x may overshoot. Deeper architecture's mechanism (more capacity) is straightforward but the throughput cost creates a competing negative mechanism.

**Expected impact**: Cubic EMA is most likely to provide a small but meaningful gain (+0.1-0.3pp) because it targets late-training averaging which is orthogonal to all current techniques. Deeper architecture has higher upside IF throughput cost is manageable, but the 25% throughput loss is a major headwind. BN bias 64x LR's impact is hardest to predict due to regime mismatch.

**Risk profile**: BN bias 64x LR has the safest failure mode (worst case: no improvement, easily reverted). Cubic EMA is also safe (worst case: slight throughput loss cancels benefit). Deeper architecture has the riskiest profile — 25% throughput regression could cause a net negative.

**Feasibility**: All three are low effort (param group changes or hyperparameter change). Cubic EMA requires ~30 lines but is straightforward. BN bias 64x LR is ~10 lines. Deeper architecture is 1 line.

## Chosen Idea
**Selected**: Cubic-Scheduled EMA (Lookahead)

**Why this idea**:
Strongest evidence chain: EXP-014 proved full state_dict EMA works mechanically (+0.05pp), identified constant β=0.999 as the specific failure mode, and goal-learnings explicitly noted "lower β for shorter training" as the fix. The airbench96 cubic schedule is a validated, production-tested solution to exactly this problem — cubic ramp provides near-zero early smoothing and strong late smoothing, tuned for short training. The mechanism is clear, the failure mode is safe (worst case: slight throughput cost), and it's orthogonal to all existing techniques in the stack.

**Hypothesis**:
Cubic-scheduled EMA (Lookahead) with full state_dict, updating every 5 steps with `decay = 0.95^5 * (step/total_steps)^3`, will improve best_test_acc by +0.1-0.3pp (targeting 96.56-96.76%) by providing noise-robust weight averaging during the polish phase of cosine LR decay. The cubic ramp avoids the early-training interference that made constant β=0.999 ineffective in EXP-014, while the every-5-steps cadence minimizes throughput cost to <1 epoch. The primary metric improvement will come from smoother convergence in the final 20-30% of training where the cosine schedule drives LR toward zero.
