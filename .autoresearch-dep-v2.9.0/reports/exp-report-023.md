# Report EXP-023: Cubic-Scheduled EMA (Lookahead)
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-023.md
- **Plan**: plans/plan-023.md
- **Log**: logs/exp-log-023.md

## Goal

Maximize CIFAR-10 test accuracy (best_test_acc, %, higher is better). Baseline: 96.46% (EXP-020). Threshold for improvement: >96.56% (baseline + 0.1pp).

## Idea & Hypothesis

Cubic-scheduled EMA (Lookahead) from airbench96: maintain a shadow copy of the full model state_dict (including BN buffers), updating every 5 steps with `decay = 0.95^5 * (step/total_steps)^3`. The cubic ramp starts near-zero (no interference during high-LR exploration) and grows to ~0.77 by training end (strong late-training averaging). Selected because EXP-014 proved full state_dict EMA works mechanically but identified constant β=0.999 as too conservative for short training. Hypothesis: +0.1-0.3pp from noise-robust weight averaging during the polish phase.

## Approach

Added a `LookaheadEMA` class (~25 lines) after the model definition in train.py. The class deep-copies model.state_dict() as shadow, with `update()`, `swap()`, `restore()`, and `copy_to()` methods. Integrated into training loop: every 5 steps after scheduler.step(), `ema.update(model, step, total_steps)` is called. Before per-epoch TTA evaluation, EMA weights are swapped in and restored after. On final epoch, `copy_to()` is used for permanent EMA weight application. No deviations from plan. One bug fix: initial `update()` method signature omitted `model` parameter (referenced as free variable).

## Execution

Single local run on H20 GPU. Training completed 95 epochs in 300.0s (vs baseline ~99 epochs — ~4 epoch throughput cost from EMA overhead). No infrastructure errors or retries needed.

## Results

- **Primary metric**: 96.02% (baseline: 96.46%, delta: -0.44pp, -0.46%)
- **Observations**: Early-epoch EMA evaluations showed near-random accuracy (6.14% at epoch 1), indicating the shadow severely lagged model weights during high-LR warmup. EMA accuracy gradually caught up (~88% by epoch 20, ~95.4% by epoch 85+) but never matched baseline. The cubic schedule `alpha = 0.95^5 * (step/total_steps)^3` starts near-zero, meaning early steps barely update the shadow — the shadow effectively remains at initialization quality for many epochs. Combined with ~4 epoch throughput cost (95 vs 99 epochs), net effect was strongly negative.
- **Analysis**: The hypothesis was wrong. The cubic ramp that was supposed to avoid early-training interference instead caused catastrophic shadow lag — the shadow remained close to initialization while the model trained rapidly under cosine warmup+decay. By the time the cubic factor reaches meaningful values (late training), the shadow has accumulated a large deficit that cannot be closed. The airbench96 recipe works at 37 epochs with a very different architecture and training regime; the cubic schedule does not transfer to our 99-epoch cosine regime.
- **Key Learning**: EMA approaches are exhausted for this model/budget — three variants (parameter-only, constant β, cubic schedule) all failed for different but compounding reasons: BN mismatch, over-conservative β, and catastrophic shadow lag.

## Verification

- **Conditions**: Condition 1 FAILED (96.02% < 96.56%), Conditions 2-3 PASSED (clean completion, validation frequency correct)
- **Review Notes**: Results confirmed trustworthy — metrics consistent across log output, summary block complete, 95 epochs within expected range for EMA overhead
- **Verdict**: no-improvement
- **Verdict Basis**: Primary metric 96.02% failed to exceed 96.56% threshold (Condition 1 failed)

## Unexplored Avenues

- **Delayed-start EMA**: Initialize shadow only after warmup phase (epoch 5+) rather than at step 0. This avoids the shadow-lag problem by not accumulating error during the highest-LR phase. However, the throughput cost remains (~4 epochs lost to state_dict copy/lerp).
- **EMA with higher constant β (e.g., 0.99 or 0.995)**: Lower than 0.999 (EXP-014) but not schedule-dependent. May balance smoothing vs. tracking better for 99-epoch training. However, EXP-014 showed even the mechanics of per-step EMA cost ~6 epochs.

Given three failed EMA variants addressing different failure modes, the approach class is likely exhausted for this model/budget combination.

## Next Steps
- **BN Bias 64x LR** (medium confidence): Completely untried, zero throughput cost. airbench96-validated technique. Risk: 64x was tuned for a very different training regime — the multiplier may need adjustment for our setup.
- **Deeper architecture (NUM_BLOCKS=4, ResNet-26)** (low-medium confidence): Adds ~33% more parameters for increased capacity. Risk: ~25% throughput loss (74 vs 99 epochs) may negate capacity gain.
- **Knowledge distillation from a larger pretrained model** (low confidence): Fundamentally different training signal — soft targets from a teacher model could provide more information than hard labels. Risk: requires a pretrained teacher and careful temperature tuning.
