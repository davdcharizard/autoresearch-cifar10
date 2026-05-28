# Brainstorm EXP-011
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Squeeze-and-Excitation Networks (Hu et al., CVPR 2018)** (https://arxiv.org/pdf/1709.01507)
  SE blocks add lightweight channel attention via global-avg-pool → FC(C, C/r) → ReLU → FC(C/r, C) → Sigmoid → channel-wise scale. On CIFAR-10, SE-ResNet-110 reduced error by 18.7% and SE-ResNet-164 by 14.1% over their non-SE counterparts. The overhead is minimal — two small FC layers per block with reduction ratio r=16.

- **RES-SE-NET: Boosting Performance of Resnets by Enhancing Bridge-connections** (https://arxiv.org/pdf/1902.06066)
  Confirms SE blocks consistently improve ResNet performance on CIFAR-10/100 across depths with "extremely small increase in computational complexity."

- **OneCycleLR / Super-Convergence (Smith & Topin, 2018)** (https://docs.pytorch.org/docs/2.8/generated/torch.optim.lr_scheduler.OneCycleLR.html)
  PyTorch's OneCycleLR implements the 1cycle policy: warmup to max_lr over pct_start fraction, then cosine-decay to near-zero. Default params: pct_start=0.3, div_factor=25, final_div_factor=10000. Smooth profile avoids discrete LR transitions. Reports show 92-94% on CIFAR-10 with ResNet in 30-50 epochs.

- **SWA: Averaging Weights Leads to Wider Optima and Better Generalization (Izmailov et al., 2018)** (https://arxiv.org/pdf/1803.05407)
  Stochastic Weight Averaging (SWA) averages model weights from multiple points along the SGD trajectory. Achieves +0.4% improvement on CIFAR-10 with ResNet-164, VGG-16, and WRN-28-10. Virtually zero computational overhead — only requires maintaining a running average of weights.

- **SGDR: Stochastic Gradient Descent with Warm Restarts (Loshchilov & Hutter, 2017)** (https://arxiv.org/pdf/1608.03983)
  Cosine annealing with warm restarts helps escape local minima. Multiple restart cycles let the optimizer explore different basins. Combined with SWA, this can be particularly effective.

## Experimental History Review

- **Current best**: 95.39% (EXP-009, batch-256 + LR 0.2 + 5-epoch warmup)
- **Trajectory**: BASE 91.72 → 92.29 (width-2x) → 92.92 (+aug) → 93.33 (+WD) → 94.44 (+AMP) → 94.82 (width-4x) → 95.39 (batch-256). 7 improvements across 11 experiments (000-010).
- **Primary improvement drivers**: Width scaling (+capacity), augmentation (+regularization), AMP (+throughput/epochs), batch scaling (+throughput). Each wave of improvement targets a different axis.
- **What worked**: Wall-clock-fractional MultiStepLR with drops at 0.5/0.75 (HIGH importance pattern — validated across model sizes). TrivialAugmentWide + RandomErasing is a free lunch. WD=5e-4 amplifies the second LR drop. AMP gives 1.54x throughput. Batch doubling yields 18% more epochs.
- **What failed**: CosineAnnealingLR with wrong T_max (EXP-000). Nesterov+label_smoothing per-step overhead (EXP-004). Shifted LR drops (EXP-006). torch.compile zero speedup on H20 (EXP-008). CutMix α=1.0 over-regularization (EXP-010).
- **Key constraint**: ~98 epochs in 300s at batch 256 on H20. Throughput-to-accuracy conversion remains strong — each additional epoch contributes meaningfully. Any change that reduces epoch count must compensate with per-epoch quality.
- **Untried approaches**: Architectural modifications (SE blocks, deeper models), OneCycleLR or cosine schedule, SWA, Mixup (lighter than CutMix), knowledge distillation.

## Candidate Ideas

### 1. Squeeze-and-Excitation (SE) Blocks in BasicBlock
**Summary**: Add SE channel attention to each BasicBlock in the ResNet-20 architecture. After the second conv+BN in each block (before the residual addition), insert a squeeze-and-excitation module: global average pooling → FC(C, C/16) → ReLU → FC(C/16, C) → Sigmoid → channel-wise multiplication. This recalibrates feature maps by learning channel interdependencies, allowing the network to emphasize informative features and suppress less useful ones — a genuinely new capability the current architecture lacks.

**Reasoning**: SE blocks have strong evidence across multiple studies (Hu et al. CVPR 2018, RES-SE-NET 2019) showing consistent accuracy improvements with minimal overhead. On CIFAR-10 specifically, SE reduced ResNet error rates by 14-19%. The current model (WIDTH_MULT=4, channels 64/128/256) has wide channels that provide a rich feature space for SE's channel attention to operate on. The added parameters are tiny: ~32K on top of ~4.29M (0.7% increase). The per-step compute overhead is two small FC layers and an element-wise multiply per block — likely <1ms per step, preserving the ~98 epoch count. This is an architectural improvement that adds a new mechanism rather than tweaking existing hyperparameters, making it orthogonal to all prior experiments.

**Sources**: Hu et al. 2018 (https://arxiv.org/pdf/1709.01507), RES-SE-NET (https://arxiv.org/pdf/1902.06066), goal-learnings patterns on throughput sensitivity

**Estimated Effort**: medium — requires adding an SE module class and modifying BasicBlock's forward pass. Straightforward implementation, well-documented pattern.

**Risk Assessment**: Main risk is per-step overhead reducing epoch count. With ~32K extra params and two tiny FC layers per block, this should be negligible (<1ms/step on 9-10ms baseline). Worst case: SE adds 1-2ms/step overhead, reducing epochs from 98 to ~85, which could negate the quality gain. The reduction ratio r=16 is standard; r=8 would be a fallback if r=16 is too aggressive for the 64-channel first layer (64/16=4 neurons in bottleneck).

### 2. OneCycleLR Schedule
**Summary**: Replace the wall-clock-fractional MultiStepLR with PyTorch's OneCycleLR. Configuration: max_lr=0.2 (current LR), total_steps estimated from prior epochs (~98 epochs × ~194 batches/epoch ≈ 19012 steps), pct_start=0.3 (30% warmup), anneal_strategy='cos', div_factor=25 (start at 0.008), final_div_factor=10000 (end near 0). This replaces both the manual 5-epoch warmup and the step-decay schedule with a single smooth cosine profile. The wall-clock adaptation is preserved by using total_steps rather than epochs.

**Reasoning**: OneCycleLR has been widely adopted for CIFAR-10 training and avoids the sharp LR=0.01 instability observed with AMP in EXP-005 (oscillations epochs 34-52). The smooth cosine decay ensures a gradual transition to low LR, which may be more compatible with FP16 training. The current step-decay wastes half the budget at full LR before any decay — OneCycleLR starts decaying after 30%, giving more time in the descending phase. However, the goal-learnings record the (0.5, 0.75) step-decay as HIGH importance and "near-optimal," so this is a deliberate departure from a validated pattern.

**Sources**: PyTorch OneCycleLR docs (https://docs.pytorch.org/docs/2.8/generated/torch.optim.lr_scheduler.OneCycleLR.html), EXP-005 report (AMP instability at LR=0.01), EXP-006 learnings (shifting drops earlier hurts), Smith & Topin 2018 (super-convergence)

**Estimated Effort**: low — replace LambdaLR with OneCycleLR, remove manual warmup. Minimal code changes.

**Risk Assessment**: The HIGH importance pattern says the (0.5, 0.75) step-decay schedule is critical. EXP-006 showed that shifting drops earlier (0.35/0.55) reduced accuracy by 0.27pp. OneCycleLR's different profile may not outperform the validated schedule. The total_steps estimate depends on epoch count being stable (~98), which is uncertain. Worst case: no-improvement similar to EXP-006, accuracy drops 0.2-0.5pp.

### 3. Stochastic Weight Averaging (SWA)
**Summary**: Add SWA to the last ~20% of training. After the second LR drop (at 75% of budget), begin averaging model weights using PyTorch's `torch.optim.swa_utils.AveragedModel` and `SWALR`. The SWA learning rate is set to a constant (e.g., 0.002, slightly above the final LR=0.002). At training end, update batch norm statistics on the averaged model before evaluation. This is applied on top of the existing training pipeline with no changes to the schedule during the first 75%.

**Reasoning**: Izmailov et al. (2018) showed SWA improves accuracy by ~0.4% on CIFAR-10 with various architectures by finding wider optima with better generalization. SWA has virtually zero computational overhead — it only maintains a running average of weights. The key advantage is that it's purely additive: it doesn't change the existing training recipe during the first 75%, only modifies the final 25% where the model is already in a good basin. This respects the validated (0.5, 0.75) schedule pattern. The only cost is a single BN update pass after training (a few seconds).

**Sources**: Izmailov et al. 2018 (https://arxiv.org/pdf/1803.05407), goal-learnings patterns on second LR drop importance, EXP-009 baseline trajectory

**Estimated Effort**: medium — requires importing swa_utils, creating AveragedModel, switching to SWALR at 75% budget, and running BN update. Moderate code changes but well-documented in PyTorch.

**Risk Assessment**: SWA literature reports +0.4% on CIFAR-10, but this was measured on standard training recipes, not on an already-heavy augmentation + AMP stack. The BN update at the end requires an extra forward pass through the training set, which adds ~5-10s overhead within the 300s budget (but this is after training, so it doesn't reduce epoch count — it only affects total wall clock). Risk: the averaging window may be too short (~20 epochs) to accumulate meaningful diversity. Worst case: marginal or no improvement, wasted implementation effort.

## Idea Evaluation

**Evidence strength**: SE blocks have the strongest external evidence — CVPR 2018 best paper with consistent 14-19% error reduction on CIFAR-10 ResNets specifically. SWA has solid evidence (+0.4% on CIFAR-10) but measured on standard recipes, not heavily augmented/AMP setups. OneCycleLR has broad adoption but the current step-decay is already validated as near-optimal (HIGH importance pattern).

**Mechanism clarity**: SE blocks have the clearest mechanism — channel attention recalibrates feature responses, providing a genuinely new capability the model currently lacks. This is orthogonal to all prior improvements (width, augmentation, throughput, batch scaling). OneCycleLR's mechanism is schedule smoothing, but the current schedule is already well-tuned. SWA's mechanism (weight averaging for flatter minima) is well-understood but the benefit is incremental.

**Expected impact**: SE blocks target model expressiveness — the one axis we haven't yet explored. Given that width scaling (EXP-001, 007) and regularization (EXP-002, 003) both delivered gains, adding attention within the existing width should compound. OneCycleLR is a schedule change competing against a validated schedule. SWA's expected +0.4% may be diminished by the already-strong augmentation stack.

**Risk profile**: SE blocks have the safest failure mode — even if the attention doesn't help, the model degrades gracefully (no-improvement, not crash). OneCycleLR risks violating the HIGH importance step-decay pattern. SWA is safe but likely incremental.

**Feasibility**: All three are implementable within a single loop. SE requires the most code (new module + BasicBlock modification) but is well-documented. OneCycleLR is simplest. SWA is moderate.

**Verdict**: SE blocks are the strongest candidate — they have the most evidence, the clearest mechanism, and target an untried axis (architectural expressiveness) rather than tweaking already-optimized hyperparameters.

## Chosen Idea
**Selected**: Squeeze-and-Excitation (SE) Blocks in BasicBlock

**Why this idea**:
SE blocks are the strongest candidate because they add a genuinely new capability (channel attention) that is orthogonal to all prior improvements. The evidence is robust (CVPR 2018 best paper, 14-19% error reduction on CIFAR-10 ResNets), the overhead is minimal (~32K params, <1ms per step), and the mechanism is clear — recalibrating channel responses improves feature quality without requiring more epochs. Every prior improvement targeted capacity (width), regularization (augmentation, WD), or throughput (AMP, batch size). SE targets expressiveness within the existing capacity — a new axis with strong compounding potential.

**Hypothesis**:
Adding SE blocks with reduction ratio r=16 to each BasicBlock in the WIDTH_MULT=4 ResNet-20 will improve best_test_acc from 95.39% to ~95.6-95.9% by enabling the model to learn channel-wise feature importance. The per-step overhead will be negligible (<1ms), preserving the ~98 epoch count, while the improved feature calibration delivers accuracy gains similar to the ~0.2-0.5pp improvements seen in the SE-ResNet literature on CIFAR-10.
