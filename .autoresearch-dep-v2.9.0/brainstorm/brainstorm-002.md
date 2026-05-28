# Brainstorm EXP-002
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

Carried over from brainstorm-000 and brainstorm-001 where these sources were fully cited. No new papers warranted — the augmentation literature is well-established.

- **TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation — Müller & Hutter, ICCV 2021** (https://arxiv.org/abs/2103.10158)
  Matches or exceeds AutoAugment/RandAugment on CIFAR-10 with zero hyperparameter search. On WRN-28-10 the paper reports +0.5-1.0pp over no-aug baseline. Available in torchvision since 0.13 (project pins 0.24.1). Recent 2024-2025 work continues to use TrivialAugment as the default augmentation for WRN-class CIFAR-10 models.

- **Random Erasing Data Augmentation — Zhong et al. 2020** (https://arxiv.org/abs/1708.04896)
  The torchvision `transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))` is the standard post-normalize Cutout-equivalent. On CIFAR-10 ResNets the reported gain is +0.5-0.9pp. Operates on tensors (post-ToTensor/Normalize).

- **Wide Residual Networks — Zagoruyko & Komodakis 2016** (https://arxiv.org/abs/1605.07146)
  Table 5: WRN-16-2 ≈ 93.2%, WRN-28-2 ≈ 93.8% on CIFAR-10 under SGD-momentum-WD-step-decay with standard augmentation (crop+flip). The paper uses weight_decay=5e-4 (vs our 1e-4).

## Experimental History Review

Current baseline: **92.29%** (EXP-001, width-2x ResNet-20, 1.07M params, 69 epochs in 300s, wall-clock-fractional schedule, commit dc1538a).

- **BASE** (91.72%): Unmodified ResNet-20, 270K params, MultiStepLR, 97 epochs.
- **EXP-000** (88.79%, no-improvement): Bundled 5 changes including CosineAnnealingLR(T_max=200), Cutout, label smoothing, Nesterov. Failed because cosine T_max=200 with only 91 actual epochs never reached low-LR regime.
- **EXP-001** (92.29%, improvement, +0.57pp): Width-2x with wall-clock-fractional schedule. First successful experiment. Gap of ~1.2pp to WRN-paper anchor (~93.5% for n=3,k=2 at 200 epochs).

Key patterns from goal-learnings:
- Wall-clock-fractional schedule (drops at 0.5/0.75 of TIME_BUDGET_S) is the validated schedule for any architecture variant (High Importance)
- Width-2x gets 69 epochs at ~11ms/step (Medium Importance)
- First LR drop delivers most of the accuracy gain; second drop marginal (Medium Importance)

Untouched axes on the current baseline:
- **Augmentation**: Only crop+flip. No TrivialAugment, no RandomErasing, no Cutout.
- **Weight decay**: 1e-4 (vs WRN paper's 5e-4 for wider models)
- **Throughput**: No AMP, no channels_last, batch size unchanged at 128
- **Optimizer**: No Nesterov, no gradient clipping
- **Architecture**: Width-4x not yet tried

## Candidate Ideas

### 1. TrivialAugmentWide + RandomErasing on the width-2x baseline

**Summary**: Add `transforms.TrivialAugmentWide()` before `ToTensor()` and `transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))` after `Normalize()` in the training transform pipeline. Everything else — architecture (WIDTH_MULT=2), wall-clock-fractional schedule, optimizer (SGD lr=0.1, momentum=0.9, WD=1e-4), batch size 128 — stays exactly at EXP-001 values. Two lines added to `train_tf`.

**Reasoning**: The standard "capacity first, then regularization" sweep order in the CIFAR-10 literature (WRN paper → AutoAugment paper → RandAugment paper) establishes that augmentation gains compound with model width. The 1.07M-param width-2x model has more representational capacity than the 270K baseline, so the generalization gap that augmentation closes is larger in absolute terms. TrivialAugment is the zero-search-cost choice that matches or beats AutoAugment on CIFAR-10 WRN models. RandomErasing is the standard complement (Cutout-equivalent, tensor-space). The EXP-000 loop included Cutout but bundled it with 4 other changes, so the augmentation axis has never been cleanly isolated.

**Sources**: TrivialAugment paper (https://arxiv.org/abs/2103.10158, Table 1), Random Erasing paper (https://arxiv.org/abs/1708.04896), brainstorm-001 § Candidate 3 (the augmentation-only idea for the narrow baseline, now applied to the wider model), reports/exp-report-001.md § Next Steps #1.

**Estimated Effort**: Very low — two lines added to the transforms.Compose list in train.py.

**Risk Assessment**: Low variance. The schedule and architecture are unchanged, so the EXP-000 failure mode is impossible. The main risk is that augmentation overhead reduces epoch count from 69 to ~60-63 (the EXP-000 run lost ~6 epochs to Cutout-style augmentation on the narrow model; the wider model's step time is already longer so the augmentation overhead is a smaller fraction of per-step cost). Under the wall-clock-fractional schedule, fewer epochs don't disrupt the LR drop timing. Worst case is a flat outcome (~92.2-92.3%) if the augmentation's regularization effect doesn't outweigh the epoch-count reduction on the wider model.

### 2. Weight decay 5e-4 on the width-2x baseline

**Summary**: Change `WEIGHT_DECAY = 1e-4` to `WEIGHT_DECAY = 5e-4` in the hyperparameters block. A single constant change. Everything else unchanged.

**Reasoning**: The WRN paper uses weight_decay=5e-4 for all width-multiplied models (Tables 4-5), vs the original ResNet paper's 1e-4 that our baseline inherits. Wider models have more parameters and higher effective model capacity, so stronger L2 regularization is standard. The 5x increase is the exact ratio the WRN paper uses and is well-supported across the CIFAR-10 literature for models in the 1-10M parameter range.

**Sources**: WRN paper (https://arxiv.org/abs/1605.07146, experimental setup section), goal-learnings patterns about width-2x model characteristics.

**Estimated Effort**: Very low — single constant change.

**Risk Assessment**: Low. Weight decay is a well-understood regularizer. 5e-4 is the literature standard for WRN-class models. The failure mode is a slight regression if the 69-epoch budget doesn't give enough training iterations for the stronger regularization to help (WD-induced underfitting is a risk at very high WD values, but 5e-4 is well within the standard range). Expected gain is modest (+0.2-0.5pp) based on the WRN paper's ablations.

### 3. AMP (torch.cuda.amp) + channels_last for throughput on the width-2x baseline

**Summary**: Wrap the forward-backward pass in `torch.amp.autocast('cuda', dtype=torch.float16)` and use `torch.amp.GradScaler()` for loss scaling. Set `model = model.to(memory_format=torch.channels_last)` for NHWC layout. These are PyTorch-native features requiring no new dependencies. The goal is to reduce per-step time from ~11ms, fitting more epochs in the 300s budget and improving convergence quality.

**Reasoning**: EXP-001's gap to the WRN literature anchor (~1.2pp) is partly attributable to fewer epochs (69 vs 200). AMP on H20 GPUs (which have FP16 Tensor Cores) typically delivers 1.5-2x throughput on conv-heavy models. If per-step time drops from 11ms to ~7ms, the epoch count would rise from 69 to ~100+, approaching the baseline's epoch count and giving the schedule more training iterations to work with. Channels_last memory format is the standard companion optimization for conv-heavy models on NVIDIA GPUs.

**Sources**: PyTorch AMP documentation, goal-learnings § Patterns about width-2x epoch count, reports/exp-report-001.md § Unexplored Avenues.

**Estimated Effort**: Medium — requires wrapping the training loop with autocast context manager, adding GradScaler, modifying the forward/backward flow, and channels_last conversion. ~15-20 lines changed.

**Risk Assessment**: Medium. AMP can introduce numerical instability on some architectures (NaN gradients from FP16 underflow/overflow), though GradScaler is designed to handle this. The channels_last conversion has edge cases with some PyTorch operations. The wall-clock-fractional schedule should handle the changed step rate correctly (that's its design purpose). However, this changes the training dynamics (FP16 vs FP32 gradient accumulation) which could affect the final accuracy in either direction. If AMP introduces NaN losses, the experiment fails cleanly (abort criterion catches it).

## Idea Evaluation

**Evidence strength**: Candidate 1 (augmentation) has the strongest directly-applicable evidence — the TrivialAugment paper reports gains on WRN-class CIFAR-10 models under the same SGD-momentum-WD recipe family. Candidate 2 (WD) has solid evidence from the WRN paper's experimental setup but limited ablation data on WD alone. Candidate 3 (AMP) has general throughput evidence but no specific CIFAR-10 accuracy evidence — the benefit is indirect (more epochs → better convergence).

**Mechanism clarity**: Candidate 1 has the clearest mechanism — augmentation increases the effective training set diversity, closing the generalization gap that the wider model exposes. Candidate 2's mechanism is also clear (stronger L2 shrinkage prevents overfitting in the wider model). Candidate 3's mechanism is indirect — throughput improvement → more epochs → better convergence quality, but the accuracy effect depends on how much the extra epochs contribute at the schedule's fractional-drop positions.

**Expected impact**: Candidate 1's expected band is +0.5-1.5pp (landing at 92.8-93.8%), which would close a significant portion of the gap to the WRN literature anchor. Candidate 2's expected band is +0.2-0.5pp (landing at 92.5-92.8%) — meaningful but smaller. Candidate 3's impact is harder to predict — the throughput gain is likely 1.5-2x, but the accuracy effect of extra epochs is nonlinear and depends on where the additional epochs fall relative to the schedule's LR plateaus.

**Risk profile**: Candidates 1 and 2 are both low-risk with the schedule unchanged. Candidate 3 is medium-risk due to potential FP16 numerical issues and the multi-site code change surface.

**Single-axis attribution**: Candidate 1 isolates the augmentation axis cleanly. Candidate 2 isolates the WD axis. Candidate 3 changes the numerical precision and memory layout — a less clean axis for attribution if the accuracy changes.

**Selection**: Candidate 1 is the strongest choice — highest evidence, clearest mechanism, largest expected impact, lowest risk, and the natural next step in the capacity-first-then-regularization sweep order. The wider model has the most headroom to benefit from augmentation, and the clean two-line change makes the attribution unambiguous. If augmentation delivers, the WD sweep (Candidate 2) is the natural EXP-003 follow-up to compound with it.

## Chosen Idea

**Selected**: Candidate 1 — **TrivialAugmentWide + RandomErasing on the width-2x baseline**

**Why this idea**: It is the highest-evidence single-axis regularization change for the width-2x model. The TrivialAugment paper demonstrates gains on WRN-class CIFAR-10 models under the same training recipe family, and the "capacity first, then regularization" sweep order in the CIFAR-10 literature establishes that augmentation gains compound with model width. The two-line change is the cleanest possible augmentation intervention, the schedule and architecture are preserved identically from EXP-001, and the attribution of any metric delta to the augmentation axis is unambiguous.

**Hypothesis**: Adding TrivialAugmentWide before ToTensor and RandomErasing(p=0.25, scale=(0.02, 0.2)) after Normalize to the width-2x ResNet-20's training transforms — while keeping every other setting at EXP-001 values — will raise best_test_acc from 92.29% to the range **92.8-93.5%**, a +0.5-1.2pp gain. The mechanism is the increased effective training-set diversity closing the generalization gap that the 1.07M-param model exposes. The improvement bar of 92.39% (baseline + 0.1pp) is exceeded by the expected band's lower edge. The augmentation overhead will reduce epoch count from 69 to ~60-63, but the wall-clock-fractional schedule absorbs this without disrupting LR drop timing.
