# Brainstorm EXP-010
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (Yun et al., 2019)** (https://arxiv.org/abs/1905.04899)
  CutMix replaces a rectangular patch of one image with a patch from another and mixes labels proportionally to the area ratio. On CIFAR-10 with ResNet-56, it reports +0.97% over baseline (96.68% vs 95.71%). Unlike Cutout, which zeros pixels (losing information), CutMix fills the removed region with useful training signal from another sample. CutMix is a batch-level operation with negligible throughput cost — it operates on already-loaded tensors. The `torchvision.transforms.v2.CutMix` API is available but operates as a batch transform (applied after DataLoader), not a per-sample transform.

- **Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates (Smith & Topin, 2018)** (https://arxiv.org/abs/1708.07120)
  The 1cycle policy achieves super-convergence: a large peak LR acts as regularization, requiring reduction of other regularization forms (WD, dropout). On CIFAR-10 with ResNet-56, achieved 92.4% after only 10K iterations vs 91.2% at 80K with piecewise-constant schedule — 8x fewer iterations for +1.2%. PyTorch provides `torch.optim.lr_scheduler.OneCycleLR` with built-in warmup and cooldown phases. Key finding: large LR and WD are substitutes, not complements — when using 1cycle with large peak LR, WD should be reduced.

- **Cosine annealing vs step decay on CIFAR-10** (web search synthesis)
  Multiple sources (SGDR paper, practitioner benchmarks) indicate cosine annealing and step decay perform comparably on CIFAR-10 when both are well-tuned. The difference is typically <0.2pp. The main advantage of cosine is simplicity (one hyperparameter T_max vs milestone positions) and smoother decay. EXP-000 failed cosine with T_max=200 >> actual epochs — the idea was never properly tested with correct T_max.

## Experimental History Review

- **Current best**: 95.39% (EXP-009, batch-256 with linear LR scaling + 5-epoch warmup)
- **Verification threshold**: best_test_acc > 95.49% (baseline + 0.1pp)
- **Trajectory**: 7 improvements in 10 experiments, +3.67pp total from 91.72% baseline. Major drivers: width scaling (+0.57pp, +1.53pp), augmentation (+0.63pp), WD (+0.41pp), AMP throughput (+1.11pp), batch scaling (+0.57pp).
- **What worked**: Capacity increases (width-2x, width-4x), throughput increases (AMP, batch-256), regularization additions (TrivialAugmentWide+RandomErasing, WD=5e-4). The wall-clock-fractional step-decay at (0.5, 0.75) is validated as near-optimal.
- **What failed**: CosineAnnealingLR with wrong T_max (EXP-000), Nesterov+label_smoothing (EXP-004, throughput cost), shifted LR schedule (EXP-006), torch.compile (EXP-008, zero speedup).
- **Key pattern**: Throughput-to-accuracy conversion is the primary improvement driver. Each additional epoch still contributes meaningfully at 98 epochs. Per-step time 16ms at batch-256.
- **Untried approaches**: CutMix/MixUp batch augmentation (CutMix was suggested but never tried), OneCycleLR/cosine schedule with correct T_max, batch size 512+, architectural changes (deeper model, squeeze-excite).
- **Diminishing returns signal**: Batch-512 expected to yield only ~10-15% more epochs due to sublinear throughput scaling. Further width increases would reduce epoch count. The next +0.1pp will be harder.

## Candidate Ideas

### 1. CutMix Batch Augmentation
**Summary**: Add CutMix as a batch-level augmentation applied after the DataLoader produces a batch, before the forward pass. CutMix selects a random rectangular region in each image and replaces it with the corresponding region from another image in the batch, mixing labels proportionally to the area ratio (λ drawn from Beta(α, α)). Use α=1.0 (uniform λ distribution, the standard setting from the paper). This is applied ON TOP of the existing per-sample augmentation pipeline (RandomCrop, RandomHorizontalFlip, TrivialAugmentWide, RandomErasing). Implementation: after loading inputs and targets to GPU, generate a random λ, compute bounding box, create shuffled indices, blend images and labels, then compute cross-entropy against the mixed labels.

**Reasoning**: CutMix provides a different augmentation axis than what's currently used. The existing pipeline operates spatially (crop, flip, erase) and per-pixel (TrivialAugment), but none of them mix information across samples. CutMix's cross-sample mixing creates harder training examples that improve generalization. The original paper reports +0.97% on CIFAR-10 (ResNet-56). With 98 epochs and a model that's still converging, the training budget is sufficient for CutMix's regularization to take effect without the over-regularization risk flagged in EXP-004's brainstorm (which was about label smoothing, a different mechanism). CutMix is a pure tensor operation on already-loaded batches — zero throughput cost.

**Sources**: Yun et al. 2019 (CutMix paper), EXP-009 report § Unexplored Avenues (CutMix flagged as next candidate), goal-learnings § Patterns (TrivialAugmentWide+RandomErasing is free lunch — CutMix is complementary)

**Estimated Effort**: low — ~20 lines of code added to the training loop (CutMix logic between DataLoader output and forward pass), no hyperparameter search beyond α=1.0

**Risk Assessment**: Over-regularization is the primary risk — stacking CutMix on top of TrivialAugmentWide+RandomErasing+WD=5e-4 could slow convergence enough that 98 epochs aren't sufficient to reach the accuracy ceiling. Mitigation: α=1.0 is the standard setting; if over-regularized, accuracy will plateau lower but won't crash. Mixed labels change the loss landscape, which could interact poorly with the step-decay schedule's sharp transitions. Worst case: no-improvement (accuracy matches or slightly trails baseline).

### 2. OneCycleLR with Reduced Weight Decay
**Summary**: Replace the wall-clock-fractional step-decay LR schedule with PyTorch's `OneCycleLR`. Configure with max_lr=0.3 (1.5x current peak), div_factor=25 (start LR = 0.012), final_div_factor=1e4 (end LR = 3e-5), pct_start=0.3 (30% warmup, 70% cosine decay), and total_steps estimated from 98 epochs × ~194 steps/epoch ≈ 19012 steps. Simultaneously reduce WD from 5e-4 to 1e-4 per Smith & Topin's finding that large LR and WD are substitutes. Remove the existing warmup mechanism (WARMUP_EPOCHS) since OneCycleLR has built-in warmup.

**Reasoning**: The current step-decay schedule wastes the first 50% of budget at a fixed high LR, then makes two sharp drops. OneCycleLR's continuous warmup+decay profile reaches higher peak LR (more regularization, wider exploration) and then smoothly decays, avoiding the sharp transitions that cause instability with AMP at LR=0.01 (flagged in EXP-005). Smith & Topin demonstrated that 1cycle achieves better accuracy in fewer iterations — our throughput-bound regime directly benefits. Reducing WD compensates for the additional regularization from the larger peak LR.

**Sources**: Smith & Topin 2018 (super-convergence), EXP-005 pattern (AMP unstable at LR=0.01 — OneCycleLR's smooth decay avoids sharp transitions), EXP-009 report § Unexplored Avenues (OneCycleLR flagged), goal-learnings § Patterns (first LR drop delivers majority of accuracy gain — suggests the schedule shape matters)

**Estimated Effort**: medium — replace the LR schedule function and warmup mechanism, adjust WD, need to estimate total_steps correctly for the 300s budget

**Risk Assessment**: The total_steps parameter must match actual training steps closely — if overestimated, the LR won't decay enough; if underestimated, training continues at minimum LR wasting the schedule. This is the same T_max problem that killed EXP-000. Mitigation: use a wall-clock-based approach to dynamically adjust, or estimate conservatively from EXP-009's 98 epochs. Reducing WD from 5e-4 to 1e-4 is a gamble — WD=5e-4 was validated as beneficial in EXP-003. The interaction between OneCycleLR's continuous schedule and the wall-clock-based training termination adds complexity. Worst case: accuracy regression from WD reduction or T_max mismatch.

### 3. Cosine Annealing with Correct T_max
**Summary**: Replace the wall-clock-fractional step-decay schedule with `CosineAnnealingLR(T_max=estimated_steps)` where estimated_steps ≈ 19012 (from EXP-009's 98 epochs × ~194 steps/epoch). Keep the 5-epoch linear warmup. Keep WD=5e-4 unchanged. This is the minimal-risk schedule change: swap only the decay shape (step → cosine) while preserving everything else.

**Reasoning**: EXP-000 failed cosine because T_max=200 >> actual epochs, keeping LR too high. With T_max correctly calibrated to ~19000 steps, cosine will reach near-zero LR by the end of training. The smooth decay avoids the sharp LR=0.01 instability seen with AMP in EXP-005. Literature suggests cosine and step decay perform within ~0.2pp on CIFAR-10, but the smooth profile may better suit the AMP training dynamics. This is the safest schedule experiment — one variable changed, well-understood mechanism.

**Sources**: EXP-000 failure (wrong T_max — the idea was invalidated by implementation, not by the technique), goal-learnings § Failed Approaches (CosineAnnealingLR with T_max >> actual epoch count), EXP-009 report § Unexplored Avenues (cosine annealing flagged)

**Estimated Effort**: low — replace the LambdaLR schedule function with CosineAnnealingLR, adjust T_max calculation

**Risk Assessment**: Literature suggests cosine vs step is typically marginal (<0.2pp). The verification threshold requires +0.1pp, so the expected effect size is close to the noise floor. The T_max estimation introduces the same risk as EXP-000 if miscalibrated — but we have strong data from EXP-009 (98 epochs, ~19000 steps) to calibrate accurately. The wall-clock termination means actual steps may vary ±5% from the estimate, but cosine is much more robust to this than the original T_max=200 failure. Worst case: marginal improvement that doesn't clear the +0.1pp threshold (no-improvement).

## Idea Evaluation

**Evidence strength**: CutMix has the strongest external evidence — +0.97% in the original paper on a comparable setup (CIFAR-10 ResNet). OneCycleLR has strong evidence for faster convergence but the evidence is on a different baseline (standard training without heavy augmentation). Cosine annealing has the weakest evidence — literature suggests marginal gains on CIFAR-10.

**Mechanism clarity**: CutMix's mechanism is clear — cross-sample mixing creates harder examples that improve generalization, orthogonal to existing augmentation (which is spatial/per-pixel, not cross-sample). OneCycleLR's mechanism is also clear — smooth warmup+decay with higher peak LR provides implicit regularization — but it requires simultaneously changing WD, introducing a confound. Cosine's mechanism is straightforward (smoother decay) but the causal path to accuracy improvement is weak given the validated step-decay schedule.

**Expected impact**: CutMix targets a genuinely untried augmentation axis with a reported +0.97% effect size, though stacking on existing augmentation will likely reduce the marginal gain. Realistically, +0.2-0.5pp seems plausible. OneCycleLR could yield +0.2-0.4pp from better LR profile but risks regression from WD reduction. Cosine is likely +0.0-0.2pp based on literature.

**Risk profile**: CutMix fails gracefully — worst case is no-improvement from over-regularization. OneCycleLR has the riskiest failure mode — wrong total_steps estimation could cause regression (echoing EXP-000), and reducing WD undoes a validated improvement. Cosine is low-risk but also low-reward, likely insufficient to clear the +0.1pp threshold.

**Feasibility**: CutMix is the most straightforward — a self-contained batch transform added to the training loop. No hyperparameter interactions to manage. OneCycleLR requires coordinating schedule replacement + WD reduction + warmup removal + total_steps estimation. Cosine is simple but shares the T_max estimation challenge.

**Verdict**: CutMix has the strongest evidence (original paper +0.97%), clearest orthogonal mechanism (cross-sample mixing vs existing spatial augmentation), safest failure mode (no-improvement), and lowest implementation complexity. OneCycleLR is promising but too many simultaneous changes (schedule + WD + warmup) for a single experiment. Cosine is too marginal to reliably clear the threshold.

## Chosen Idea
**Selected**: CutMix Batch Augmentation

**Why this idea**:
CutMix targets a genuinely untried augmentation axis (cross-sample mixing) with strong external evidence (+0.97% in the original paper). It is orthogonal to the existing augmentation pipeline (which operates per-sample spatially), has zero throughput cost (pure tensor ops on GPU), and fails gracefully (over-regularization → no-improvement, not crash). The implementation is self-contained (~20 lines) with no hyperparameter interactions to manage.

**Hypothesis**:
Adding CutMix with α=1.0 to the existing training pipeline will improve best_test_acc above 95.49% (baseline 95.39% + 0.1pp) by providing cross-sample regularization that improves generalization. The effect will be smaller than the original paper's +0.97% due to stacking on existing augmentation, but the 98-epoch budget is sufficient for the additional regularization to converge. Expected improvement: +0.2-0.5pp, targeting 95.6-95.9%.
