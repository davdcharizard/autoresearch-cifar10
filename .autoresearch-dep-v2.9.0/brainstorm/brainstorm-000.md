# Brainstorm EXP-000
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **"94% on CIFAR-10 in 3.29 Seconds on a Single GPU"** (https://arxiv.org/html/2404.00498v2)
  Key techniques for fast CIFAR-10 training: triangular LR schedule, Cutout augmentation (12px for 96% accuracy), patch-whitening initialization, GELU activations. 95% in 10.4s, 96% in 46.3s on A100. Cutout is the single biggest contributor when pushing from 95% to 96%.

- **CIFAR-10 training tricks notebook** (https://juliusruseckas.github.io/ml/cifar10-tricks.html)
  Systematic evaluation of individual tricks: label smoothing (eps=0.1) + Cutout together are more effective than either alone. Mixup (alpha=0.4) works well but needs longer training. OneCycleLR with AdamW reached 85.8% on ResNet18 in only 12 epochs (undertrained).

- **WRN-28-10 with AutoAugment + Cutout** (https://davidstutz.de/2-percent-test-error-on-cifar10-using-pytorch-autoagument/)
  WideResNet-28-10 achieves 97.44% accuracy with AutoAugment + Cutout + SGD with Nesterov + weight decay 5e-4 + cosine schedule. ResNet-50 achieves 96.87% with same recipe. AutoAugment is built into torchvision as `transforms.AutoAugment`.

- **ResNet-ELU-Cutout** (https://github.com/Amihaeseisergiu/Cifar-10-ResNet-ELU-Cutout)
  Small ResNet (663K params) achieves 95% with Cutout + ELU activations. Shows that moderate model size + good augmentation is very effective.

## Experimental History Review

First experiment under this goal — no prior history. Baseline: 91.72% (ResNet-20, 272K params, 97 epochs in 300s).

Key baseline analysis:
- LR schedule is mismatched: MultiStepLR milestones at 32K/48K steps, but only 37,796 steps complete in 300s. Second LR drop never fires. Model spends most of training at LR=0.1, drops to 0.01 only for final ~5,800 steps.
- VRAM headroom is massive: 330MB used of 98GB available. Model can be significantly larger.
- No advanced augmentation or regularization techniques are used.
- 97 epochs in 300s means ~3.1s/epoch — fast iteration allows many epochs even with larger models.

## Candidate Ideas

### 1. Training Recipe Modernization (Keep ResNet-20)
**Summary**: Keep the ResNet-20 architecture unchanged but overhaul the training recipe: replace MultiStepLR with cosine annealing (properly fit to actual training duration), add Cutout augmentation (16x16 patches), add label smoothing (eps=0.1), and enable Nesterov momentum. This isolates training improvements from architectural changes.

**Reasoning**: The current LR schedule is fundamentally mismatched — it was designed for 64K iterations but training is time-bounded. Cosine annealing naturally adapts to whatever number of steps are completed. Cutout and label smoothing are individually proven to add 0.5-1% each on CIFAR-10, and the cifar10-tricks source shows they compound when combined. These are all well-understood, low-risk changes.

**Sources**: cifar10-tricks notebook (label smoothing + Cutout synergy), fast CIFAR-10 paper (Cutout as key contributor to 96%), WRN AutoAugment recipe (Nesterov + cosine)

**Estimated Effort**: low — all changes are hyperparameter/augmentation only, no architecture changes

**Risk Assessment**: Very low risk. Each technique is well-established. Worst case is minimal improvement if ResNet-20 is too small to benefit from better regularization. The LR schedule fix alone should help.

### 2. WideResNet Architecture + Cosine Schedule
**Summary**: Replace ResNet-20 with a WideResNet architecture (e.g., WRN-16-4 or WRN-28-2) that has significantly more capacity, paired with cosine annealing LR. Keep augmentation minimal (existing crop + flip). WideResNet is proven on CIFAR-10 — wider layers learn richer features per block compared to deeper-but-narrow alternatives.

**Reasoning**: The baseline uses only 330MB of 98GB VRAM, and the 300s budget allows ~97 epochs with a tiny model. A wider model (e.g., WRN-16-4 with ~2.7M params or WRN-28-2 with ~1.5M params) will still train many epochs in 300s but with far more representational capacity. WideResNets are the dominant architecture family for CIFAR-10 accuracy at all compute levels. Cosine annealing is included as the schedule fix is nearly free and clearly needed.

**Sources**: WRN paper (Zagoruyko & Komodakis, 2016), WRN-28-10 AutoAugment recipe (97.44% accuracy), fast CIFAR-10 paper (wider channels key to 96%)

**Estimated Effort**: medium — requires writing WideResNet architecture (BasicBlock with wider channels, dropout between convolutions, proper shortcut projections)

**Risk Assessment**: Medium risk. The architecture change is more complex and could introduce bugs. Need to ensure the model fits in 300s training budget — if too large, fewer epochs complete and accuracy could drop. WRN-16-4 (~2.7M params) is a safe choice that should still train fast.

### 3. Full Modern Stack (Architecture + Augmentation + Regularization)
**Summary**: Combine a wider architecture (WRN-16-4) with the full modern training recipe: cosine annealing, Cutout (16x16), label smoothing (eps=0.1), Nesterov SGD, and potentially Mixup (alpha=0.2). This is the maximum-impact approach, targeting 95%+ accuracy.

**Reasoning**: The research consistently shows that the biggest gains on CIFAR-10 come from combining architectural capacity with strong regularization/augmentation. WRN + AutoAugment + Cutout achieves 97%+. Even without AutoAugment (which may be slow to apply), WRN + Cutout + label smoothing should push well past 94%. The fast CIFAR-10 paper shows 96% with Cutout in only 46s on A100 — we have 300s on H20 which is ample.

**Sources**: All sources above — this synthesizes the best practices from each

**Estimated Effort**: high — architecture rewrite + multiple augmentation/regularization additions + hyperparameter tuning needed

**Risk Assessment**: Higher risk due to many simultaneous changes. If accuracy doesn't improve, hard to attribute which change helped or hurt. Mixup adds training complexity. The model might be too large for 300s if poorly sized. However, the individual components are all proven — the risk is in integration, not in the techniques themselves.

## Idea Evaluation

**Evidence strength**: All three ideas are backed by strong evidence. Candidate 1 is safest — each technique is individually validated. Candidate 2 has strong architectural evidence (WRN dominates CIFAR-10 leaderboards). Candidate 3 combines both but introduces attribution risk.

**Mechanism clarity**: Candidate 1 has the clearest mechanism — the LR schedule is demonstrably mismatched and Cutout/label smoothing have well-understood regularization effects. Candidate 2's mechanism is also clear — wider layers learn richer per-block representations. Candidate 3 stacks mechanisms but makes it harder to isolate what works.

**Expected impact**: Candidate 1 likely reaches ~93-94%. Candidate 2 likely reaches ~94-95%. Candidate 3 could reach ~95-96% but with more uncertainty. For a first experiment, establishing a strong improved baseline matters more than maximizing a single result.

**Risk profile**: Candidate 1 has the safest failure mode — worst case is ~92% (minimal improvement). Candidate 2 could fail if the architecture has a bug or is too slow. Candidate 3 has the most failure modes.

**Feasibility**: Candidate 1 is a few lines of code. Candidate 2 requires writing a new architecture (~80 lines). Candidate 3 requires the most changes.

**Recommendation**: Candidate 1 (Training Recipe Modernization) is the best first experiment. It has the highest evidence-to-risk ratio, isolates training recipe improvements from architecture changes, and will establish how much headroom remains for architectural changes in future experiments. If it lands at ~93-94%, the next experiment can layer on architecture changes with confidence.

## Chosen Idea
**Selected**: Training Recipe Modernization (Keep ResNet-20)

**Why this idea**:
The LR schedule mismatch is the single most impactful bug in the current setup — the model never reaches the second LR decay. Fixing this alone should yield measurable improvement. Layering Cutout and label smoothing on top is well-supported by research showing they compound. As the first experiment, isolating training recipe changes from architecture changes gives the clearest signal about where improvements come from, informing future experiments.

**Hypothesis**:
Replacing MultiStepLR with cosine annealing (fit to actual training duration), adding Cutout (16x16), label smoothing (eps=0.1), and Nesterov momentum will improve best_test_acc from 91.72% to approximately 93-94%, primarily driven by the LR schedule fix allowing more effective use of the 300s training budget.
