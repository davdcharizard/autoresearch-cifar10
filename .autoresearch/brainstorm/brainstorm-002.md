# Brainstorm EXP-002
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **CutMix paper (Yun et al. 2019)** (https://arxiv.org/pdf/1905.04899)
  CutMix outperforms CutOut by +0.97% on CIFAR-10. Works by cutting a patch from one image and pasting onto another, mixing labels proportionally. Best applied at image level, not feature maps. Stronger regularization since masked region gets useful training signal instead of zeros.

- **WideResNet paper (Zagoruyko & Komodakis 2016)** — already in knowledge from EXP-001 brainstorm. Width more efficient than depth. WRN-28-10 achieves 96.11% on CIFAR-10 at ~36.5M params.

## Experimental History Review

**Current state** (2 experiments on this goal):
- BASE: 91.81% (ResNet-20 original)
- EXP-000: 92.10% (+0.29%), cosine LR + CutOut + label smoothing (recipe only)
- EXP-001: 94.03% (+1.93%), k=2 width + AMP + torch.compile + projection shortcuts

**Key learnings**:
- Width is the primary accuracy lever (k=2 gave +1.93% vs +0.29% recipe-only)
- T_max mismatch causes large best/final gap (94.03% best vs 91.93% final with T_max=55, actual 78 epochs)
- AMP+compile give ~2.5x speedup, better than estimated
- Peak VRAM only 325MB at k=2 — massive headroom on H20 (98GB)

**Critical issue**: T_max=55 was set for estimated 55 epochs but actual was 78. The cosine schedule completed at epoch ~60, LR stayed at minimum for ~18 epochs → model peaked then degraded. Fixing this alone should significantly improve accuracy.

**Untried**: k=3 or k=4 width, CutMix, adaptive/dynamic T_max, larger batch size, deeper model

## Candidate Ideas

### 1. k=3 Width + Dynamic T_max Calibration + CutMix

**Summary**: Increase width to k=3 ({48, 96, 192}, ~2.4M params) and permanently solve the T_max problem by measuring per-epoch timing after epoch 1, then computing T_max dynamically. Replace CutOut with CutMix for stronger regularization on the larger model. Keep AMP, torch.compile, Nesterov, projection shortcuts, label smoothing.

Dynamic T_max: after completing epoch 1, measure wall-clock seconds per epoch, then estimate total_epochs = TIME_BUDGET_S / seconds_per_epoch. Set T_max = total_epochs - WARMUP_EPOCHS. This eliminates T_max guessing for all future width changes.

**Reasoning**: k=3 gives 2.25x more capacity than k=2. EXP-001 showed width is the primary lever. At k=2 we got 78 epochs; at k=3 expect ~35-40 epochs. Dynamic T_max ensures the cosine schedule matches perfectly. CutMix replaces CutOut for +0.97% (per CutMix paper) and provides stronger regularization needed for the larger model. Multiple improvements compound.

**Sources**: WideResNet paper, EXP-001 analysis (width lever + T_max issue), CutMix paper

**Estimated Effort**: medium

**Risk Assessment**: ~35-40 epochs may be borderline for convergence with k=3. CutMix implementation requires label mixing which adds complexity. Multiple simultaneous changes make attribution harder. But T_max calibration reduces the biggest risk from prior experiments.

### 2. k=2 + Fixed T_max=80 + CutMix

**Summary**: Keep the proven k=2 architecture, fix T_max to 80 (matching actual ~78 epochs), replace CutOut with CutMix. Minimal architecture change — just fix the known T_max issue and upgrade augmentation.

**Reasoning**: This is the safest path to recover the known 2.1% best/final gap from EXP-001. With proper T_max, the model should converge to a higher best_test_acc AND maintain it through the end of training. CutMix adds ~1% from literature. Combined: potentially 95-95.5%.

**Sources**: EXP-001 analysis (T_max mismatch), CutMix paper (+0.97%)

**Estimated Effort**: low

**Risk Assessment**: Very low risk. k=2 architecture is proven. T_max=80 is well-calibrated from EXP-001's actual 78 epochs. CutMix is the only new variable. Worst case: marginal improvement.

### 3. k=4 Width + Dynamic T_max + Batch=256

**Summary**: Aggressive width increase to k=4 ({64, 128, 256}, ~4.3M params). Increase batch size to 256 to improve GPU utilization and throughput. Dynamic T_max calibration. Keep CutOut (don't change augmentation with so many other changes).

**Reasoning**: k=4 is a 16x capacity increase over the baseline and 4x over k=2. At batch=256 with AMP+compile, might achieve ~20-25 epochs. The WideResNet paper shows massive capacity can compensate for fewer epochs, especially with proper LR scheduling.

**Sources**: WideResNet paper, EXP-001 VRAM observation (325MB → room for 4x)

**Estimated Effort**: medium

**Risk Assessment**: High risk. ~20-25 epochs is very aggressive for convergence. Even with perfect T_max, the model may not have enough training iterations to learn well. Larger batch with linear LR scaling could destabilize training. Multiple big changes simultaneously.

## Idea Evaluation

**Evidence strength**: Idea 2 has the strongest evidence — it purely fixes a known issue (T_max) on a proven architecture and adds CutMix (literature-backed). Idea 1 combines two evidence-backed changes (width + T_max fix) with CutMix. Idea 3 is the most speculative — no evidence that k=4 with ~20 epochs can converge.

**Mechanism clarity**: Idea 2 is clearest — fix T_max so cosine properly decays, replace zeros-augmentation with signal-preserving augmentation. Idea 1 adds the width mechanism (more features per layer). Idea 3's mechanism is unclear given the epoch constraint.

**Expected impact**: Idea 1 has the highest ceiling — more capacity + T_max fix + CutMix could reach 95%+. Idea 2 targets ~95% from T_max fix + CutMix on k=2. Idea 3 could theoretically reach higher but the convergence risk undermines the estimate.

**Strategy**: Idea 1 strikes the best balance — it increases capacity (k=3 is a moderate step up from k=2, not as risky as k=4), permanently solves the T_max issue with dynamic calibration, and upgrades augmentation. The dynamic T_max is especially valuable as a reusable infrastructure improvement.

## Chosen Idea

**Selected**: k=3 Width + Dynamic T_max Calibration + CutMix

**Why this idea**:
It combines the three most impactful improvements: more capacity (k=3, proven direction), dynamic T_max calibration (permanently fixes the recurring T_max issue), and CutMix (literature-backed +0.97% over CutOut). The dynamic T_max approach is particularly valuable because it eliminates guessing for all future width experiments.

**Hypothesis**:
Widening to k=3 ({48,96,192}, ~2.4M params) with dynamic T_max calibration and CutMix will improve best_test_acc from 94.03% to approximately 95-96%, driven by 2.25x more capacity with proper LR scheduling and stronger regularization.
