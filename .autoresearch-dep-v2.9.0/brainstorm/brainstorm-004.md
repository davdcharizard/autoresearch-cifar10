# Brainstorm EXP-004
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new sources — the existing literature corpus (WRN paper, TrivialAugment, He-2015) is sufficient. The key new observation from the experimental trajectory is that the model is still converging at epoch 69 (EXP-003 climbed from 92.99% to 93.33% in the final 5 epochs), which points to throughput as the next binding constraint.

## Experimental History Review

Current baseline: **93.33%** (EXP-003, width-2x + TrivialAugmentWide + RandomErasing + WD=5e-4, 69 epochs, commit a3e19f8).

Trajectory (all improvements):
- BASE 91.72% → EXP-001 92.29% (+0.57, capacity) → EXP-002 92.92% (+0.63, augmentation) → EXP-003 93.33% (+0.41, WD)
- Total: +1.61pp. Now exceeds WRN-16-2 anchor (93.2%).

Key insight from EXP-003: the model was still climbing at epoch 69 (93.33% at final epoch, up from 92.99% five epochs earlier). This means the 300s budget is the binding constraint — more epochs would yield more accuracy. The escalating second-LR-drop contribution (EXP-001: +0.02pp, EXP-002: +0.52pp, EXP-003: +1.8pp) confirms that the regularized model has a large optimization gap the low-LR phase is closing.

Remaining axes:
- **Throughput** (AMP, channels_last, larger batch): more epochs in same budget
- **Nesterov**: free +0.1-0.3pp
- **Label smoothing**: regularization axis
- **Width-4x**: higher capacity, but fewer epochs

## Candidate Ideas

### 1. Nesterov momentum + label smoothing 0.1 (recipe polish)

**Summary**: Add `nesterov=True` to the SGD optimizer and `label_smoothing=0.1` to `F.cross_entropy()`. Two keyword arguments. Everything else unchanged.

**Reasoning**: Both are standard recipe components that the WRN paper and modern CIFAR-10 training recipes use. Nesterov improves gradient estimation (+0.1-0.3pp typical). Label smoothing prevents overconfidence and adds regularization (+0.1-0.3pp). Together they represent "recipe polish" — small orthogonal gains that compound. Bundling these two is justified because both are well-understood single-keyword changes with no interaction risk, unlike EXP-000's problematic 5-change bundle.

**Sources**: WRN paper (Nesterov used), Inception-v3 paper (label smoothing), reports/exp-report-003.md § Next Steps.

**Estimated Effort**: Very low — two keyword arguments.

**Risk Assessment**: Very low. Both are standard improvements with no failure modes. The label smoothing changes the loss landscape slightly (training loss will read higher) but this is cosmetic. Worst case: no-improvement if the gains are too small to clear the +0.1pp threshold on this particular seed.

### 2. AMP (torch.cuda.amp) for throughput

**Summary**: Wrap forward/backward in `torch.amp.autocast('cuda', dtype=torch.float16)`, add `torch.amp.GradScaler()`, set `model.to(memory_format=torch.channels_last)`. Goal: reduce per-step time from ~11ms, fitting more epochs in 300s. The model was still improving at epoch 69 — more epochs means more accuracy.

**Reasoning**: EXP-003 showed the model gaining +0.34pp in its final 5 epochs alone. If AMP delivers 1.5-2x throughput, the epoch count rises from 69 to 100+, adding ~30 extra epochs in the 0.001 LR polish phase. This targets the binding constraint directly (the 300s budget limits convergence).

**Sources**: PyTorch AMP docs, goal-learnings § Patterns about model still converging at budget end.

**Estimated Effort**: Medium — ~15-20 lines changed (autocast context, GradScaler, channels_last).

**Risk Assessment**: Medium. AMP can introduce NaN gradients from FP16 underflow (GradScaler handles this but adds complexity). channels_last has edge cases. The accuracy effect of FP16 vs FP32 training is uncertain — it could be slightly worse per-step but better overall due to more steps.

### 3. Batch size 256 + LR 0.2 (linear scaling)

**Summary**: Double batch size to 256 and scale initial LR proportionally to 0.2 (linear scaling rule). The wall-clock-fractional schedule thresholds (0.5, 0.75) remain unchanged. Each step processes twice as many images, effectively doubling throughput at the cost of fewer gradient updates per epoch.

**Reasoning**: Larger batch sizes increase GPU utilization and reduce per-image overhead. On the H20 with the width-2x model at 598 MB VRAM, batch 256 should fit comfortably. The linear scaling rule (LR ∝ batch size) is well-established for SGD-momentum on CIFAR-10 up to batch sizes of 512-1024.

**Sources**: Linear scaling rule (Goyal et al. 2017), goal-learnings about H20 VRAM headroom.

**Estimated Effort**: Low — two constants changed (BATCH_SIZE, LR).

**Risk Assessment**: Low-medium. Larger batch may reduce generalization quality (the "generalization gap" between large and small batch training). The epoch structure changes (195 steps/epoch vs 390), which is fine for the wall-clock-fractional schedule. The LR scaling may need adjustment — 0.2 is the linear rule but could be slightly too high with strong regularization.

## Idea Evaluation

**Evidence strength**: Candidate 1 has the most direct evidence — Nesterov and label smoothing are universal improvements with no known failure modes. Candidate 2 has strong throughput evidence but uncertain accuracy impact from FP16. Candidate 3 has mixed evidence — the linear scaling rule works but the generalization gap is a real concern at this accuracy level.

**Expected impact**: Candidate 1 targets +0.2-0.5pp through recipe polish. Candidate 2 could deliver more (the model was gaining 0.07pp/epoch in the final epochs, so 30 more epochs could add +2pp), but the FP16 accuracy trade-off is uncertain. Candidate 3 is similar to Candidate 2 in mechanism but with a different trade-off profile.

**Risk**: Candidate 1 is nearly risk-free. Candidates 2 and 3 both change training dynamics.

**Strategy**: Candidate 1 is the "take the sure win" choice — low-hanging recipe polish that locks in another increment before pursuing riskier throughput changes. After EXP-004, the throughput experiments (Candidates 2 and 3) become the natural next steps with a stronger baseline to work from.

## Chosen Idea

**Selected**: Candidate 1 — **Nesterov momentum + label smoothing 0.1**

**Why this idea**: Both are nearly risk-free standard recipe improvements that the WRN paper and modern CIFAR-10 recipes use. They represent the last low-hanging fruit before moving to riskier throughput or architectural changes. Bundling them is justified by their orthogonality and well-understood interaction (both are regularization-adjacent but target different mechanisms — gradient estimation vs confidence calibration).

**Hypothesis**: Adding `nesterov=True` to SGD and `label_smoothing=0.1` to cross_entropy — while keeping all other settings at EXP-003 values — will raise best_test_acc from 93.33% to **93.5-93.8%**, a +0.2-0.5pp gain. The improvement bar is 93.43% (baseline + 0.1pp).
