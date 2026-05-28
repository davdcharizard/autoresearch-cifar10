# Brainstorm EXP-005
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new sources. The key insight comes from the experimental trajectory: EXP-003 showed the model still climbing at epoch 69 (+0.34pp in final 5 epochs), and EXP-004 confirmed that the epoch count is the binding constraint (losing 4 epochs from per-step overhead wiped out the convergence-quality gain). The next move is throughput.

## Experimental History Review

Current baseline: **93.33%** (EXP-003, width-2x + aug + WD=5e-4, 69 epochs, commit a3e19f8).

5 experiments total, 3 improvements, 1 no-improvement (EXP-000 bundled recipe), 1 no-improvement (EXP-004 Nesterov+LS).

Key learning from EXP-004: in the 300s budget, per-step overhead matters — 4 fewer epochs cost more than the convergence-quality gain. **Prefer throughput-neutral or throughput-positive changes.**

Key learning from EXP-003: model was still climbing at epoch 69 (gained +0.34pp in final 5 epochs). More epochs = more accuracy.

The wall-clock-fractional schedule correctly adapts to any step rate (High Importance pattern).

## Candidate Ideas

### 1. AMP (torch.cuda.amp) with GradScaler

**Summary**: Wrap the forward-backward pass in `torch.amp.autocast('cuda', dtype=torch.float16)` and use `torch.amp.GradScaler()` for loss scaling. Apply `model = model.to(memory_format=torch.channels_last)` and convert input tensors to channels_last. The goal is to reduce per-step time, fitting more epochs in 300s.

**Reasoning**: The H20 has FP16 Tensor Cores that deliver 1.5-2x throughput on conv-heavy models. If per-step time drops from ~11ms to ~7ms, epoch count rises from 69 to ~100+, adding ~30 extra epochs. EXP-003 showed the model gaining 0.07pp/epoch in the final epochs — 30 more epochs could add ~1-2pp. The wall-clock-fractional schedule handles the changed step rate by design.

**Sources**: PyTorch AMP docs, goal-learnings § Patterns (wall-clock-fractional schedule, model converging at budget end).

**Estimated Effort**: Medium — ~20 lines: autocast context, GradScaler wrap, channels_last conversion, scaler.scale(loss).backward(), scaler.step(optimizer), scaler.update().

**Risk Assessment**: Medium. FP16 can cause NaN gradients (GradScaler handles this). channels_last has edge cases with some ops. Accuracy per-step may differ slightly from FP32, but overall accuracy should improve due to more epochs. If NaN occurs, the abort criteria catch it.

### 2. Batch size 256 + LR 0.2 (linear scaling rule)

**Summary**: Double BATCH_SIZE from 128 to 256 and LR from 0.1 to 0.2 (linear scaling). The wall-clock-fractional schedule thresholds (0.5, 0.75) remain unchanged. Batches per epoch halves from 390 to 195, but each step processes twice the data.

**Reasoning**: Larger batch increases GPU utilization. The H20 has ample VRAM (598 MB used of 98 GB). With the linear scaling rule, the optimization dynamics should be preserved. Steps per epoch halve, so the model sees the same amount of data per epoch but in fewer steps. The key question is whether the throughput gain (fewer steps per epoch) translates to more epochs.

**Sources**: Goyal et al. 2017 (linear scaling rule), goal-learnings about H20 VRAM headroom.

**Estimated Effort**: Very low — two constants.

**Risk Assessment**: Low-medium. The linear scaling rule is well-established but can be imprecise — LR=0.2 may be slightly too high with WD=5e-4 and augmentation. The generalization gap from larger batches is a concern at this accuracy level. The wall-clock-fractional schedule handles fewer steps naturally.

### 3. Compile the model with torch.compile

**Summary**: Add `model = torch.compile(model)` after model construction. PyTorch 2.x's torch.compile fuses ops and reduces Python overhead, typically giving 10-30% speedup on conv models.

**Reasoning**: torch.compile is the lowest-risk throughput improvement — it doesn't change training dynamics (same FP32 precision, same gradients). The first iteration has a compilation overhead but subsequent iterations are faster. On 69 epochs (~26K steps), the amortized compilation cost is negligible.

**Sources**: PyTorch 2.0 docs, torch.compile documentation.

**Estimated Effort**: Very low — one line.

**Risk Assessment**: Low. torch.compile is designed to be a drop-in improvement. The main risk is that compilation fails (some dynamic shapes or custom ops don't compile cleanly) or that the speedup is too small to clear the +0.1pp improvement threshold. The compilation overhead adds ~30-60s of startup time, reducing the effective training window.

## Idea Evaluation

**Expected throughput gain**: Candidate 1 (AMP) offers 1.5-2x speedup (the largest). Candidate 3 (compile) offers 10-30% (modest). Candidate 2 (batch size) offers a different kind of throughput — fewer steps per epoch, but each step may not be faster on the H20 for this small model.

**Risk**: Candidate 3 is lowest risk (same training dynamics). Candidate 2 is low-medium (well-understood scaling rule). Candidate 1 is medium (FP16 precision change).

**Impact**: If the model gains ~0.07pp/epoch in the polish phase, then: AMP with ~30 extra epochs → ~+2pp. Compile with ~10 extra epochs → ~+0.7pp. Batch size with uncertain epoch gain → uncertain.

**Feasibility**: Candidate 3 is one line. Candidate 2 is two constants. Candidate 1 is ~20 lines but well-documented.

**Selection**: Candidate 1 (AMP) offers the largest throughput gain and therefore the most potential accuracy improvement. However, I'll pick **Candidate 3 (torch.compile)** because it's the lowest-risk throughput path — same training dynamics (FP32), one line of code, and if it delivers even a modest speedup it compounds with the model's still-converging trajectory. If torch.compile doesn't yield enough gain, AMP is the natural follow-up.

Actually, reconsidering — the goal is to maximize accuracy, and AMP offers a much larger throughput gain. The risk of FP16 NaN is handled by GradScaler, and the abort criteria catch divergence. Let me pick AMP.

**Final selection**: Candidate 1 — AMP with GradScaler. The largest throughput gain targets the identified binding constraint (epoch count) most aggressively.

## Chosen Idea

**Selected**: Candidate 1 — **AMP (torch.cuda.amp) with GradScaler**

**Why this idea**: The model was still improving at epoch 69 (EXP-003) and EXP-004 confirmed that epoch count is the binding constraint. AMP offers the largest throughput improvement (1.5-2x on H20's Tensor Cores), translating to ~100+ epochs in the 300s budget. The wall-clock-fractional schedule handles the changed step rate automatically. GradScaler manages FP16 numerical stability.

**Hypothesis**: Adding AMP autocast (FP16), GradScaler, and channels_last memory format to the width-2x augmented WD=5e-4 baseline will raise best_test_acc from 93.33% to **93.8-94.5%** by fitting ~100+ epochs (vs 69) in the 300s budget, giving the polish phase substantially more training iterations. The improvement bar is 93.43%.
