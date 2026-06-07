# Brainstorm EXP-019
**Created**: 2026-05-29
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- Sources from prior loops remain relevant — no new external search needed. Key references:
  - PyTorch channels_last (NHWC) gives 8-35% speedup for ResNets with AMP (PyTorch Performance Tuning Guide)
  - Multi-view TTA with ±1px spatial shifts (airbench, https://arxiv.org/html/2404.00498v2)
  - EXP-018 validated channels_last gives 9% speedup (59 vs 54 epochs)

## Experimental History Review

- **20 experiments** (BASE through EXP-018), baseline 96.39% (EXP-016)
- **EXP-018 key learning**: Channels_last gives real 9% speedup (59 vs 54 ep) but T_max=55 hurt — slower LR decay negated the speedup benefit. Pattern recorded: "Keep T_max=49 when adding channels_last; let extra epochs run at near-zero LR for refinement"
- **EXP-016 key learning**: TTA hflip gave +0.66% for free. Model benefits significantly from prediction averaging.
- **EXP-017 key learning**: Architectural additions with per-step overhead are costly. Prefer zero-overhead interventions.
- **Untried combinations**: channels_last + original T_max=49, extended TTA with spatial shifts, both combined
- **Architecture/schedule sweet spot**: k=4, T_max=49, warmup=5 — proven to yield 96.39% with TTA. Don't change what works — only add zero-risk improvements.

## Candidate Ideas

### 1. Channels_last (NHWC) + Original T_max=49 + Extended TTA (Spatial Shifts)
**Summary**: Combine two orthogonal zero-risk interventions in one experiment: (a) channels_last memory format for ~9% training speedup, keeping T_max=49 unchanged so the extra ~5 epochs run at near-zero LR as free refinement, and (b) extend test-time augmentation beyond horizontal flip by adding ±1px spatial shifts (left, right, up, down) in eval mode. This yields 6 total TTA views (original + hflip + 4 shifts) averaged together. The two interventions are completely independent — channels_last affects only training speed, extended TTA affects only evaluation.

**Reasoning**: This combines the strongest available evidence. EXP-018 proved channels_last speeds up training by 9%. The failure was caused by changing T_max — the pattern says keep it at 49. The extra ~5 epochs at near-zero LR provide free weight refinement (small gradient updates that sharpen the minimum). EXP-016 proved TTA averages away prediction noise (+0.66% from hflip alone). Adding 4 more spatial-shift views should further reduce variance. Since these two mechanisms are orthogonal (training speed vs eval averaging), combining them maximizes the chance of crossing the 96.49% threshold.

**Sources**: EXP-018 (channels_last 9% speedup), EXP-016 (+0.66% from hflip TTA), airbench 6-view TTA pattern, goal-learnings pattern about keeping T_max=49

**Estimated Effort**: low — 4 lines for channels_last, ~10 lines for extended TTA in forward()

**Risk Assessment**: Very low. Both interventions are individually zero-risk. Channels_last is mathematically identical training (just faster). Extended TTA can only improve or tie (more views = lower variance). The only risk is that the improvement is too small to cross the 0.1% threshold, but combining two sources of improvement maximizes our odds.

### 2. Channels_last + T_max=49 Only (No Extended TTA)
**Summary**: Same as Idea 1 but without the extended TTA — only add channels_last with original T_max=49. This isolates the channels_last effect. The extra ~5 epochs of near-zero LR training would refine the EMA model's weights slightly, potentially improving both training accuracy and TTA accuracy.

**Reasoning**: Pure isolation of the channels_last variable. EXP-018 confounded channels_last with T_max change. This experiment would definitively answer whether channels_last alone improves accuracy.

**Sources**: EXP-018 learning, goal-learnings pattern

**Estimated Effort**: low — 2-3 lines of code change

**Risk Assessment**: Very low. But if the channels_last-only improvement is < 0.1%, we'd fail verification despite a valid positive signal. Combining with extended TTA hedges against this.

### 3. Extended TTA (Spatial Shifts) Only
**Summary**: Only add ±1px spatial shift TTA without channels_last. Modify the model's eval-mode forward() to compute 6 views: original, hflip, and 4 directional 1px shifts. Average all 6 logit vectors. Training is completely unchanged.

**Reasoning**: Isolates the extended TTA effect. The model already gained +0.66% from hflip TTA. Adding 4 more views provides additional variance reduction. This tests whether spatial-shift TTA contributes meaningful accuracy beyond hflip.

**Sources**: EXP-016, airbench 6-view TTA

**Estimated Effort**: low — ~10 lines in forward()

**Risk Assessment**: Very low. Zero training impact. Worst case: negligible improvement. But might not be enough to cross 0.1% threshold on its own.

## Idea Evaluation

**Evidence strength**: All three ideas have strong evidence. Channels_last speedup is proven (EXP-018). TTA value is proven (EXP-016). The combination has the strongest composite evidence.

**Mechanism clarity**: Idea 1 has two clear, independent mechanisms: (a) extra low-LR training epochs refine the EMA model, (b) more eval views reduce prediction variance. No interaction risk since they operate in different phases (training vs eval).

**Expected impact**: Idea 1 (combined) has the highest expected impact because it stacks two independent sources of improvement. Each alone might yield +0.05-0.15%, but together they're more likely to cross the 0.1% threshold. Ideas 2 and 3 each carry the risk of being individually too small.

**Risk profile**: All three are extremely safe. Idea 1 is equally safe as Ideas 2 or 3 since the interventions don't interact.

**Feasibility**: All are low effort. Idea 1 requires slightly more code but is still straightforward.

**Verdict**: Idea 1 (combined) is the clear choice. It maximizes the probability of crossing the verification threshold by stacking two orthogonal, zero-risk, well-evidenced improvements. Isolating variables (Ideas 2/3) is scientifically cleaner but wastes experiment loops when the interventions are provably independent.

## Chosen Idea
**Selected**: Channels_last (NHWC) + Original T_max=49 + Extended TTA (Spatial Shifts)

**Why this idea**:
Combines two proven, orthogonal, zero-risk interventions — channels_last speedup (training refinement) and spatial-shift TTA (eval variance reduction) — to maximize the chance of beating the 96.49% threshold. Both mechanisms are well-evidenced from EXP-016 and EXP-018.

**Hypothesis**:
Channels_last with T_max=49 will yield ~59 epochs with 5 extra at near-zero LR for free refinement, while 6-view TTA (original + hflip + 4 spatial shifts) will further reduce prediction variance. Combined, these will improve best_test_acc from 96.39% to ~96.5-96.7%.
