# Brainstorm EXP-020
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **airbench96 — 96% CIFAR-10 in <1s on A100** (https://github.com/KellerJordan/cifar10-airbench)
  Uses 12-pixel Cutout (not RandomErasing), wider architecture (128/512 channels), extra conv per block, and LR decaying smoothly to zero. The smooth LR decay to zero (rather than step decay) is a recurring pattern in high-accuracy CIFAR-10 recipes.

- **OneCycleLR / Superconvergence for CIFAR-10** (web search results from PyTorch forums and blog posts)
  OneCycleLR with SGD achieves 92-94% in 20-50 epochs on standard ResNets. The technique uses a triangular LR schedule with warm-up then annealing. Results suggest it accelerates convergence but hasn't been shown to clearly exceed multi-step schedules at the same epoch count on CIFAR-10.

- **Muon optimizer in CIFAR-10 speedruns** (web search results)
  Latest speedrun records use the Muon optimizer (orthogonalized momentum) to achieve 94% in 2.59s on A100. However, this requires custom optimizer code and the results are for a different accuracy tier.

- **CosineAnnealingLR for CIFAR-10** (PyTorch documentation + training recipes)
  CosineAnnealingLR with T_max matched to actual training duration provides smooth decay to near-zero LR, avoiding the abrupt transitions of step decay. Widely used in modern training recipes. EXP-000 failed with CosineAnnealingLR because T_max=200 was far too large for the ~91 actual epochs — the LR never reached the low regime. With T_max correctly set to ~100 epochs, the schedule would decay LR from 0.2 to ~0 over the full training duration.

## Experimental History Review

- **Current best**: 95.91% (EXP-019, TTA horizontal flip on top of EXP-015 baseline 95.57%)
- **Improvement trajectory**: BASE 91.72 → width-2x 92.29 → augmentation 92.92 → WD=5e-4 93.33 → AMP 94.44 → width-4x 94.82 → batch 256 95.39 → label smoothing 95.57 → TTA 95.91
- **Regularization stack is saturated**: CutMix (EXP-010), Mixup (EXP-017), DropPath (EXP-018) all hurt when stacked on TrivialAugmentWide+RandomErasing+LS=0.2+WD=5e-4
- **EMA was marginal**: Full state_dict EMA (EXP-014) gave only +0.05pp with throughput cost; parameter-only EMA (EXP-013) was broken for BN models
- **SE blocks too expensive**: Both implementations (EXP-011, EXP-012) cost ~9ms/step, reducing epochs 98→83
- **LR schedule is critical**: Wall-clock-fractional MultiStepLR at (0.5, 0.75) is validated (EXP-001, High Importance pattern). Earlier drops (0.35/0.55) hurt (EXP-006). CosineAnnealingLR failed only due to T_max=200 mismatch (EXP-000) — not a rejection of cosine decay itself.
- **BN momentum tuning marginal**: +0.02pp (EXP-016), not worth pursuing further
- **torch.compile zero speedup on H20**: (EXP-008)
- **Untried gaps**: CosineAnnealingLR with correct T_max, deeper architecture (ResNet-26/32), different optimizers (AdamW, LAMB), Cutout (used in airbench96 but different from RandomErasing)

## Candidate Ideas

### 1. CosineAnnealingLR with T_max Matched to Training Duration
**Summary**: Replace the wall-clock-fractional MultiStepLR schedule with CosineAnnealingLR, setting T_max to the actual number of epochs (~98-100). The current schedule drops LR abruptly by 10x at 50% and 75% wall-clock progress. CosineAnnealingLR provides a smooth decay from the initial LR (0.2) to near zero, spending more time in the intermediate LR regime and reaching a lower final LR. This is a drop-in change to the scheduler construction — no other code modifications needed. The 5-epoch linear warmup is retained by applying it as a multiplier on top of the cosine schedule.

**Reasoning**: EXP-000 failed with CosineAnnealingLR because T_max=200 was far too large for ~91 actual epochs — the LR never dropped below 0.058, missing the critical low-LR convergence regime. This is a known failure mode (goal-learnings § Failed Approaches, Low Importance). With T_max correctly set to ~100, the cosine schedule will decay LR to ~0 by the end of training — reaching a much lower final LR than the current schedule's 0.002 (0.2 × 0.01). The smooth decay also avoids the accuracy oscillation observed in the intermediate LR=0.02 regime with AMP (EXP-005 pattern). Modern high-accuracy CIFAR-10 recipes (airbench96) use smooth LR decay to zero rather than step schedules. Zero throughput cost — same epoch count, same per-step time.

**Sources**: EXP-000 failure analysis (goal-learnings), EXP-005 AMP instability pattern, airbench96 recipe, PyTorch CosineAnnealingLR docs

**Estimated Effort**: low — single scheduler change, ~5 lines of code

**Risk Assessment**: Replaces a High Importance validated pattern (MultiStepLR drops at 0.5/0.75). The step schedule has been the backbone of 8 successive improvements. If cosine doesn't work, the regression should be small (<0.5pp) since the LR still reaches low values. The main risk is that the abrupt drops at 50%/75% create a specific convergence dynamic that cosine doesn't replicate — the first drop is responsible for the +3pp accuracy jump (EXP-001).

### 2. Deeper Architecture: NUM_BLOCKS=4 (ResNet-26)
**Summary**: Increase NUM_BLOCKS from 3 to 4, creating a ResNet-26 (6×4+2=26 layers). This adds one BasicBlock per stage (3 stages), increasing model depth from 20 to 26 layers and parameter count from ~4.3M to ~5.7M. The same WIDTH_MULT=4 is retained, so channel widths remain {64, 128, 256}. The additional capacity comes from depth rather than width — adding more nonlinear transformations per spatial resolution.

**Reasoning**: The model has been at WIDTH_MULT=4 since EXP-007, and the regularization stack is saturated — additional regularizers hurt rather than help. The natural next axis for accuracy improvement is capacity. Width has been extensively explored (WIDTH_MULT 1→2→4), but depth has been fixed at NUM_BLOCKS=3 throughout. Deeper models can learn more complex feature hierarchies. With ~80-85 estimated epochs in 300s (vs 98 at current depth), the model still gets sufficient training. The existing regularization stack (TrivialAugmentWide+RandomErasing+LS=0.2+WD=5e-4) is well-calibrated and should prevent overfitting at the higher parameter count.

**Sources**: EXP-019 report next steps (medium confidence for deeper architecture), ResNet paper (He et al. 2015), goal-learnings pattern on epoch count

**Estimated Effort**: low — single constant change (NUM_BLOCKS = 4), no other code modifications

**Risk Assessment**: Fewer epochs (~80-85 vs 98) means ~15% less training. The wall-clock-fractional LR schedule adapts automatically, but the regularization stack was calibrated for ~98 epochs. With fewer epochs, the model may not converge as tightly in the polish phase. SE blocks (EXP-011/012) showed that losing 15 epochs to overhead costs accuracy even when adding capacity. However, depth adds capacity without per-step overhead increase (unlike SE), so the tradeoff is different — more capacity per epoch, fewer total epochs.

### 3. Cutout Replacing RandomErasing
**Summary**: Replace RandomErasing(p=0.25, scale=(0.02, 0.2)) with Cutout using a 12-pixel square patch. Cutout zeroes a fixed-size square region of the input image, while RandomErasing replaces a random-aspect-ratio rectangle with random values. The 12-pixel patch size comes from airbench96's validated recipe. Implementation: add a custom Cutout transform class (or use `transforms.RandomErasing` with `value=0` and fixed square aspect ratio) applied after normalization.

**Reasoning**: airbench96 achieves 96% CIFAR-10 accuracy using 12-pixel Cutout as a key component. Cutout's fixed-size square occlusion may be more effective than RandomErasing's variable-aspect random-fill approach because: (1) zeroing the occluded region creates a stronger regularization signal than random fill, (2) the fixed 12×12 size (covering ~14% of the 32×32 image) is optimized for CIFAR-10's spatial scale, vs RandomErasing's variable 2-20% coverage. Since the regularization stack is near saturation, this is a substitution (not addition) — removing one input-space augmenter and replacing with another.

**Sources**: airbench96 recipe (12px Cutout), original Cutout paper (DeVries & Taylor 2017), goal-learnings on regularization saturation

**Estimated Effort**: low — replace one transform in the augmentation pipeline

**Risk Assessment**: RandomErasing and Cutout are functionally similar — both occlude regions of the input. The improvement may be marginal or zero. The goal-learnings show the regularization stack is near saturation, so swapping one occlusion method for another may not move the needle. Additionally, the 12-pixel Cutout in airbench96 is part of a different architecture and training recipe — the optimal patch size may differ for our ResNet-20 WIDTH_MULT=4.

## Idea Evaluation

**Evidence strength**: CosineAnnealingLR has the strongest evidence — EXP-000's failure was conclusively attributed to T_max mismatch (not cosine decay itself), and airbench96's use of smooth decay to zero validates the approach at the 96% accuracy tier. Cutout has decent evidence from airbench96 but the comparison with RandomErasing is indirect. Deeper architecture has the weakest direct evidence — no prior depth experiments exist for this goal.

**Mechanism clarity**: CosineAnnealingLR has the clearest mechanism — it provides a lower final LR (→0 vs current 0.002), smoother transitions (avoiding AMP instability at LR=0.02), and more time in intermediate LR regimes. Cutout's mechanism (stronger occlusion signal via zeroing) is plausible but incremental over RandomErasing. Deeper architecture's mechanism (more nonlinear capacity) is sound but the epoch-count tradeoff makes the net effect uncertain.

**Expected impact**: CosineAnnealingLR is likely to have moderate impact (+0.1-0.3pp) — the lower final LR alone should tighten convergence. Cutout is likely marginal (0 to +0.15pp) given functional similarity to RandomErasing. Deeper architecture has the highest ceiling but also the highest floor risk due to epoch loss.

**Risk profile**: CosineAnnealingLR has the safest failure mode — if it doesn't help, the accuracy should be comparable to baseline (within ±0.2pp). It replaces a High Importance pattern, but the replacement is evidence-based and the T_max issue from EXP-000 is resolved. Cutout is equally safe. Deeper architecture has a wider variance.

**Feasibility**: All three are low effort. CosineAnnealingLR and Cutout are both ~5-line changes. Deeper architecture is a single constant change.

## Chosen Idea
**Selected**: CosineAnnealingLR with T_max Matched to Training Duration

**Why this idea**:
Strongest evidence among the three candidates — EXP-000's failure was conclusively diagnosed as a T_max mismatch rather than a cosine schedule problem, and high-accuracy CIFAR-10 recipes (airbench96 at 96%) use smooth LR decay to zero. The mechanism is clear: reaching a lower final LR (~0 vs current 0.002) tightens convergence in the polish phase, and smooth decay avoids the AMP instability observed at the intermediate LR=0.02 regime. Zero throughput cost — identical epoch count and per-step time. The approach is a focused, low-risk intervention that targets the optimization schedule rather than adding regularization (saturated) or capacity (epoch-cost tradeoff).

**Hypothesis**:
Replacing the wall-clock-fractional MultiStepLR (drops at 0.5/0.75 to 0.1×/0.01×) with CosineAnnealingLR (T_max=~100, decay from 0.2 to ~0) will improve best_test_acc by +0.1-0.3pp (target: >96.01%) because the smooth decay to a lower final LR will produce tighter convergence in the final training phase, and the intermediate LR regime will be traversed smoothly rather than via the abrupt 10× drop that causes transient AMP instability.
