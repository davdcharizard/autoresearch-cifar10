# Report EXP-005: Second ReZero-gated residual block in layer3 (capacity-via-depth probe)
- **Created**: 2026-06-28

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) within a fixed 300s training budget, editing only `train.py`. Baseline entering this experiment: **96.00%** (EXP-004, commit ae31206). Improvement bar: beat baseline by ≥0.1pp → **≥96.10%**.

## Idea & Hypothesis
Chosen idea: append a second ReZero-gated residual block — `GatedResidual(512)` — to `layer3` (operating at 4×4 spatial resolution), reusing the exact mechanism that worked in EXP-004 (where a ReZero block in layer2/8×8 lifted accuracy +0.13pp). The reasoning: EXP-004 proved that identity-init (ReZero α=0) residual capacity can be added safely without LR retuning and that the capacity gain can outrun its throughput cost. The hypothesis was that a *second* such block, placed at layer3/4×4 — the location with the **smallest activation footprint** and therefore the **least throughput penalty** — would add usable capacity while preserving the most annealing budget, pushing accuracy to ~96.05–96.15% (central ~96.08%). The decisive premise was that a 4×4/512 block is FLOP-equal to EXP-004's proven 8×8 block but cheaper in wall-clock.

## Approach
Single-token edit to `train.py:151`: `self.layer3 = nn.Sequential(conv_bn(256, 512), nn.MaxPool2d(2), Residual(512), GatedResidual(512))` — appending `GatedResidual(512)` to the existing layer3. The `GatedResidual` ReZero class (added in EXP-004) was reused verbatim; no other code changed. PEAK_LR held at 0.4 (ReZero's gradual α-ramp removes the LR-retune rationale → clean single-variable capacity test). Net grew from 10 to 12 learnable conv layers (+4.72M params, 12,505,268 total). Milestone-1 smoke confirmed: identity at init (α=0, `allclose(blk(h),h)`), a rigorous **2-step trainability check** (branch conv grad = 0 at α=0 on step 1; α moves to 5.26e-3; branch grad = 4.02 at step 2 once α≠0 — proving the block genuinely trains once the gate opens), correct output/pool shapes, exact param count, frozen whitening. Deviation from plan: none — implemented as specified. Note (plan-review #1): init is not bit-equivalent to EXP-004 (the new block's convs consume extra kaiming RNG before `fc`), a noise-level perturbation, not a confound.

## Execution
One clean run on GPU 1 (`CUDA_VISIBLE_DEVICES=1`), `timeout 600 ... uv run train.py`, exit 0, wall 432.0s. No retries, no divergence, no adjustments. Training ran the full 300.0s budget producing 131 epochs / 12645 steps. The run completed normally and reached verification.

## Results
- **Primary metric**: 95.90% (baseline: 96.00%, delta: **−0.10pp, −0.10%**) — peaked at ep128, final 95.77% @ ep131.
- **Observations**: Two findings overturned the idea's premise.
  1. **The 4×4/512 block ran SLOWER, not cheaper.** Throughput dropped to ~23.4k img/s (vs EXP-004's ~26.1k, ≈10% slower) → **131 epochs vs EXP-004's 142 (11 fewer)**. The FLOP-equal premise was falsified at the kernel level: cuDNN evidently selects a less-efficient kernel for small-spatial(4×4)/large-channel(512) convolutions — exactly the plan-review #6 risk. So the block bought *less* annealing budget, not more.
  2. **The 12-conv net never led EXP-004.** Trajectory: ep1 61.63 (vs 58.70, noise), **ep10 81.16 (vs 85.19 — 4.03pp BEHIND)**, ep25 92.73 (≈92.63), ep50 94.11 (≈94.00), ep100 95.32, peak 95.90 (vs 96.00). The deeper net converged *slower early* (extra depth + α-ramp) and only reached transient parity mid-training (+0.10pp at ep25/ep50), never a sustained lead, and finished behind. Contrast EXP-004's layer2 block, which showed a decisive mid-training capacity lead (ep25 92.63 vs EXP-003's 88.84).
- **Analysis**: The hypothesis fails on **both** legs. (a) The throughput-saving premise is false — a 4×4/512 block is wall-clock *more* expensive than EXP-004's 8×8/256 block despite equal FLOPs, because of cuDNN kernel-selection at small spatial size. (b) Even setting aside throughput, the added capacity produced no usable representational gain at this location: the deeper net never out-accuracied EXP-004 at matched epochs, so 4×4 coarse capacity is not where the model is bottlenecked. The −0.10pp is the combined cost of fewer epochs (under-annealing) plus zero capacity benefit. This is an approach-specific failure (layer3/4×4), not a refutation of ReZero capacity generally — EXP-004 remains the proof that the mechanism works at 8×8.
- **Key Learning**: A second ReZero block at layer3/4×4 gives −0.10pp — coarse 4×4 capacity is unused AND a 4×4/512 conv is wall-clock slower than a FLOP-equal 8×8/256 conv (cuDNN kernel selection), so depth-at-the-coarsest-stage is the wrong place to add capacity.

## Verification
- **Conditions**: C1 (clean run within wall guard) PASS; C2 (full 300s budget + scope: only train.py changed, prepare.py byte-identical) PASS; **C3 (improvement ≥+0.1pp, i.e. ≥96.10) FAIL** — best_test_acc 95.90 < 96.10, and below the 96.00 baseline.
- **Review Notes**: Results confirmed trustworthy. The metric is real (from `Eval.evaluate`; per-epoch max == summary best 95.90), the run was clean (exit 0, no divergence), scope was clean (one-token layer3 append, seed/eval/prepare untouched). No false-failure risk (the value is well-formed and consistent across the log). No reward-hacking concern — the change is squarely within the allowed intervention class (architecture in train.py).
- **Verdict**: no-improvement
- **Verdict Basis**: Valid run, necessary condition C3 (metric ≥ bar) failed — metric below baseline. Not invalid (no constraint violation) and not a crash (clean completion).

## Unexplored Avenues
- **Second ReZero block at layer2/8×8 instead of layer3/4×4** — the fallback noted in the brainstorm. EXP-004's winning block was at layer2/8×8; a *second* block at the same proven stage would add capacity where it demonstrably helped, and 8×8/256 convs run at EXP-004's full ~26k img/s (no 4×4 kernel penalty). This is the natural next capacity probe if depth-via-ReZero is pursued further. Risk: layer2 may already be near its capacity sweet spot (diminishing returns), and 2 extra 8×8 blocks cost more wall-clock than 1.
- **Width over depth** — widen an existing stage (e.g. layer3 channels 512→640) rather than adding depth, ReZero-gating any new channels. Untested whether the bottleneck is depth or width.
- **Spend the budget on TTA, not capacity** — pivot away from architecture: multi-crop TTA (airbench tta_level=2, currently only flip-TTA) is the documented path from ~96.0 to ~96.05–96.1 at this scale, and costs no training throughput. Strong candidate given capacity-via-depth at layer3 is now exhausted.

## Next Steps
1. **Pivot to multi-crop TTA** (medium-high confidence) — flip-TTA only is currently used; adding translation/multi-crop TTA at eval is the documented lever from 96.0→~96.05–96.1 for this net scale, with zero training-budget cost. Highest expected value.
2. **Second ReZero block at layer2/8×8** (low-medium confidence) — the proven-stage capacity probe; avoids the 4×4 kernel penalty but may hit diminishing returns and costs ~2× EXP-004's throughput hit.
3. **Tune existing knobs** (medium confidence) — a focused sweep of PEAK_LR / EMA_DECAY / TTA_START_FRAC / cutout size around the 96.00 recipe may recover ≥0.1pp more cheaply than further architecture changes.

## Exit Action Results
<!-- No exit actions defined for this goal. -->
- None — no exit actions defined.
