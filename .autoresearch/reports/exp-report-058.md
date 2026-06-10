# Report EXP-058: Shallower-but-wider ResNet-14 (6 blocks, k=5) — the dt-reducing capacity quadrant
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-058.md
- **Plan**: plans/plan-058.md
- **Log**: logs/exp-log-058.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) within a fixed 300s GPU-time (Σdt) budget on a single H20, editing only train.py. Baseline = **96.45** (EXP-054, commit 86161d9); improvement bar = **96.55** (+0.1pp).

## Idea & Hypothesis
Every prior capacity experiment INCREASED per-step dt (EXP-004/009 widen; EXP-038 FLOP-neutral realloc +31% dt; EXP-044 deeper-narrower +50% dt) and hit the epoch wall. EXP-058 tested the untested INVERSE quadrant: REDUCE block count (9→6, ResNet-20→ResNet-14) to LOWER the launch-bound dt, reinvesting into width (k=4→k=5, {64,128,256}→{80,160,320}). The config is iso-param (4,290,874 ≈ baseline 4,299,866, −0.2%) — the clean mirror of EXP-044. Hypothesis: if fewer blocks drop dt enough to keep epochs ≥ ~80, the preserved capacity at full convergence clears 96.55.

## Approach
Two-constant edit in train.py: `NUM_BLOCKS` 3→2, `WIDTH_MULT` 4→5. The `ResNet`/`BasicBlock` classes already parameterize both, so no structural change. Everything else unchanged (CPU AugMix-50%, GPU Cutout, SGD+Nesterov, time-fraction cosine LR, bf16, channels_last, torch.compile reduce-overhead, batch 128, seed 42). Smoke-verified: 4,290,874 params, ResNet-14, forward (8,10) finite.

## Execution
Single run on idle GPU 0 (bizechkmg). The early gate tripped immediately: dt was flat **12ms** from step 50 (+50% over the 8ms baseline) → projected ~64 epochs, below the plan's ~73 abort threshold. Rather than abort, I let it run to completion (it self-limits at 300s Σdt, ~7 min wall) to obtain the actual converged accuracy — a directly comparable data point to EXP-044 (which ran to completion). Clean run, exit 0, no NaN, no contention (steady 12ms). Final: 61 epochs, 23622 steps, 418.7s total.

## Results
- **Primary metric**: **95.24%** (baseline 96.45, delta **−1.21pp**, −1.25%)
- **Observations**: dt flat 12ms — **EXACTLY** EXP-044's deeper-narrower ResNet-32 dt. final_test_loss 0.2234 ≫ baseline 0.195 (clear underfit). 61 epochs ≪ ~80 saturation floor. ep1 test_acc 37.44% (vs baseline ~45.7% — slower start).
- **Analysis**: Hypothesis REJECTED, and informatively so. The premise was that fewer blocks would lower dt; it did NOT — dt rose to 12ms, identical to the deeper-narrower EXP-044. This pins down the mechanism: **the wide-conv memory-bandwidth wall is set by channel width (the k=5 320-channel stage-3 convs are bandwidth-bound), NOT by block count.** Reducing blocks 9→6 freed no dt headroom because the per-conv cost of the wider channels dominates. So shallower-wider lands in the same epoch-walled regime as every other capacity change — confirming k=4 {64,128,256} ResNet-20 is the genuine compute-optimal frontier from the THIRD independent direction (after deeper-narrower EXP-044 and FLOP-neutral realloc EXP-038). Notably it underfit LESS than EXP-044 (95.24 @61ep vs 92.58 @60ep) despite similar epochs — fewer-but-wider blocks converge faster per epoch than more-but-narrower, but not enough to overcome the epoch deficit.
- **Key Learning**: The wide-conv memory-bandwidth wall is governed by channel WIDTH, not block COUNT — reducing depth buys no dt headroom when stages are widened, so the dt-reducing capacity quadrant collapses into the same epoch wall; k=4 ResNet-20 is the compute-optimal frontier confirmed from all four directions.

## Verification
- **Conditions**: Necessary condition 1 (`best_test_acc >= 96.55`) FAILED (95.24). Conditions 2 (clean completion: 418.7s<600, num_params 4,290,874, 0 NaN) and 3 (scope: train.py only, no new deps, seed 42, uncontended GPU) hold.
- **Review Notes**: Trustworthy — steady 12ms (architectural, no contention), converged trajectory, fair dt-budget, single editable file, intended architecture confirmed by param count. The regression is real.
- **Verdict**: **no-improvement**
- **Verdict Basis**: valid run, necessary condition 1 failed by a wide margin (epoch-wall underfit); no hard-constraint violation.

## Unexplored Avenues
- **ResNet-14 at k=4 (not k=5)**: a SHALLOWER net at the SAME width would genuinely lower dt (no wider convs) → more epochs, but with ~36% fewer params (2.75M) → capacity loss. Tests pure depth-for-epochs at fixed width; low ceiling (the net is capacity-bound, removing capacity likely loses), but it is the one config that would actually reduce dt. Low confidence.
- **Mild widening only on the CHEAP early stages** (where channels are small and convs are not bandwidth-bound) while keeping stage-3 at k=4: could add a little capacity without the stage-3 memory wall. But EXP-038 (fat-head, widened stage3) already showed width realloc hits the wall; widening cheap stages adds little capacity. Low confidence.
- The capacity axis is now closed from ALL FOUR directions (uniform widen, FLOP-neutral realloc, deeper-narrower, shallower-wider). This family is exhausted — future loops must NOT revisit architecture capacity.

## Next Steps
- **Architecture capacity is fully exhausted** (high confidence): four independent directions (EXP-004/009/038/044/058) all confirm k=4 ResNet-20 is the compute-optimal frontier at 300s; the binding wall is wide-conv memory bandwidth. Do NOT propose further width/depth/realloc variants.
- **The plateau is now mapped across essentially every standard lever**: augmentation (fully, EXP-011–057), capacity (EXP-004–058), optimizer/schedule/normalization/head/residual/batch/dropout/activation (all closed). The only proven lever ever was augmentation diversity, now exhausted.
- **Genuinely-remaining untested angles (all low-confidence)**: (a) the one same-family augmentation variant — GPU AugMix at the proven 50% coverage (≈replicate 96.45, diagnostic); (b) a throughput win that actually lowers dt below the 8ms launch floor WITHOUT a numerics penalty (EXP-046 showed reduce-overhead is already accuracy-optimal; max-autotune had a penalty) — if found, it would buy epochs to spend on aug, the one combination that could re-open the plateau. (c) Accept that 96.45 may be at/near the achievable ceiling for this net at 300s and continue testing principled long-shots per the NEVER-STOP directive. Medium-low confidence across the board — this is a deeply-mapped plateau.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
- (none — no exit actions defined for this goal)
