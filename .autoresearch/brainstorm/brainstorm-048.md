# Brainstorm EXP-048
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **GridMask Data Augmentation** (Chen, Chen, Yang, Mu, Wang & Zeng, 2020, arXiv:2001.04086): drops a regular GRID of small squares from the image (controlled by unit length `d`, kept-ratio `r`, and a random offset), rather than one large hole (Cutout) or random scattered pixels (Random Erasing). The structured, DISTRIBUTED deletion balances "delete enough to force redundancy" against "keep enough continuous region to preserve object structure" — reported to beat Cutout on CIFAR-10/100 and ImageNet. Fully GPU-vectorizable (build a binary mask from arithmetic on coordinate grids, multiply) → throughput-neutral, exactly like the project's existing `cutout_batch`.
- **project context** (`cutout_batch` in train.py; EXP-002/003/013): occlusion is a PROVEN lever on this net — Cutout lifted 94.90→96.00 (+1.1pp combined) and is "orthogonal to TA, not redundant" (EXP-013). But only Cutout's HOLE SIZE was ever varied (8/16/20px); the occlusion PATTERN (one hole vs a distributed grid) has never been tested. The only plateau-breaker (TrivialAugment, EXP-012, +0.22) won by SUBSTITUTING a more effective augmentation at zero convergence cost — the same shape as a Cutout→GridMask swap.
- **project-insights / goal-learnings** (local): augmentation QUALITY tweaks have nulled (border-mode EXP-037), and ADDING regularizers regresses on this saturated recipe (CutMix EXP-018, dropout EXP-022, GhostBN EXP-047) — so the right framing is a SWAP (replace Cutout, hold occlusion strength ≈ constant), not an ADD.

## Experimental History Review

- **Current best / baseline**: 96.22% (EXP-012, 6c417a4), k=4 ResNet-20, ~91 ep @ 8ms. **38 consecutive no-improvements** (EXP-013..047). Epoch-SATURATED at ~91 (EXP-007/045/046) and compute-optimal.
- **ALL conventional axes CLOSED**: capacity (width/depth/realloc EXP-004/009/038/044), augmentation [strength EXP-013/021, policy EXP-014, mixing EXP-011/018, cooldown EXP-033/034/035, border-quality EXP-037], LR schedule (peak/floor/shape), regularizer-adds (dropout/WD/Mixup/CutMix), classifier-head, feature-routing, activations, weight-averaging (EMA/SWA — incl. Lookahead, explicitly do-not-retry), optimizer (AdamW/GC/SAM/PolyLoss), bag-of-tricks, large-batch, throughput→epochs, and **normalization (GhostBN, EXP-047 −1.06pp)**.
- **Two governing walls**: epoch wall (compute/layer adds → underfit) and polish-vs-top1 wall (compute-neutral tweaks → loss↓, top-1 flat; the net is generalization-bound). The ONLY escape ever found: a throughput-neutral change that increases effective input-data diversity (TrivialAugment).
- **The one untested augmentation sub-lever**: occlusion PATTERN. Cutout (single 16px hole) has been the fixed occlusion since EXP-002; its size was tuned but never its spatial structure. GridMask (distributed grid of small holes) is a distinct, literature-backed occlusion pattern never tried here.
- **Near-misses worth noting (for the alternate)**: EXP-034 aug-cooldown @0.10 → 96.26 (the ONLY above-baseline result, +0.04 within noise); GC (EXP-031) and PolyLoss (EXP-041) improved loss at ≈baseline top-1.

## Candidate Ideas

### 1. GridMask occlusion — SWAP Cutout's single hole for a distributed grid of holes (occlusion-PATTERN test)
**Summary**: Replace `cutout_batch` (one 16×16 hole) with a GPU-vectorized GridMask that deletes a regular grid of small squares (unit length `d` sampled per-image from ~[8,16], kept-ratio `r` chosen so the deleted-area fraction ≈ Cutout-16's ~25%, with a random spatial offset). Keep everything else identical (TA + crop + flip + the rest of the recipe). This tests whether a DISTRIBUTED occlusion pattern is a more effective regularizer than one large hole, at matched occlusion strength.
**Reasoning**: Occlusion is a proven, non-redundant lever here (Cutout = +1.1pp, orthogonal to TA), but only its size was tuned — the PATTERN is the single untested augmentation sub-lever. The mechanism mirrors the only plateau-breaker (TrivialAugment SUBSTITUTED a better augmentation at zero convergence cost). GridMask's distributed deletion preserves more global object structure while still forcing feature redundancy, which literature reports beats Cutout on CIFAR. As a SWAP at matched strength it avoids the over-regularization that sank every ADD (CutMix/dropout/GhostBN), and it's GPU-vectorized → throughput-neutral (no epoch wall).
**Sources**: Chen et al. 2020 (GridMask); train.py `cutout_batch`; EXP-002/003/012/013 (occlusion is a real, orthogonal lever); EXP-037 (aug-quality null — the risk).
**Estimated Effort**: low — write a `gridmask_batch(x, ...)` mirroring the existing `cutout_batch` vectorized-mask pattern; swap the one call site. Recipe/optimizer/schedule/seed untouched.
**Risk Assessment**: MAIN RISK — augmentation QUALITY has nulled on this saturated net (border-mode EXP-037), so GridMask may land within ±0.25pp (no-improvement) and merely close the occlusion-pattern sub-lever. Lower risk of regression than an ADD because it's a matched-strength SWAP. dt risk near-zero (same vectorized-mask construction as Cutout — verify dt stays 8ms). Seed-clean (uses the same seeded torch RNG as Cutout; deterministic mask math). Worst case: a clean throughput-neutral no-improvement.

### 2. Combine near-misses — augmentation cooldown (EXP-034 @0.10) + Gradient Centralization (EXP-031)
**Summary**: Run the EXP-034 aug-cooldown (disable TA+Cutout for the final 10% of the time budget — the only lever that ever exceeded baseline, 96.26) TOGETHER with the throughput-neutral Gradient Centralization from EXP-031 (compile'd, hoisted param list). Hypothesis: GC improves the converged optimization state (it lowered loss to 0.1894) and the cooldown's clean-data tail fine-tuning (+0.21 climb) stacks on that improved base to clear +0.1.
**Reasoning**: The NEVER-STOP directive explicitly suggests combining near-misses. These two are the project's best throughput-neutral levers (both proven 8ms/91ep, no epoch wall) and act via orthogonal mechanisms (gradient-space regularization × input-distribution tail fine-tuning).
**Sources**: EXP-034 (cooldown 96.26), EXP-031 (GC throughput-neutral, loss 0.1894); reports/exp-report-034.md, -031.md.
**Estimated Effort**: medium-high — must re-implement BOTH the mid-run TA/Cutout disable (cooldown, dataloader-switch or flag) and the compiled hoisted-GC step; two interacting closed-axis components.
**Risk Assessment**: Re-treads two CLOSED axes; both were closed because their standalone effect is within the ±0.25pp base-jitter noise, so additivity to +0.1 is uncertain (GC's standalone top-1 effect was ~0). Higher implementation complexity → more bug/throughput-confound surface. Likely lands within noise.

### 3. Input per-channel std normalization (the untested `std=(1,1,1)`)
**Summary**: The recipe normalizes inputs with `std=(1,1,1)` (mean-subtract only, never divides by per-channel std ≈ (0.247,0.243,0.261)). Switch to true per-channel std normalization.
**Reasoning**: Genuinely never tested; changes the relative R/G/B scaling into conv1 (which has no preceding BN), so it is NOT fully BN-absorbed and slightly re-weights the stem's channel sensitivity vs Kaiming init.
**Sources**: train.py `mean, std` line + its README comment; standard CIFAR normalization.
**Estimated Effort**: low — one-line constant change.
**Risk Assessment**: Most of the effect IS absorbed by bn1 right after conv1, so the expected impact is tiny → almost certainly within noise (polish wall). Lowest-information experiment of the three.

## Idea Evaluation

After 38 no-improvements with every axis mapped, the only escape mechanism ever observed is a throughput-neutral change that raises effective input-data diversity (TrivialAugment). The move must therefore be throughput-neutral (no epoch wall), target generalization (not loss/calibration), and ideally be a SUBSTITUTION rather than an ADD (every ADD regressed on this saturated recipe).

- **Evidence strength**: #1 sits on the strongest in-project signal — occlusion is a proven, orthogonal, non-redundant lever (Cutout +1.1pp), its pattern is the one untested sub-lever, and GridMask has direct CIFAR literature beating Cutout. #2 rests on two levers whose standalone effects were within noise (uncertain they add). #3's effect is largely BN-absorbed.
- **Mechanism clarity**: #1 is crisp (better-structured occlusion = better redundancy-forcing regularizer at matched strength, the TrivialAugment-substitution shape). #2's additivity mechanism is plausible but speculative. #3 is mostly a no-op by construction.
- **Expected impact / risk**: #1 is the best risk-adjusted shot — throughput-neutral, a matched-strength SWAP (low regression risk), genuinely novel, fails gracefully (closes the occlusion-pattern sub-lever). #2 has a higher ceiling in theory but high implementation complexity + closed-component risk + likely-within-noise. #3 is near-certain noise.
- **Feasibility**: #1 and #3 are low-effort; #2 is medium-high and bug-prone.

#1 wins: it is the only candidate that is simultaneously genuinely untried, throughput-neutral, a matched-strength substitution (not a saturating add), mechanism-clear, and aligned with the only escape mechanism this project has ever found. #2 is the directive's "combine near-misses" but its components are noise-level and complex; held as the next move if #1 nulls. #3 is deprioritized as near-certain polish-wall noise.

## Chosen Idea
**Selected**: GridMask occlusion — swap Cutout's single hole for a distributed grid of holes at matched occlusion strength

**Why this idea**:
Occlusion is one of only two interventions that ever moved top-1 on this net (Cutout +1.1pp; TrivialAugment +0.22pp), and it is explicitly orthogonal to TrivialAugment (EXP-013) — yet the occlusion PATTERN was never varied (only its size). GridMask is a distinct, literature-backed occlusion structure (distributed grid vs one hole) that, at matched deleted-area fraction, isolates pattern from strength. It is GPU-vectorizable → throughput-neutral (dodging the epoch wall that closed every capacity idea), and a SWAP rather than an ADD (dodging the over-regularization that sank every regularizer-add on this saturated recipe). It directly mirrors the only plateau-breaking mechanism: substituting a more effective augmentation at zero convergence cost.

**Hypothesis**:
Replacing Cutout-16 with a matched-strength GridMask keeps dt at ~8ms and epochs at ~91, and IF a distributed occlusion pattern regularizes more effectively than a single hole on this net, best_test_acc rises ≥0.1pp over 96.22 (≥96.32). The more likely outcome, given augmentation-quality nulls on this saturated recipe (EXP-037) and the robust plateau, is a landing within ±0.25pp of 96.22 (no-improvement) — which would close the occlusion-PATTERN sub-lever and complete the augmentation-axis map. Either way the result is throughput-neutral and interpretable.
