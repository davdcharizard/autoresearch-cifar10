# Brainstorm EXP-049
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

<!-- Goal, metric, direction, constraints, verification live in the goal file.
     Baseline lives in experiment-indices/improve-cifar10-test-accuracy.tsv (96.22, EXP-012, 6c417a4). Bar = 96.32. -->

## Web Search & Literature Review

No new external search this loop — the relevant literature is already distilled in the knowledge base and the two near-miss reports being combined:
- **Gradient Centralization (Yong et al., ECCV 2020)** (`reports/exp-report-031.md`, `reports/exp-report-030.md`): centralizing weight gradients to per-output-unit zero-mean over fan-in acts as a projected-gradient regularizer of the weight space; reported to improve generalization and training stability across vision nets. On this net it reliably **lowered loss** (0.1894 < baseline 0.195) but left top-1 within noise (96.14–96.21) — a confirmed "polish" lever.
- **Augmentation cooldown / "FixRes"-style distribution-alignment** (`reports/exp-report-033.md`, `reports/exp-report-034.md`): training augmentation shifts the input distribution away from the clean test distribution; disabling strong aug for a short low-LR tail fine-tunes the model toward the clean test distribution. On this net the later/shorter window (0.10, start frac 0.90) produced **96.26 — the project's only ≥baseline result** — by fine-tuning from a higher pre-cooldown base (96.05).

## Experimental History Review

**Current best / baseline**: 96.22% (EXP-012, commit 6c417a4). Bar = 96.32. **41 consecutive no-improvements (EXP-013..048)**; 6 lifetime improvements (all EXP-000..012 era).

**Every accuracy axis is now mapped and closed** (per `goal-learnings` + `project-insights`): capacity (width/depth/realloc), augmentation (strength/policy/mixing/cooldown/border/occlusion-pattern — EXP-048 closed the last sub-lever), LR schedule (peak/floor/shape), regularizer-adds, classifier-head, feature-routing, activations, weight-averaging, optimizer (family/grad/objective), bag-of-tricks, large-batch, throughput→epochs (EXP-045/046 closed), normalization (GhostBN, EXP-047). Two governing walls: the **epoch wall** (compute/sequential-layer adds → underfit) and the **polish-vs-top1 wall** (compute-neutral changes lower loss but not top-1; the net is generalization-bound at fixed capacity).

**The two near-misses being combined here**:
- **EXP-034 — aug cooldown @0.10**: 96.26 (+0.04 vs baseline, within noise). The ONLY above-baseline result. Mechanism CONFIRMED (clean-distribution fine-tuning in the low-LR tail), but its marginal lift over a full-aug cosine tail is only ~+0.04 — too small alone.
- **EXP-031 — throughput-fixed Gradient Centralization**: 96.14 (within noise), loss 0.1894 < 0.195. Throughput-neutral (compiled+hoisted, dt~8ms/91ep). GC improves loss but not top-1 on its own.

**Relevant cautionary result**: EXP-035 (cooldown @0.10 + clean-tail LR reheat) → 96.12, BELOW baseline — the clean tail is sensitive; adding LR motion to it hurt. So a cooldown combination should leave the tail LR schedule untouched (frozen near-zero, as in EXP-034) and combine with an *orthogonal* lever rather than re-tuning the tail itself.

**Untried gap**: no experiment has ever COMBINED two distinct sub-noise levers. The directive explicitly endorses "combining previous near-misses" once single axes are exhausted. This is the untried space.

## Candidate Ideas

### 1. Augmentation cooldown (EXP-034) + Gradient Centralization (EXP-031)
**Summary**: Re-apply BOTH proven throughput-neutral near-misses simultaneously, unchanged: (a) the EXP-034 four-edit cooldown (a second clean CPU transform `train_tf_clean` = full pipeline minus TrivialAugment; an `aug_cooled` flag flipped at the epoch boundary once `total_training_time/TIME_BUDGET_S ≥ 0.90`; `train_set.transform` swapped to the clean transform; Cutout gated behind the flag) with `COOLDOWN_FRAC=0.10` and the tail LR left frozen near-zero (NOT reheated — EXP-035 showed reheating hurts); and (b) the EXP-031 compiled+hoisted Gradient Centralization (`_gc_compiled` over the 23 hoisted conv/fc weight grads, out-of-place, reassigned between `loss.backward()` and `optimizer.step()`). No other recipe/optimizer/schedule/seed/param/compile-mode changes.

**Reasoning**: The two levers act on orthogonal axes — GC regularizes the gradient/weight space throughout training (lowering loss, sharpening the minimum), while the cooldown realigns the input distribution to the clean test set in the final tail. The synergy hypothesis: GC converges the model to a lower-loss / better-conditioned state whose top-1 advantage is *masked* by the aug-train↔clean-test distribution mismatch; the clean cooldown tail removes that mismatch exactly when GC's better-conditioned weights can be fine-tuned to clean-distribution decision boundaries — letting GC's confirmed loss advantage finally surface as top-1. Each is individually noise-level; the bet is that input-distribution fine-tuning (cooldown) and gradient-space regularization (GC) are non-redundant and additive. Both are independently proven throughput-neutral (dt~8ms, ~91 ep) and proven implementable.

**Sources**: `reports/exp-report-034.md` (cooldown impl + 96.26), `reports/exp-report-031.md` (GC compiled impl + loss 0.1894), `reports/exp-report-035.md` (do-not-reheat-tail caution), GC paper notes (`knowledge/papers/`).

**Estimated Effort**: medium — two independent, individually-proven edit sets with no code interaction (cooldown touches the dataloader/transform + Cutout gate; GC touches the backward→step site). No novel mechanism to debug.

**Risk Assessment**: Most likely outcome is no-improvement (~96.2–96.3) — both components are individually sub-noise and may simply not add. Worst case mirrors EXP-035 (a combination lands slightly below baseline if the levers interfere). Implementation risk is low (both proven); the real risk is the EV: the synergy is plausible but speculative, and re-treads two already-closed axes — justified ONLY because the directive explicitly calls for combining near-misses after single axes are exhausted.

### 2. Augmentation cooldown (EXP-034) + label-smoothing OFF in the clean tail
**Summary**: Keep the EXP-034 cooldown, but ALSO drop `LABEL_SMOOTHING` from 0.1 to 0.0 for the clean cooldown phase only (gated on the same `aug_cooled` flag). During the clean tail the loss becomes plain cross-entropy on hard targets.

**Reasoning**: Label smoothing deliberately caps the model's confidence (soft targets) to regularize the *augmented* training regime. In the clean cooldown tail — clean data, low LR — the test set rewards confident, sharp correct predictions. Removing the soft-target ceiling exactly when the model fine-tunes on clean data lets it commit to harder clean-distribution decision boundaries. This is a *same-mechanism-reinforcing* pairing (both cooldown and LS-off target the train↔test alignment in the tail), arguably tighter than candidate 1's orthogonal pairing.

**Sources**: `reports/exp-report-034.md` (cooldown), `reports/exp-report-023.md` (LS-down full-run, null), `reports/exp-report-035.md` (tail-sensitivity caution).

**Estimated Effort**: low — one extra flag-gated branch selecting `label_smoothing=0.0` in the loss during the clean phase, on top of the cooldown edits.

**Risk Assessment**: Higher destabilization risk than candidate 1 — EXP-035 showed the clean tail is sensitive; hard targets + any residual tail LR could overfit the clean tail and regress (as the LR-reheat did). LS-down was null full-run (EXP-023), so the lever is weak; confining it to the tail is the novel angle but also the speculative one.

### 3. Pre-activation residual blocks (radical, throughput-neutral architectural change)
**Summary**: Convert the post-activation BasicBlock (conv→BN→ReLU) to the pre-activation form (BN→ReLU→conv; He et al. 2016, "Identity Mappings in Deep Residual Networks"), keeping params, width, depth, and the projection shortcut identical. A clean, never-tried structural change at fixed capacity.

**Reasoning**: Pre-activation gives a cleaner identity signal path and better-conditioned gradient flow, reported to improve generalization at fixed params — and it is genuinely orthogonal to the (exhausted) cooldown plateau, opening a fresh architectural axis rather than re-treading two closed ones.

**Sources**: He et al. 2016 (identity mappings); `train.py` BasicBlock L65-92.

**Estimated Effort**: medium — restructure BasicBlock forward + the stem/final-BN placement; verify param count and dt unchanged.

**Risk Assessment**: On a shallow ResNet-20 the pre-activation benefit is known to be small (it matters most for very deep nets); likely within noise. Restructuring risks a subtle dt/graph change (must re-verify 8ms/91ep). Lower directive-alignment than 1/2 (not a near-miss combination), but it doesn't re-tread closed axes.

## Idea Evaluation

All three are low-EV by construction — after 41 no-improvements with every axis mapped, no remaining move has a strong shot at +0.1; the task under NEVER STOP is to run the best-available experiment, not to find a guaranteed winner.

**Evidence strength**: Candidate 1 has the strongest provenance — it combines the two best-documented near-misses (the *only* ≥baseline result, EXP-034 96.26; and the best loss-improver, EXP-031 loss 0.1894), both with proven throughput-neutral implementations. Candidate 2 pairs the cooldown with a lever (LS-down) that was *null* full-run — weaker evidence and a documented tail-sensitivity hazard (EXP-035). Candidate 3 has solid general literature but weak evidence of benefit on a net this shallow.

**Mechanism clarity**: Candidate 1's mechanism is the cleanest defensible synergy — GC lowers loss / sharpens the minimum, the cooldown removes the distribution mismatch that masks that gain as top-1; the levers are orthogonal and individually validated. Candidate 2's mechanism is plausible but collides with the EXP-035 tail-sensitivity result. Candidate 3's mechanism is real but its magnitude on ResNet-20 is small.

**Risk profile**: Candidate 1 is the safest (both components proven throughput-neutral and individually ≥baseline-or-noise; no tail-LR motion). Candidate 2 carries the EXP-035-style regression risk. Candidate 3 carries dt/graph re-verification risk.

**Directive alignment**: The standing directive explicitly says, when single axes are exhausted, "try combining previous near-misses." Candidate 1 is the canonical reading (both are documented near-misses); candidate 2 is a weaker combination (one component is a null, not a near-miss); candidate 3 is a different branch ("radical architectural change") and is the natural fallback if the combination route is exhausted next loop.

**Conclusion**: Candidate 1 (cooldown + GC) is the lead — strongest near-miss provenance, cleanest orthogonal-synergy mechanism, lowest implementation/stability risk, and the most faithful execution of the directive's "combine near-misses." Candidate 3 is the strongest fallback for the following loop (fresh axis); candidate 2 is held as an alternate.

## Chosen Idea
**Selected**: Candidate 1 — Augmentation cooldown (EXP-034 @0.10) + Gradient Centralization (EXP-031), applied together unchanged.

**Why this idea**:
It combines the project's only above-baseline result (EXP-034 cooldown, 96.26) with its best loss-improver (EXP-031 GC, loss 0.1894), both proven throughput-neutral and individually implementable, on orthogonal axes (input-distribution alignment × gradient-space regularization). It directly executes the standing directive's instruction to combine near-misses once single axes are exhausted, with the lowest implementation and stability risk of the candidates (no tail-LR motion, per the EXP-035 caution), and the clearest articulable synergy mechanism (the cooldown's clean tail lets GC's confirmed-but-masked loss advantage surface as top-1).

**Hypothesis**:
Running unchanged at dt~8ms / ~91 epochs (throughput-neutral, no epoch confound), the combination IF the two sub-noise levers add — GC's lower-loss/better-conditioned state realized as top-1 through the clean cooldown tail — lifts `best_test_acc ≥ 96.32` (clearing the bar). Falsifiable: if the result lands within ±0.25pp of 96.26 (i.e. ~96.0–96.5 but below 96.32), the levers do not add usefully and the "combine near-misses" route is itself a plateau — closing this combination and pointing the next loop to the radical-architectural branch (candidate 3).
