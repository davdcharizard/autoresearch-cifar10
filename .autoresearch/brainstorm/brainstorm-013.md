# Brainstorm EXP-013
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new search this loop — remaining candidates are settled by in-project evidence plus already-known recipe lore:
- **Reflection padding for RandomCrop**: fast.ai's DAWNBench CIFAR-10 entries and several speedrun recipes pad crops with reflected image content instead of zeros; the mechanism (crops near borders see natural image statistics instead of black bands, so the network wastes no capacity modeling padding artifacts) is a data-quality fix, not added regularization pressure. Folklore magnitude ~+0.1–0.3pp on CIFAR-scale nets. Implemented as a single argument: `transforms.RandomCrop(32, padding=4, padding_mode="reflect")` (torchvision docs).
- **In-project**: EXP-012 (reports/exp-report-012.md) closed the throughput/batch axis (linear scaling metric-neutral); EXP-011 closed smoothing-class eval tricks (max-statistic insight).
- **Integrity note**: test-time augmentation (wrapping the model's forward with flip-averaging before handing it to the frozen Eval) was considered and discarded — it is an eval-compute trick rather than a training improvement (gray-zone for the goal's intent) AND infeasible anyway: doubling each of ~139 evals adds ~118s of wall clock, busting the 600s cap.

## Experimental History Review

- **Trajectory**: 91.97 → … → **96.71 (EXP-006)** → 96.00(f) → 95.76(f) → 96.25(f) → 96.14(f) → 96.46(f) → 96.66(f). Seven consecutive misses; baseline 96.71 @ 1990397.
- **Closed axes** (goal-learnings § Failed Approaches): capacity (High, count 3 + ResNet-14), regularization STACKING (dose-response crossed zero at mixup), peak LR upward at fixed batch (EXP-010), EMA/smoothing for eval (EXP-011, max-statistic), batch scaling with linear LR (EXP-012, metric-neutral — "throughput only converts when the optimizer can use the extra steps at unchanged hyperparameters").
- **Untried gaps**: WARMUP_FRAC (0.15, never moved), WEIGHT_DECAY (5e-4, never moved), crop padding mode (zero padding inherited from the 2016 recipe — never examined), batch 1024 at sub-linear LR (flagged in exp-report-012 § Unexplored Avenues with ≤ ~0.2pp headroom).
- **Key structural fact**: every recent failure perturbed the optimization/capacity equilibrium. The levers that WON after EXP-001 (RandomErasing +0.83, TA +0.17, compile +0.48) were orthogonal additions that left the equilibrium untouched. Remaining ideas should compose with the optimum, not fight it.

## Candidate Ideas

### 1. Reflection padding for RandomCrop (`padding_mode="reflect"`)
**Summary**: One-argument change to the existing crop augmentation: pad-4 with reflected image content instead of zeros before the random 32x32 crop. Everything else byte-identical.

**Reasoning**: The crop transform is the one recipe component still in its 2016 form. Zero padding means up to 4-pixel black bands on ~50% of training crops — statistics the test set never exhibits; the network spends capacity becoming invariant to an artifact. Reflection keeps local image statistics natural, raising signal per epoch at ZERO throughput/VRAM cost (CPU-side, absorbed by the 8 workers exactly like TA was in EXP-004). Crucially this is NOT regularization stacking (the saturated axis): augmentation pressure is unchanged — the same crops happen, only the fill content improves. It composes with the local optimum instead of perturbing the optimization equilibrium, the property shared by every post-EXP-001 win. Risk is simply that the effect is folklore-sized (~+0.1–0.3) and the bar is +0.1 — a coin flip on magnitude, but the cleanest coin available.

**Sources**: torchvision RandomCrop docs (padding_mode argument); fast.ai DAWNBench CIFAR recipes (reflection padding standard); reports/exp-report-004.md (CPU-side transform cost absorbed by workers); goal-learnings § Patterns (orthogonal-addition wins).

**Estimated Effort**: trivial (one argument)

**Risk Assessment**: Expected +0.0–0.3pp; zero throughput/stability risk; clean noise-level failure mode; closes the last untouched augmentation sub-axis either way.

### 2. Batch 1024 @ PEAK_LR 0.57 (sqrt scaling — salvage the +12 epochs)
**Summary**: Keep EXP-012's validated throughput gain but anneal from a sqrt-scaled peak (0.4 × √2 ≈ 0.57) to reduce the hot-phase trajectory damage.

**Reasoning**: EXP-012 proved the hardware delivers +12 epochs at 1024 and that LR 0.8 damages the trajectory by exactly that much. If the damage is LR-curvature-driven, 0.57 keeps most of the epochs at less cost. But two priors cut against it: at 1024 the run takes HALF the optimizer steps, and sqrt scaling means cooler per-example noise than baseline — the deficit may be step-count-driven, in which case no LR fixes it; and this is the third consecutive probe in the batch/LR family after two failures (EXP-010, EXP-012), with exp-report-012 itself estimating ≤ ~0.2pp headroom. Wall clock also runs ~560s — the tightest margin of any config.

**Sources**: reports/exp-report-012.md § Unexplored Avenues; goal-learnings § Failed Approaches (EXP-010, EXP-012 entries).

**Estimated Effort**: trivial (two constants)

**Risk Assessment**: Expected −0.2 to +0.2pp; clean failure; poor family prior (0-for-2); tight wall-clock margin.

### 3. WEIGHT_DECAY 5e-4 → 2.5e-4 (first move of the last untouched base constant)
**Summary**: Halve explicit weight decay on conv/linear weights — the only hyperparameter never revisited since EXP-000's 2016-inherited value.

**Reasoning**: EXP-009 showed total regularization sits at the saturation edge; explicit WD shares that budget with LS+TA+RE, so reducing it could relieve the same over-constraint mixup exposed. But the symmetric reading (EXP-006 converged healthily ⇒ 5e-4 is fine) is equally supported, WD couples to the LR schedule (a single-value probe is weakly informative), and unlike Idea 1 this PERTURBS the equilibrium rather than composing with it.

**Sources**: goal-learnings § Patterns High (dose-response entry names "PEAK_LR/WD never revisited"); reports/exp-report-009.md.

**Estimated Effort**: trivial (one constant)

**Risk Assessment**: Expected ±0.2pp, direction unknown; clean failure; moderate information value.

## Idea Evaluation

**Evidence strength**: Idea 1 carries standard-practice weight (fast.ai DAWNBench recipes) plus a clean in-project analogue (TA's CPU cost absorption, EXP-004); Idea 2 has in-project evidence AGAINST its family (0-for-2, self-estimated ≤0.2pp headroom); Idea 3 is a coin flip with no directional evidence.

**Mechanism clarity**: Idea 1 — removes a train/test distribution mismatch (black-band artifacts) at zero cost: better signal per epoch, the same mechanism class as RandomErasing's win, without adding pressure. Idea 2 — epoch salvage with an unresolved confound (LR-driven vs step-count-driven deficit). Idea 3 — total-regularization rebalance, direction unknown.

**Expected impact**: Idea 1: +0.0–0.3pp. Idea 2: −0.2–+0.2pp with poor prior. Idea 3: ±0.2pp.

**Risk profile**: Idea 1 is the only candidate with strictly-positive mechanism risk (worst case: effect within noise) and zero system risk — it cannot slow, destabilize, or starve the run. Ideas 2–3 both perturb the measured optimization equilibrium, the failure mechanism of five of the last seven loops.

**Feasibility**: All trivial. Idea 1 dominates: best mechanism, only candidate that composes with the optimum, zero cost.

## Chosen Idea
**Selected**: Idea 1 — Reflection padding for RandomCrop

**Why this idea**:
It is the last untouched component of the 2016 recipe, it improves data quality without adding regularization pressure (sidestepping the saturated axis), it costs nothing in throughput/VRAM/stability (CPU-side, worker-absorbed like TA), and it has the orthogonal-composition property shared by every successful experiment since EXP-001 — while every equilibrium-perturbing lever has now failed seven times.

**Hypothesis**:
Changing `transforms.RandomCrop(32, padding=4)` to `padding_mode="reflect"` will raise best_test_acc from 96.71% to ≥ 96.81%, because border crops stop exhibiting zero-band statistics absent from the test distribution, converting wasted invariance capacity into signal — at byte-identical throughput; predicted execution signatures are dt 22ms / ~139 epochs / VRAM ~1613MB (all identical to baseline), epoch-1 eval in the normal ~30–35% band, and a converged final≈best tail.
