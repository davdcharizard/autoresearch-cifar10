# Brainstorm EXP-025
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only. -->

## Web Search & Literature Review

No new searches; grounding from known high-signal literature and the knowledge base:

- **FixRes — Touvron et al. 2019 (arXiv 1906.06423, NeurIPS)**: train-time augmentation creates a train→test DISTRIBUTION DISCREPANCY; a short final phase aligned to the test distribution (they fine-tune at test-time preprocessing/resolution) recovers significant accuracy on ImageNet at negligible cost. The transferable mechanism: end training on the distribution the evaluator measures.
- **BN-recalibration literature (known context, e.g. AdaBN/BN-stat re-estimation)**: BatchNorm running statistics accumulated under heavy augmentation (RandomCrop-pad, flips, TrivialAugment, RandomErasing) are biased relative to clean test images; re-estimating them on the target distribution is a documented, cheap gain. Our eval path uses `base_model.eval()` with running stats — directly exposed to this bias.
- **knowledge/README (cifar10-fast lineage)**: budget-race recipes keep augmentation light precisely because heavy augmentation has a distribution cost; our recipe's aug stack (TA+RE+crop+flip) is the heaviest part of the EXP-006 recipe and was tuned for full-trajectory pressure, never examined for end-of-training distribution alignment.

## Experimental History Review

- **Current best**: 96.71 @ 1990397 (EXP-006). **Nineteen consecutive misses (EXP-007…024).**
- **RECIPE-SPACE IS CLOSED** (goal-learnings § Patterns): every constant bracketed alone or in compensated trades; EXP-023/024 bracketed the gradient-noise curve (β horizon ×2 → −0.30, ×½ → −0.22 at byte-identical signatures); four laws now bidirectionally evidenced — deferral, numerics equivalence, max-statistic, noise optimum.
- **What was NEVER probed**: time-structure of the DATA DISTRIBUTION. Every augmentation experiment (EXP-003/004/009/013) changed the dose for the WHOLE run — the pressure axis is closed for constant doses. No experiment has changed WHAT distribution the final phase trains on. The train→test discrepancy mechanism (FixRes/BN-stats) is orthogonal to every closed axis.
- **Law screen for this mechanism class**: deferral — touches only the late phase, early heat/epochs untouched ✓; numerics — same kernels, same compiled graph (input tensors identical in shape/layout) ✓; epochs — GPU step work unchanged; loader-side change only, and loader stalls live OUTSIDE dt (EXP-013, t0-after-yield) ✓; noise — partial exposure flagged honestly: removing aug in the tail reduces gradient diversity there, but the noise law was measured on FULL-TRAJECTORY constants at fixed data distribution; the alignment mechanism is distinct and the tail lr→0 makes tail gradient noise second-order.
- **Relevant infra prior**: EXP-013 — ADDING per-image CPU work busts the wall cap via loader stalls; this candidate REMOVES transforms (safe direction). Wall headroom ~70–110s (totals 484–534s vs 600 cap); a second DataLoader's worker startup (a few seconds, outside dt) fits easily.

## Candidate Ideas

### 1. Final-phase clean-data alignment: switch to un-augmented training data at progress ≥ 0.85 (FixRes-style tail)
**Summary**: Build a second `DataLoader` over the same CIFAR-10 train set with the TEST-side transform (ToTensor + Normalize only — exactly what `prepare.py`'s Eval uses, minus nothing else; no crop/flip/TA/RE). At the top of each epoch, if `total_training_time/TIME_BUDGET_S ≥ 0.85`, iterate the clean loader instead of the augmented one. ~18–20 of the final epochs train on the true test-side distribution; weights AND BN running stats adapt to it while the cosine tail anneals lr to ~0.

**Reasoning**: The evaluator measures clean images; the model has never trained on them — every train image is cropped/flipped/TA'd/erased. Two documented mechanisms say the final phase should be aligned: FixRes (fine-tune on test-distribution preprocessing recovers the discrepancy) and BN-stat bias (running stats under heavy aug differ from clean-image statistics; ~20 epochs × 98 steps at momentum 0.1 fully re-converges them). The gain target is the converged plateau LEVEL — exactly what the max-statistic harvests (~15 evals on the aligned plateau). Cost analysis: GPU step work byte-identical (same shapes, same kernels, same compiled graph — no recompile since input signature is unchanged); per-image CPU cost DROPS (fewer transforms); the one-time second-loader spin-up lands outside dt. Overfit risk is bounded: the clean phase runs at lr ≤ ~0.02 (cosine at p≥0.85) with WD+LS still active, and 50k images over ~20 epochs at near-zero lr is fine-tuning, not memorization.

**Sources**: FixRes arXiv 1906.06423; BN-recalibration (AdaBN-class results, known context); infra-errors EXP-013 (loader accounting); goal-learnings § Patterns (max-statistic rewards plateau level/length).

**Estimated Effort**: low-medium — one extra transform/Compose + one extra DataLoader at setup, plus a 2–3 line epoch-level source switch in the `while` loop. No change to the timed step, schedule, optimizer, or eval.

**Risk Assessment**: (a) Distribution-shift transient at the switch — first clean epochs may dip before BN stats re-converge (mid-plateau dip is harmless to the max; only the final level matters). (b) Late pressure-drop reading: EXP-015's pressure-down was a FULL-RUN constant (−0.30); this is a tail-only change with an alignment mechanism EXP-015 lacked — documented as the explicit difference. (c) RAM/CPU: +8 worker processes idle until first use — well within host margins. (d) Hyperparameter (switch point 0.85) is a guess; a miss leaves a one-knob follow-up (0.92). Failure graceful: converged no-improvement.

### 2. Label-smoothing anneal: LS 0.1 → 0 over the final 15% of the budget
**Summary**: The objective-side analog of Idea 1 — keep LS during hot/noisy phases, remove it in the tail so final weights optimize the sharp objective the evaluator's accuracy metric implies (`ls_now = LABEL_SMOOTHING * min(1, (1-progress)/0.15)` passed to F.cross_entropy).

**Reasoning**: Müller et al. 2019 document LS's cost to logit sharpness; the cosine tail is where the plateau forms; one scalar per step, zero execution change.

**Sources**: Müller et al. 2019 (known context); brainstorm-024 idea 3 (deferred there).

**Estimated Effort**: minimal — 2 lines in the timed loop.

**Risk Assessment**: Same late pressure-drop concern as Idea 1 but WITHOUT the distribution-alignment mechanism — accuracy is invariant to logit sharpening unless decision boundaries move, so the expected effect is small; sign uncertain. Strictly weaker mechanism than Idea 1; natural composition partner for a later loop if Idea 1 lands.

### 3. Baseline variance replicate (measurement loop, no intervention)
**Summary**: Re-run the unmodified baseline once to measure run-to-run σ of best_test_acc (seed fixed at 42 but cudnn.benchmark + bf16 atomics make runs non-deterministic) — calibrating interpretation of all ±0.1–0.3 results.

**Reasoning**: Nineteen misses span −0.05…−0.99; the noise floor under them is estimated (eval ±0.1) but never measured at the run level. NOT seed hacking: the constraint bars re-rolling for a better number; this cannot move the baseline by construction (verdict would be no-improvement regardless).

**Sources**: exp-report-024.md § Next Steps #3.

**Estimated Effort**: minimal (no diff at all).

**Risk Assessment**: Zero risk, zero upside on the metric — burns a loop while an unprobed intervention class (Idea 1) is available. Defer until intervention candidates are exhausted or a near-bar result needs adjudication.

## Idea Evaluation

**Evidence strength**: Idea 1 carries the only EXTERNAL evidence class not yet burned by a transfer failure: FixRes is a wall-clock-cheap, end-of-training alignment result whose mechanism (train→test preprocessing discrepancy) maps one-to-one onto our setup (heavy aug train, clean eval, BN running stats); BN-stat bias is textbook. Critically, the four prior transfer failures (Bag of Tricks, RegNet, WRN, airbench-init) all failed via the DEFERRAL law — fixed-epoch evidence demanding early-run resources; FixRes-style alignment spends nothing early. Idea 2's evidence is weaker (calibration ≠ accuracy). Idea 3 has no metric upside.

**Mechanism clarity**: Idea 1: the evaluator measures a distribution the model never trains on; align the final ~20 epochs and the BN stats + weights converge onto the measured distribution → plateau level rises. Clear, two documented sub-mechanisms, and falsifiable (BN-stat effect alone should be visible within 1–2 post-switch epochs). Idea 2: sharper logits → unclear accuracy path. Idea 3: not an intervention.

**Expected impact**: Idea 1: FixRes-class gains on ImageNet are large (1–2pp); here the discrepancy is smaller (no resolution gap, 32×32 both sides) but the aug stack is heavy (TA+RE) — +0.1–0.3pp plausible. Ideas 2–3: ~0.

**Risk profile**: All graceful. Idea 1's transient dip is harmless mid-plateau; its loader change is in the measured-safe direction (CPU work removed).

**Feasibility**: Idea 1 is a contained ~15-line change touching only data-source selection; the timed step, schedule, optimizer, eval, and compiled graph are untouched.

## Chosen Idea
**Selected**: Final-phase clean-data alignment (aug-off tail at progress ≥ 0.85)

**Why this idea**:
It is the first candidate since EXP-006 that operates in a genuinely unprobed intervention class — the time-structure of the training distribution — with external evidence (FixRes, BN recalibration) whose mechanism transfers through, not against, the campaign's four laws: it spends nothing in early heat, nothing in epochs (loader-side only; GPU step identical), nothing in numerics, and its tail-only noise exposure is second-order at lr→0. The gain lands exactly where the max-statistic pays: the converged plateau level.

**Hypothesis**:
Training the final ~15% of the budget on clean (test-distribution) images re-converges BN running stats and fine-tunes weights onto the evaluated distribution: signatures match baseline (dt ~22.4ms, ~139 epochs, VRAM ~1613MB — the clean loader may add a one-time stall outside dt), a transient dip ≤1pp within 1–2 epochs after the switch, then a plateau ABOVE the baseline family's — **best_test_acc ≥ 96.81 with the final-7-evals median ≥ 96.6**. A converged miss kills the alignment mechanism class (FixRes does not transfer to same-resolution CIFAR under this recipe) and routes the campaign to objective shaping (Idea 2) or the variance replicate (Idea 3).
