# Brainstorm EXP-008
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new search this loop — the decision rests on in-project measurements plus already-distilled sources:
- **WRN (arXiv 1605.07146)** (knowledge/README.md): the central external evidence for this loop — width substitutes for depth on CIFAR-10. WRN-16-8 (16 layers) reaches ~95.7% with only standard crop/flip augmentation, within 0.3pp of the much deeper WRN-28-10; the paper's thesis is that at CIFAR scale, shallow-wide ≥ deep-thin at matched or lower compute.
- **mixup (arXiv 1710.09412)** (standard knowledge, cited in exp-report-006/007 Unexplored Avenues): +0.5–1pp on CIFAR-10 WRNs at full 200-epoch schedules; interacts with label smoothing and existing aggressive augmentation.
- **goal-learnings § Failed Approaches (High, count 3)**: uniform width increases are closed — every attempt below ~70 epochs lost. New protocol rule from EXP-007: capacity steps must project ≥70 epochs from *measured compiled* dt, not extrapolated exponents.

## Experimental History Review

- **Trajectory**: 91.97 → 93.16 (one-cycle recipe) → 95.23 (4x) → 94.41 (8x fail) → 96.06 (RE) → 96.23 (TA) → 95.12 (5x fail) → 96.71 (compile) → 96.00 (6x fail). Baseline: **96.71 @ 1990397** (4x ResNet-20, compiled TA+RE recipe, 139 epochs, dt 22ms).
- **Patterns**: time-keyed one-cycle, compile (1.22x, pure profit), RE (+0.83), TA (+0.17) all validated and kept. Width scales steeply when epochs stay above the floor (+2.07pp at 4x/114ep). Compiled dt scales ~linearly-or-worse in FLOPs across width (EXP-007) — projections must use measured-regime dt.
- **Failed Approaches (High, count 3)**: capacity without throughput — 8x@40ep, 5x@52ep, aligned+compiled 6x@55ep. Uniform widening of ResNet-20 is conclusively closed: no WIDTH_MULT > 4 keeps epochs ≥ 70.
- **Key structural fact**: the budget is TIME, not FLOPs — capacity is only affordable if it comes at (near-)constant time. Depth is the one dimension not yet traded.
- **Untried gaps**: wider-but-shallower topology (exp-report-007 § Unexplored Avenues), Mixup/CutMix at 139 epochs, EMA weights for eval, compile max-autotune (sub-bar standalone), non-uniform stage widths.

## Candidate Ideas

### 1. Wider-but-shallower: NUM_BLOCKS 3 → 2 with WIDTH_MULT 6 (ResNet-14, 96/192/384)
**Summary**: Trade depth for width at roughly constant time: 2 blocks per stage (14 layers) at 6x width. ~6.15M params (1.43x of baseline's 4.29M), ~1.46x FLOPs. Projected dt ≈ 22ms × 1.46 × 1.15 (EXP-007's measured 6x-width inefficiency factor) ≈ 36–40ms → **~80–85 epochs, above the ~70 floor**. All channels (96/192/384) 32-aligned. Everything else byte-identical.

**Reasoning**: This is the only capacity path not foreclosed by the count-3 epoch-starvation failure: it adds width (the dimension with the steep measured payoff, +2.07pp at 4x) while *removing* depth FLOPs so the epoch count stays above the floor. WRN provides direct external evidence that 14–16-layer wide nets match much deeper ones on CIFAR-10. The new EXP-007 protocol rule is satisfied by construction: the projection uses the measured compiled 6x-width dt inefficiency, and an early dt check (abort if projected epochs < ~65 at step ~100) guards the residual estimate risk.

**Sources**: arXiv 1605.07146 Table 5 (WRN-16-8); reports/exp-report-007.md § Unexplored Avenues + Key Learning; goal-learnings § Failed Approaches (High); project-insights § High (alignment) + Medium (regime-specific dt).

**Estimated Effort**: trivial (two constants: NUM_BLOCKS, WIDTH_MULT)

**Risk Assessment**: Two genuine risks: (a) dt lands worse than projected → starvation again — mitigated by the early-abort guard, and the failure is clean/no-improvement; (b) depth 14 itself costs accuracy beyond what width recovers — WRN says no at this scale, but our 32x32 ResNet variant with pad-shortcuts is not byte-identical to WRN blocks. VRAM ~2GB, wall clock ~450s — both comfortable.

### 2. Mixup (α = 0.2) on the 139-epoch compiled 4x recipe
**Summary**: Batch-level mixup in the training step: `lam ~ Beta(0.2, 0.2)`, mixed inputs, loss = lam·CE(y_a) + (1−lam)·CE(y_b); keep TA+RE+LS unchanged.

**Reasoning**: The strongest remaining *regularization* lever, re-promoted by EXP-006's epoch unlock (139 epochs is closer to published schedules than the 114 when TA gave only +0.17). Published +0.5–1pp on CIFAR-10 WRNs. But it stacks a fourth regularizer (after LS, RE, TA) on a 4.29M-param net, against the in-project diminishing trend — and 139 epochs is still 0.7x of published 200-epoch schedules where mixup needs long training to pay off.

**Sources**: arXiv 1710.09412; reports/exp-report-004.md (diminishing stacking); exp-report-007.md § Next Steps.

**Estimated Effort**: low-medium (loss-path change — more bug surface than constants)

**Risk Assessment**: Moderate chance of sub-noise or negative at current capacity/epochs; failure is clean but ambiguous (over-regularization vs undertraining vs capacity). Zero throughput cost.

### 3. EMA weight averaging for evaluation
**Summary**: Maintain an exponential moving average of weights (decay ~0.99–0.999, updated each step on GPU) and pass the EMA model to `evaluator.evaluate()` once per epoch.

**Reasoning**: Orthogonal to capacity and augmentation; standard in modern recipes; helps most when late-training weights oscillate. Counterpoint: our one-cycle LR anneals to ~0, which already implicitly averages late weights — EMA gains in cosine-to-zero schedules are typically small (+0.0–0.2pp). The per-step EMA update (~4.3M-param lerp) adds ~0.5–1ms/step inside the timed budget, costing ~3–6 epochs.

**Sources**: standard practice (PyTorch ImageNet recipes); no in-project datapoint.

**Estimated Effort**: low (model copy + per-step lerp + eval-call switch)

**Risk Assessment**: Likely sub-bar given the annealed schedule; eval-path change needs care (must evaluate the EMA copy, never compiled); failure clean.

## Idea Evaluation

**Evidence strength**: Idea 1 has both the strongest external evidence (WRN's core result is precisely "shallow-wide wins on CIFAR") and the strongest in-project support (width pays +2.07pp when epochs ≥ ~110; every capacity failure was an epoch-floor failure, not a capacity failure). Idea 2 has good external evidence but at 1.4x our epoch count and without three stacked co-regularizers. Idea 3 has weak expected effect under an annealed-to-zero schedule.

**Mechanism clarity**: Idea 1 — capacity up 1.43x at ~80 epochs, above the measured floor; the mechanism (capacity throttled by time, not memory) is the single most-validated fact in this project. Idea 2 — vicinal risk minimization where regularization is already showing diminishing returns. Idea 3 — weight smoothing already largely provided by the LR schedule.

**Expected impact**: Idea 1: +0.2–0.6pp if depth-14 holds (interpolating the 4x→6x capacity gain at non-starved epochs); −0.3 to −0.7 if depth or the dt estimate betrays. Idea 2: −0.2 to +0.3pp. Idea 3: 0 to +0.2pp.

**Risk profile**: All fail cleanly to no-improvement. Idea 1 carries the early-abort dt guard required by the EXP-007 learning, capping wasted budget. Idea 1's failure would also be maximally informative: it would close capacity entirely and rotate the search to regularization/optimization for good.

**Feasibility**: Idea 1 is a two-constant change on validated infrastructure; Ideas 2–3 touch the loss/eval paths.

## Chosen Idea
**Selected**: Idea 1 — Wider-but-shallower: NUM_BLOCKS 2 + WIDTH_MULT 6 (ResNet-14, 96/192/384)

**Why this idea**:
Capacity remains the highest-payoff dimension (+2.07pp at the last non-starved step) and this is the only capacity move that respects the now-High epoch-floor constraint — it buys 1.43x params for only 1.46x FLOPs by spending the depth budget, projecting ~80–85 epochs from measured-regime dt. WRN directly validates the depth-for-width trade at this scale. A clean failure would conclusively close capacity, which is itself high-value direction information.

**Hypothesis**:
Changing NUM_BLOCKS 3→2 and WIDTH_MULT 4→6 (ResNet-14, stage widths 96/192/384, ~6.15M params) on the compiled TA+RE recipe will raise best_test_acc from 96.71% to ≥96.85%, because capacity gains persist when epochs stay above the ~70 starvation floor and the projected ~80–85 epochs (dt ~36–40ms from EXP-007's measured 6x-width compiled scaling) clear it; predicted execution signatures are dt ≤ 42ms at steady state and epochs in [72, 95], with dt > 47ms at step ~100 (projecting < 65 epochs) as the early-abort trigger.
