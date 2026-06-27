# Experiment Report EXP-033: Augmentation taper — original-ResNet light transform (crop+flip) for the final 12% of budget

- **Date**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-033.md
- **Plan**: plans/plan-033.md
- **Exp-log**: logs/exp-log-033.md
- **Verdict**: **no-improvement** (96.25 vs bar 96.81; baseline 96.71, Δ −0.46 ≈ −2σ — real measured loss)

## Goal
Maximize CIFAR-10 best_test_acc (%) within the fixed 300s charged budget, train.py only. Baseline 96.71 @ 1990397 (mean ≈96.57, σ ≈0.16); bar 96.81.

## Idea & Hypothesis
**Idea**: Interpolate the measured pressure-schedule endpoints: training byte-identical to baseline until 88% of the charged budget, then epochs draw from a light-augmentation loader (RandomCrop(32,4)+flip — the original He-2015 CIFAR recipe; TrivialAugment and RandomErasing dropped) while the cosine anneal completes normally. EXP-025 had measured the upside (+0.35 immediate jump on a fully-clean tail) and the failure mode (zero pressure → overfit collapse); keeping crop+flip was the designed repair. Decision-boundary class per EXP-032's diagnosis.

**Hypothesis**: Eval jump ≥ +0.15 within 2 taper epochs, SUSTAINED (no overfit reversal), riding the completing anneal to a plateau ≥ +0.25 over the baseline mean ⇒ best ≥ 96.81 at unchanged dt/epochs.

## Approach
Four train.py edits (30 insertions / 1 deletion): `AUG_TAPER_FRAC = 0.88`; `light_tf` Compose; `tail_set`/`tail_loader` (identical loader args, lazy persistent workers); per-epoch `epoch_loader` selection on `total_training_time/TIME_BUDGET_S`. Timed step, LR schedule, model, optimizer, eval untouched.

## Execution
Single clean run (gates passed immediately: GPU 0 free, load 6). rc=0, total 472.1s, startup 10.0s, 139 epochs / 13,472 steps, 139 evals, VRAM 1613.0MB, params 4,286,026. Profile: 268 windows, mean 22.3ms, 0 slow — pristine. Taper engaged ~ep 124 (train loss dropped 0.7695→0.5196 as the lighter data arrived). No retries needed.

## Results
**best_test_acc 96.25; final 96.22; final_test_loss 0.1972. −0.46 vs baseline ≈ 2σ below the mean — outside the noise band; the taper actively HURT.**

The trail decomposes cleanly into the two predicted phases plus one unpredicted outcome:
1. **The alignment transient is real and matched prediction**: ep124 95.62 → ep125 96.10 (+0.48) — same genus as EXP-025's +0.35, confirming the train/test distribution gap costs ~0.3–0.5 of MEASURED accuracy at any instant, recoverable by narrowing the gap.
2. **The repair worked as designed against overfit**: test_loss stayed FLAT (~0.197) for all 15 post-taper epochs — no EXP-025 reversal. Crop+flip pressure is sufficient to prevent clean-set overfitting at this capacity.
3. **But learning STOPPED**: the baseline's final 12% gains ~+0.4 accuracy with test_loss falling to ~0.185 — the anneal's endgame, where low-LR steps against the HARD (TA+RE) distribution convert into generalization. On the light distribution those same low-LR steps had nothing left to learn (train loss 0.52 and flat), so the run banked the one-time +0.48 alignment gain but forfeited the ~+0.4 anneal climb AND the heavy-distribution plateau level — netting −0.46.

**Mechanism (the axis's closing statement)**: the heavy augmentation is not just a regularizer that could be relaxed once "training is done" — under a time-keyed anneal, training is NEVER done before the budget ends (EXP-025's lesson), and the final low-LR phase is precisely where the hard distribution does its highest-value work. The pressure schedule is now bracketed at three points: **full pressure → 96.6 plateau; light tail → 96.2 (alignment gain < forfeited endgame); zero tail → collapse (−0.87)**. Monotone in pressure. The static dose-response (peaked at the current recipe) and the schedule axis now agree: maximum sustained pressure to the last step is optimal. 28 consecutive misses; the EXP-025 "+0.35 transient" is now understood as a measurement artifact of distribution alignment, not banked capability — it cannot be harvested without paying more elsewhere.

## Verification
- Condition 1 (best ≥ 96.81): **FAIL** — 96.25. Pre-condition profile PASS (268 win, mean 22.3ms, 0 slow; 139 epochs exact; params/training_seconds/eval-count exact). Single clean run, no contention. Conditions 2–3 informationally pass (472.1s ≤ 600; 139 = 139).
- Trustworthiness: high — the −0.46 is a smooth, mechanism-consistent trajectory (jump → freeze), not an anomaly.
- Verdict basis: clean miss, real deficit → **no-improvement**.

## Key Learning
Under a time-keyed anneal the final low-LR phase is where the HEAVY training distribution does its highest-value work — tapering augmentation banks a one-time alignment jump (+0.48, same genus as EXP-025's +0.35) but freezes learning (loss flat, no overfit) and forfeits the anneal's ~+0.4 endgame, netting −0.46. Pressure schedule is now bracketed monotone (full 96.6 / light 96.2 / zero collapse): maximum sustained pressure to the last step is optimal, and "alignment transients" are measurement artifacts of distribution proximity, not harvestable capability.

## Unexplored Avenues
- **Reverse taper (heavy→heavier late)**: the bracketing suggests pressure-to-the-end is good, but ADDING pressure late was also measured bad statically (mixup −0.46) and the dose-response is peaked — closed by composition of the two axes.
- **Earlier taper (0.95: only ~7 epochs light)**: shrinks both the forfeited endgame and the harvest window proportionally; the freeze mechanism applies at any taper point — expected interpolation toward baseline from below, sub-bar by construction. Closed by mechanism.
- **Gap-narrowing WITHOUT pressure loss** (e.g., test-set-statistics-aware augmentation): the +0.48-instantaneous-gap datum says ~0.5 accuracy is "hidden" by the train/test distribution distance at eval time — but every attempt to collect it (EXP-025, EXP-029, EXP-033) pays more than it earns because collecting requires reducing the very pressure that builds the level. This is now a measured conservation-like pattern, not an open avenue.

## Next Steps
1. **ResNet-26 depth probe with strict early-dt gate** (brainstorm-033 Candidate 2): the last unbracketed capacity direction; ~90s gate cost if dt >24.5ms, one run if not; pure information value. Confidence: low (gain), high (axis closure).
2. **Per-stage width reshaping at constant FLOPs within alignment constraints** (e.g., 96/128/224 vs 64/128/256 — H20-aligned): allocation POSITION is first-class (EXP-017) and only depth-direction reallocation was measured; width-profile reallocation at matched dt is unprobed. Must pass the dt gate ≤23ms. Confidence: low.
3. **Strategic note for next brainstorm**: 28 misses with every recipe/data/eval-side axis measured-closed; remaining open territory is exclusively architecture-shape at matched dt (depth, width profile, stem) — all gate-screenable in ~90s each. Prioritize cheap axis-closing probes over high-effort singles. Confidence: high (as strategy).

## Exit Action Results
(no exit actions defined for this goal)
