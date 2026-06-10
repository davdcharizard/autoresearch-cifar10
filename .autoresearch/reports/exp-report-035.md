# Report EXP-035: Clean-tail LR reheat (aug cooldown @0.10 + re-annealed LR 0.02→0 on the clean phase)

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-035.md
- **Plan**: plans/plan-035.md
- **Log**: logs/exp-log-035.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher-is-better) within the fixed 300s budget, editing only `train.py`. Baseline = **96.22%** (EXP-012, commit 6c417a4); pass bar = **96.32%**.

## Idea & Hypothesis
Combine-near-misses refinement of EXP-034. EXP-034's augmentation cooldown (disable TA+Cutout for the final 10%) reached 96.26 but its marginal lift over a full-aug tail was only ~+0.04. Diagnosed cause: the cooldown fires at frac 0.91 where the global cosine LR has annealed to ~0.001–0.005, so the clean-data fine-tune is **LR-starved** — the model can't take meaningful steps toward the clean/test distribution. Hypothesis: giving the clean phase its own re-annealed LR (cosine `CLEAN_LR0=0.02 → 0`) supplies real adaptation budget, lifting best_test_acc above 96.26 toward the 96.32 bar, throughput-neutral.

## Approach
Single new variable on top of the proven EXP-034 config. Re-applied the EXP-034 cooldown scaffold (2nd CPU transform `train_tf_clean` = pipeline minus TrivialAugment; `aug_cooled` flag + epoch-boundary `train_set.transform` swap with an observable marker; Cutout gated behind the flag) at `COOLDOWN_FRAC = 0.10`, and ADDED the LR override: when `aug_cooled`, `lr = CLEAN_LR0 * 0.5 * (1 + cos(pi * clean_progress))` with `clean_progress` running 0→1 over the final 10% (`CLEAN_LR0 = 0.02`, grounded in EXP-020's best SWA floor). Smoke test confirmed params 4,299,866, the reheat LR profile (~0.0195 at frac 0.91 vs global cosine 0.0044, annealing to 0), AST clean, diff = train.py only.

## Execution
Single run, exit 0 in 403.5s wall (300s training). Cooldown+reheat fired once at `ep 83 frac 0.91` (identical point to EXP-034); clean-phase step LR confirmed reheated (`lr: 0.0145 → 0.0125 → ...` decaying, vs ~0.004 frozen). dt settled to 8ms (early ~9ms in compile-warmup), 91 epochs, no NaN/errors.

## Results

- **Primary metric**: best_test_acc = **96.12%** @ ep88 (baseline 96.22, delta **−0.10pp**; −0.20pp vs bar; −0.14pp vs EXP-034's 96.26). Bar NOT cleared.
- **The reheat MECHANISM worked directionally, but the run was confounded and net-negative**:
  - *Pre-cooldown base was ~0.25pp LOWER than EXP-034* (95.80 @ ep81 vs 96.05). The augmented phase shares EXP-034's code path, so this is run-to-run throughput-jitter variance: early dt ran ~9ms before settling to 8ms, and the time-fraction LR schedule diverges when accumulated dt differs (documented High-Importance protocol noise). This handicapped the absolute final number.
  - *The reheat produced a LARGER clean-tail climb than EXP-034.* Clean tail: ep83 95.62 → ep85 96.06 → **ep88 96.12 (peak)** → ep91 96.11. That is +0.32 over the pre-cooldown best (95.80), exceeding EXP-034's frozen-LR tail climb of +0.21 (96.05→96.26). So more clean-phase LR did drive more clean-distribution adaptation — the LR-starvation hypothesis is directionally supported.
  - *But the reheat did NOT settle the loss.* final_test_loss stayed **0.2003** (vs baseline 0.195 and EXP-034's 0.1951), and the tail bounced in a ~0.2pp band (96.06/95.92/95.96/96.12/96.02/96.07/96.11). The extra LR kept the model moving in a higher-loss region instead of annealing into a sharp minimum — it bought top-1 movement at the cost of convergence.
- **Analysis**: Two effects net out unfavorably. (1) The reheat's bigger top-1 climb is real but is partly *recovering* from a noise-lowered base, not adding above EXP-034's ceiling — the peak 96.12 < EXP-034's 96.26 despite the bigger climb, because the base was lower AND the loss never settled. (2) The non-settling loss is the same signature as the closed weight-averaging / polish cluster but inverted: where EMA/SWA/GC settle loss without moving top-1, the reheat moves top-1 in the tail without settling loss — neither converts to a durable bar-clearing gain. The single-run base-noise confound (±0.25pp) is itself larger than any plausible reheat benefit, which is the decisive practical problem: the augmentation-schedule axis cannot deliver a reliable +0.1 when one run's augmented-phase base varies by more than that. Fits the firmly-established generalization-bound-at-fixed-capacity plateau.
- **Key Learning**: Giving the augmentation-cooldown's clean tail a re-annealed LR (0.02→0) instead of the frozen cosine tail produces a LARGER tail-climb (+0.32 vs EXP-034's +0.21) — confirming the clean fine-tune was LR-starved — but it does not settle the loss (0.200) and the result is dominated by ±0.25pp augmented-phase base noise, landing 96.12 (below baseline). The cooldown axis matches the plateau under both frozen and reheated tails; it does not break it.

## Verification

- **Conditions**: Cond 1 (≥96.32) **FAILED** — 96.12 < 96.32 (−0.10 vs baseline). Cond 2 (clean, <600s, 0 errors) passed (403.5s). Cond 3 (only train.py; params 4,299,866; eval-count 91 == epochs; core torch; seed 42) passed.
- **Review Notes**: Trustworthy as a fair test of the reheat mechanism — cooldown+reheat fired once at the correct point (ep83/frac0.91), clean-phase LR confirmed reheated, throughput-neutral (91 ep, dt 8ms). Intervention is on the augmentation schedule + LR-within-the-clean-phase (intended class), not a measurement-gap exploit. Caveat noted: the absolute number is confounded by a ~0.25pp-lower augmented-phase base than EXP-034 (run-to-run noise).
- **Verdict**: no-improvement
- **Verdict Basis**: Valid, trustworthy result; primary condition (clear the bar) failed by 0.20pp and the result is below baseline; no constraint violated.

## Unexplored Avenues
- **Lower CLEAN_LR0 (~0.01) for the clean tail** — the 0.02 reheat moved top-1 but didn't settle loss; a 0.01 reheat might get part of the climb while still annealing the loss. LOW confidence it clears the bar (the base-noise confound, ±0.25pp, still dominates any reheat effect), but it would bracket the reheat-strength optimum.
- **Re-run EXP-035 at 0.02 unchanged** — purely to get a luckier (≥96.05) augmented-phase base; the +0.32 reheat climb from a 96.05 base would project to ~96.37. This is essentially seed/throughput-lottery, NOT a genuine algorithmic gain (borders on the no-seed-hacking constraint), so not a legitimate path to a durable improvement.
- The drop-only-TA-keep-Cutout cooldown variant (brainstorm-034) remains untried but is now very low-value.

## Next Steps
- **Close the augmentation-SCHEDULE axis.** Across EXP-033 (0.15→96.10), EXP-034 (0.10→96.26, frozen tail), EXP-035 (0.10 + 0.02 reheat→96.12), the cooldown matches the 96.22 plateau under every window and tail-LR variant and never clears +0.1; the per-run augmented-phase base noise (±0.25pp) exceeds any cooldown benefit. Confidence: high that this axis is exhausted.
- **Treat 96.22 as the confirmed k=4/300s ceiling** (~27 axes now mapped: capacity, all augmentation, full LR-schedule incl. this clean-phase reheat, all regularizers, architecture, optimizer/GC, weight-averaging, batch). Confidence: high the plateau is real.
- **If continuing, the only remaining moves are radical and low-confidence**: e.g. a cheap-capacity asymmetric architecture (extra/wider layer3 blocks at 8×8 to dodge the uniform-widening compute wall) — flagged LOW confidence because the epoch wall is monotone and goal-learnings warns against capacity adds at this budget, but it is the one structural lever not yet tested in a FLOP-targeted form. Confidence: low.

## Exit Action Results
<!-- No exit actions defined for this goal. -->
