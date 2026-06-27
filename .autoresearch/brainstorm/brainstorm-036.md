# Brainstorm EXP-036
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **cifar10-fast (Page)** (knowledge/README.md References, github.com/davidcpage/cifar10-fast): uses **label smoothing 0.2** in its budget-matched one-cycle large-batch recipe — the closest in-domain anchor to our regime (CIFAR-10, wide net, one-cycle, batch 512, mixed precision).
- **CIFAR-10 airbench (Jordan)** (knowledge/README.md References, arXiv 2404.00498): the current speedrun record recipe also uses **label smoothing 0.2** on a wide net with derandomized flip + cutout-class augmentation — second independent in-domain anchor at the same value.
- **When Does Label Smoothing Help? (Müller, Kornblith, Hinton, NeurIPS 2019)**: LS improves test accuracy across vision tasks but the dose-response is task-dependent; LS interacts with the rest of the regularization budget — under heavier augmentation the optimum dose shifts. No universal 0.1.
- No new searches needed: both project anchors and the canonical paper are already in the knowledge base/context.

## Experimental History Review

- 37 experiments, 6 improvements (last EXP-006); baseline 96.71 @ 1990397 (mean ≈96.57, σ ≈0.16); bar 96.81; **30 consecutive misses**.
- **Audit finding — the "recipe space closed" claim has a measured gap**: the local-optimum law (goal-learnings Patterns, EXP-007…024) closed every PROBED constant, but the probe list (009 mixup, 010 peak LR, 012 batch, 013 reflect, 014 warmup, 015 WD, 023/024 momentum) never included LABEL_SMOOTHING. LS=0.1 entered at EXP-000 inside a bundle and has ridden along unmeasured for 36 experiments. The TSV confirms: no LS-dose row exists.
- LS is not data-pressure (the bracketed axis covers data diversity: TA/RE/mixup/reflect) and not heat/noise (signatures are byte-identical at any LS) — it is LOSS-TARGET smoothing, a separate dial whose only measured neighbor is the EXP-032 finding that calibration is fine and accuracy is decision-boundary-limited. LS dose directly shapes logit margins (target max prob 1−ε), i.e., the boundary-relevant training pressure.
- All other axes measured-closed (recipe, schedule, optimizer, sharpness implicit+explicit, data, eval-side, capacity incl. depth EXP-034). Throughput converts to zero. Per-step-cost mechanisms pay deficit. A zero-dt, signature-identical, single-constant level probe is the only move that risks NOTHING in the closed currencies.
- σ discipline: single-constant probes need the TRUE effect ≥ +0.3 to clear the bar; both in-domain anchors sitting at 2× our dose make "0.1 is materially low" a live possibility worth one run.

## Candidate Ideas

### 1. LABEL_SMOOTHING 0.1 → 0.2 (single-constant dose probe toward the in-domain anchors)
**Summary**: One-line change. Everything else byte-identical: dt 22.4ms, ~139 epochs, identical signatures — a pure plateau-LEVEL read at full epoch count.

**Reasoning**: The last unmeasured recipe constant, and the only one whose unmeasured value DISAGREES with both in-domain anchors (cifar10-fast and airbench both use 0.2 on this dataset with wide nets and budget-matched schedules). Mechanism: LS dose sets the target logit margin (max-prob target 1−ε); EXP-032 diagnosed the ceiling as decision-boundary-limited, and margin pressure is the most direct loss-side lever on boundaries. Doubling LS also strengthens the implicit logit-norm regularization that pairs with heavy augmentation in both anchor recipes. Risk-free in every closed currency: zero dt, zero heat change, zero noise change, zero VRAM. If it misses, 0.1 is confirmed at-or-above optimal from one side and the 0.05/0 direction inherits a measured prior.

**Sources**: knowledge README rows for cifar10-fast + airbench; Müller et al. 2019; goal-learnings Patterns (local-optimum entry's probe list — the gap); EXP-032 boundary diagnosis.

**Estimated Effort**: low — one constant, standard composite run.

**Risk Assessment**: (a) Effect could be small (±0.1–0.2, within noise) — outcome still closes the constant; (b) over-smoothing can compress inter-class logit gaps and HURT top-1 — that is precisely the falsifiable read; (c) no signature risk at all — the cleanest instrument available.

### 2. LABEL_SMOOTHING 0.1 → 0 (remove smoothing entirely)
**Summary**: Opposite end of the same dial: hard targets, maximal margin pressure per example.

**Reasoning**: Some strong WRN+TA recipes use no LS, and Müller et al. note LS can suppress logit-distance information. But both in-domain budget-matched anchors point the OTHER way, and hard targets under heavy augmentation (noisy effective labels from aggressive TA crops) risk fitting label noise. Better as the bracketing follow-up if 0.2 reads NEGATIVE (then 0.1 sits between two worse points and the constant closes; if 0.2 reads positive-but-sub-bar, interpolation logic applies instead).

**Sources**: Müller et al. 2019; WRN reference recipes.

**Estimated Effort**: low.

**Risk Assessment**: Weaker anchors than Candidate 1; running it FIRST wastes the better-evidenced direction.

### 3. Margin-shaping loss (logit-norm regularization or complement-class penalty)
**Summary**: Replace/augment CE with an explicit margin objective (e.g., penalize ‖logits‖ or non-target logit mass directly), 1–2 lines, dt-free.

**Reasoning**: Also targets the boundary-limited diagnosis, but with no in-domain anchor at this budget, more researcher degrees of freedom (form + coefficient), and overlap with what LS dose already parameterizes — LS IS a complement-class mass control. Test the cheap, anchored dial (Candidate 1) before the unanchored generalization.

**Sources**: EXP-032 diagnosis; LogitNorm (Wei et al., ICML 2022 — calibration/OOD-focused, weak accuracy evidence).

**Estimated Effort**: low-medium.

**Risk Assessment**: Coefficient tuning under σ=0.16 noise invites a multi-run fishing expedition; dominated by Candidate 1 for the first probe on this axis.

## Idea Evaluation

All three candidates live on the loss-target axis — the one dial never measured. Candidate 3 is the general form but unanchored and tunable (bad under our noise floor); it only becomes interesting if LS dose shows the axis is live. Candidates 1 and 2 are the two ends of the same constant; evidence strength decides order: TWO independent in-domain anchors (same dataset, wide nets, budget-matched one-cycle, heavy aug) sit at 0.2, none at 0. Candidate 1 is also the better information geometry: a positive read confirms the anchors and may clear the bar outright; a negative read brackets 0.1 from above, making 0 (Candidate 2) the one remaining point. Mechanism clarity: high (margin-target dose on a boundary-limited model); risk: the safest of any experiment yet run — byte-identical execution signatures, zero cost in every closed currency. Candidate 1 wins.

## Chosen Idea
**Selected**: LABEL_SMOOTHING 0.1 → 0.2 (Candidate 1)

**Why this idea**:
It is the last unmeasured recipe constant — an audit gap in the "recipe space closed" law — and the only constant whose current value disagrees with both in-domain anchor recipes (cifar10-fast and airbench, both 0.2 on CIFAR-10 wide nets under budget-matched schedules). It is loss-side margin pressure aimed exactly at the EXP-032 boundary-limited diagnosis, and it is free in every measured currency (dt, heat, noise, VRAM, epochs), making it the cleanest pure-level instrument available after 30 misses.

**Hypothesis**:
Doubling the smoothing dose to the anchor value raises the converged plateau LEVEL (stronger margin/logit-norm regularization composing with the certified augmentation budget) with execution signatures byte-identical to baseline (dt 22.4ms, ~139 epochs) — predicting best_test_acc ≥ 96.81. Falsified by a clean converged plateau within or below the baseline band (96.4–96.7), which brackets LS=0.1 as at-or-above optimal from one side; a clearly NEGATIVE read (<96.4) closes the dial entirely (0.1 between two worse points by interpolation with the 0-end's known overconfidence risk).
