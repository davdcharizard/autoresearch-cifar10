# Brainstorm EXP-028
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only. -->

## Web Search & Literature Review

- **Muon optimizer (Keller Jordan, kellerjordan.github.io/posts/muon/; github.com/KellerJordan/cifar10-airbench)**: MomentUm Orthogonalized by Newton-Schulz — orthogonalizes the momentum update of 2D weight matrices via 5 quintic Newton-Schulz iterations (coefficients 3.4445, −4.7750, 2.0315, bf16-stable). **Set the CIFAR-10 94% speed record (3.3 → 2.59 A100-seconds) on airbench — the optimizer was developed ON this exact task class** (small conv net, seconds-scale budget). Also the engine of the NanoGPT speedrun (~1.35× sample-efficiency over AdamW).
- **airbench Muon hyperparameters** (deepwiki.com/KellerJordan/cifar10-airbench/2.2-muon-optimizer, fetched 2026-06-10): Muon on **convolutional weights only** — lr 0.24, momentum 0.6, nesterov; NS-5; head weights and biases/BN on plain SGD. (airbench additionally renormalizes conv weight norms each step — a dual-norm scheme we will NOT import, to bound the variable count.)
- **Mechanism**: orthogonalization equalizes the singular values of the update, preventing the update from being dominated by a few directions — a per-step conditioning improvement that compounds over the whole trajectory. This is sample-efficiency, NOT activation arithmetic — the EXP-026 transfer failure (speedrun activation didn't survive our dt budget) does not automatically apply: Muon's cost is a fixed per-step optimizer overhead (~19 small matmul chains), priceable at the early-dt gate, while its benefit class (fewer steps to a given loss) is exactly what a fixed-time budget monetizes.
- **EXP-027 effect-size screen (new, binding)**: candidates need TRUE effects ≥ +0.25–0.4pp. Muon's speedrun gains (~20–35% sample efficiency) are the only remaining evidence class of that magnitude.

## Experimental History Review

- **Current best**: 96.71 @ 1990397 (EXP-006); replicates put the config's true mean at ≈96.57 with run-level σ ≈ 0.16 (EXP-027) — the bar 96.81 is ≈ +1.5σ above the mean. **Twenty-two recorded non-improvements (007–027).**
- **What the laws leave open**: a mechanism improving PER-STEP LEARNING QUALITY at near-baseline signatures. Recipe constants bracketed (within SGD geometry); activations closed on cost; capacity closed except stage-3 widen (arithmetic below); data/alignment closed. **The optimizer FAMILY has never been changed** — every optimization probe (momentum trades, EMA, schedule shapes) stayed inside SGD's update geometry. Muon is a different geometry, not a different constant.
- **Law screen for Muon**: deferral — orthogonalization acts from step 0, no warm-up structure ✓; epochs — NS cost is real (~285 small bf16 matmuls/step eager, est. +2–4ms → ~120–127 epochs), priced via the early-dt gate with an explicit required-effect calculation ✓ (gate kill >27ms); numerics-equivalence — N/A (deliberate training change, not a same-recipe execution swap), epoch arithmetic handled explicitly; noise — Muon(0.6 momentum, orthogonalized) is a different noise geometry: flagged honestly as outside the SGD-bracketed noise law, this is the experiment's point; max-statistic — gain lands as plateau level if it lands ✓.
- **Effect-size arithmetic**: at +3ms (25.4ms, ~123 epochs) the epoch deficit costs ≈ −0.24pp; Muon must deliver ≥ ~+0.5pp gross to clear the bar. Speedrun-scale sample-efficiency (≥20%) over ~123 epochs is worth far more than that IF it transfers — the only candidate whose evidence magnitude exceeds its cost arithmetic.
- **Capacity check (stage-3 widen, named in exp-report-027)**: stage-3 convs at 320ch cost ×1.56 FLOPs in the stage → ~+18% total → ~26ms → ~120 epochs ≈ −0.29pp deficit; the marginal capacity (one stage 256→320) plausibly returns +0.1–0.2 — NET NEGATIVE by arithmetic under the new screen. Rejected without a run.

## Candidate Ideas

### 1. Muon optimizer for conv weights (airbench-anchored hybrid), 2-point LR design
**Summary**: Implement Muon in train.py (pure torch, ~30 lines: NS-5 orthogonalization + momentum buffer): conv weights (ndim=4, reshaped to 2D) optimized by Muon with the existing time-keyed one-cycle SHAPE scaling a Muon peak LR; fc weight, BN params, biases stay on the baseline SGD path (peak 0.4, momentum 0.9, nesterov, selective WD). Muon group: nesterov momentum 0.6 (airbench), decoupled WD 5e-4 (p.mul_(1 − lr·wd) — coupled WD would be distorted by orthogonalization), update scale ×sqrt(max(1, rows/cols)) per Jordan. Two-point LR design pre-authorized: Run 1 at peak 0.24 (the only measured anchor); if the early trajectory (ep1–10) runs clearly below the baseline family or diverges → Run 2 at peak 0.12. Early-dt gate at 27ms.

**Reasoning**: The last unprobed axis with evidence magnitude ≥ the EXP-027 effect-size requirement. Regime-matched provenance is stronger than any prior import: Muon was not just used on CIFAR speedruns — it was DEVELOPED on them. Its benefit is per-step conditioning (compounds all run), its cost is fixed per-step overhead (gate-priceable in 90 seconds). A converged miss at clean signatures closes the optimizer-geometry axis — the last big-swing class — with one experiment.

**Sources**: kellerjordan.github.io/posts/muon/; deepwiki.com/KellerJordan/cifar10-airbench/2.2-muon-optimizer (hyperparameters, fetched this loop); goal-learnings § Protocol Findings (σ ≈ 0.16, effect-size screen; early-dt gate EXP-008/026); exp-report-027 § Next Steps #1.

**Estimated Effort**: medium — NS function + custom step loop for the Muon group (replace the single optimizer with a hybrid: keep optim.SGD for the non-Muon group, hand-rolled Muon step under no_grad for conv weights), LR plumbing through lr_at().

**Risk Assessment**: (a) dt overhead worst-case kills at the gate (90s, cheap); (b) peak-LR transfer from a 10-epoch dual-norm recipe to a 139-epoch WD recipe is the weakest link — mitigated by the pre-authorized 2-point design and the divergence guard; (c) interaction with bf16 autocast/compile is nil (optimizer runs outside the compiled graph, NS in bf16 is its native design); (d) graceful failure: converged miss closes the axis. VRAM +~17MB (one extra momentum buffer set for convs).

### 2. Squeeze-and-Excitation blocks (gate-screened)
**Summary**: SE module per BasicBlock (global-pool → 2 small FCs → channel scale), reduction 16.

**Reasoning**: Classic +0.2–0.5 on CIFAR WRNs at fixed epochs. But: pointwise multiplies + per-block global reductions cost dt (EXP-026 cost ladder says 5–10% likely → −7–13 epochs ≈ −0.1–0.2), new Kaiming-init FC weights must be learned during peak heat (EXP-020's deferral toll, −0.13 class), and fixed-epoch evidence does not transfer (8 confirmations). Net expected ≈ 0 ± 0.2 — below the effect-size screen.

**Sources**: Hu et al. 2017 (SENet, known context); goal-learnings EXP-020/026 entries.

**Estimated Effort**: medium.

**Risk Assessment**: graceful but predictably noise-band — fails the new screen.

### 3. Stage-3-only width 256→320
**Summary**: Widen the cheapest stage only.

**Reasoning / Rejection**: arithmetic above — ~+18% FLOPs → ≈ −0.29pp epoch deficit vs +0.1–0.2 plausible capacity gain = net negative. The effect-size screen rejects it without spending a run; recorded here to close the "named but untried" thread from exp-report-027.

**Sources**: exp-report-017/027; project-insights (H20 alignment).

**Estimated Effort**: low.

**Risk Assessment**: not run — rejected by arithmetic.

## Idea Evaluation

**Evidence strength**: Muon dominates: developed on this exact task class, holds the CIFAR-10 speed record, and replicated across domains (NanoGPT). SE's evidence is fixed-epoch (the transfer class with 8 documented failures). Stage-3 widen has no affirmative evidence at this delta.

**Mechanism clarity**: Muon — equalized singular values → better-conditioned steps → faster loss descent per step → higher plateau within the fixed budget. SE — channel attention, fixed-epoch benefit. Widen — marginal capacity.

**Expected impact**: Muon is the only candidate whose evidence magnitude (≥20% sample efficiency) exceeds both its dt cost arithmetic AND the EXP-027 effect-size requirement. SE and widen are pre-computed noise-band.

**Risk profile**: All graceful. Muon's LR-transfer risk is bounded by the 2-point design + divergence guard + gate.

**Feasibility**: Muon is ~30 lines of pure torch (no new packages); the hybrid optimizer structure maps cleanly onto the existing two-param-group setup.

## Chosen Idea
**Selected**: Muon optimizer for conv weights (airbench-anchored hybrid, 2-point LR design)

**Why this idea**:
It is the last unprobed big-swing axis — the optimizer family itself — and the only remaining candidate whose evidence class (speedrun-record sample-efficiency, developed on CIFAR-10 conv nets) matches the ≥ +0.3pp true-effect requirement that EXP-027 established. Its cost is measurable in 90 seconds at the early-dt gate; its failure mode is a clean converged miss that closes the axis.

**Hypothesis**:
Muon on the 19 conv weight matrices (NS-5, nesterov 0.6, peak 0.24 on the time-keyed one-cycle, decoupled WD; everything else baseline) improves per-step optimization quality enough to overcome its measured dt overhead: dt ≤ 27ms (gate), epochs ≥ ~115, the mid-run trajectory at matched progress runs AT or ABOVE the baseline family, and the converged plateau forms ≥ +0.25 above the baseline MEAN — **best_test_acc ≥ 96.81 with final-7 median ≥ 96.6**. If Run 1 (peak 0.24) diverges or trails the family badly by ep10, Run 2 at peak 0.12 is pre-authorized; a converged miss on the better of the two closes the optimizer-geometry axis.
