# Brainstorm EXP-035
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **SAM: Sharpness-Aware Minimization** (Foret et al., ICLR 2021, arXiv 2010.01412): explicitly minimizes worst-case loss in an ε-ball (ρ≈0.05) via one extra ascent forward-backward per step; on CIFAR-10 WRN-28-10 with augmentation cuts error ~3.5%→~2.7% at FIXED epochs — a converged-plateau LEVEL gain, the only currency the max-statistic pays (distinct from the bracketed SGD-noise axis: SAM is explicit worst-case geometry, EXP-023/024 bracketed implicit stochastic flatness).
- **LookSAM** (Liu et al., CVPR 2022, arXiv 2203.02714): SAM's ascent direction changes slowly — applying the full ascent every k steps (k=5) retains most of the generalization gain at ~1.2× step cost instead of 2×. Direct evidence that PERIODIC sharpness pressure suffices.
- **Implementation note (timm/canonical repos)**: best practice disables BN running-stat updates on the second (perturbed) forward so eval-time normalization constants reflect only unperturbed weights — exactly our NORMALIZATION-CONSTANT preservation law (EXP-029 measured its violation at −10.9).
- All implementable in pure torch (~20 lines), no new packages.

## Experimental History Review

- 36 experiments, 6 improvements (last: EXP-006); baseline 96.71 @ 1990397 (mean ≈96.57, σ ≈0.16); bar 96.81 needs TRUE effect ≥ +0.3. 29 consecutive misses.
- **Every previously open axis is now measured-closed**: recipe constants both directions (EXP-007…015, 023, 024); schedule family/shape/heat (014, 016, 010); optimizer internal + geometry (Muon EXP-028); data (augmentation dose 003/004/009/013, schedule 025/033, resolution 031); eval-side (BN 029, weight averaging 011/032, head 030); capacity (magnitude 001/002/005/007, allocation 017, depth both ways 008/034). EXP-034 added the hardware law: per-block launch cost makes deeper shapes uniformly worse.
- **Laws constraining EXP-035 design**: max-statistic (only plateau LEVEL pays; transit/throughput converts to ZERO — EXP-021's +10 epochs and EXP-031's +46 epochs both converted to nothing); deferral (every +1ms dt ≈ −6 epochs ≈ −0.014×Δep near plateau, measured EXP-026); normalization-constant preservation (EXP-029); gradient-noise optimum is bracketed for IMPLICIT noise — explicit sharpness regularization is a different, unprobed mechanism.
- **Untried gap**: the loss-landscape-geometry class. Nothing in 000–034 ever modified what gradient is descended (beyond optimizer geometry); EXP-032's diagnosis (decision-boundary-limited, calibration fine) is consistent with sharp-minimum boundary brittleness that SAM targets.
- Throughput levers (H2D prefetch overlap, further compile tuning) are measured zero-conversion — discarded as primary candidates.

## Candidate Ideas

### 1. Periodic SAM — full ascent-descent every 5th step (ρ=0.05, BN-protected second pass)
**Summary**: Every 5th training step runs SAM: forward-backward at current weights → normalize gradient, perturb weights by ρ·g/‖g‖ (eager foreach on param.data, compile-safe) → second forward-backward at the perturbed point with BN running-stat updates disabled → restore weights → optimizer.step() with the perturbed-point gradient. Other 4 steps byte-identical to baseline. Expected dt: (4×22.4 + ~46)/5 ≈ 27.1ms → ~115 epochs. All recipe constants unchanged; the time-keyed anneal completes as always.

**Reasoning**: The one unprobed level mechanism with top-venue evidence at this exact task/architecture family. SAM's CIFAR gains (+0.5–0.8 at fixed epochs on WRN with augmentation) are plateau-LEVEL gains — the only thing the max-statistic rewards. LookSAM shows periodic application retains most of the gain, which is what makes the wall-clock arithmetic survivable: k=5 costs ~24 epochs (≈ −0.33 by the linear deficit law) vs full SAM's ~65 (≈ −0.9, unaffordable). The mechanism targets EXP-032's diagnosis directly: flat minima = decision boundaries robust to weight-space perturbation. Honest risk: if SAM's retained gain under TA+RE (already strong implicit regularization) is <0.6, the net misses the bar — but the axis gets a clean measured closure either way, and the failure mode is graceful (a converged, slightly-lower plateau, classifiable on its merits).

**Sources**: arXiv 2010.01412 (SAM), 2203.02714 (LookSAM); goal-learnings GRADIENT-NOISE law (distinct mechanism), EXP-029 normalization law (BN protection), EXP-026 deficit arithmetic; EXP-032 boundary-limited diagnosis.

**Estimated Effort**: medium — ~25 lines in the train loop (SAM branch, foreach perturb/restore, BN momentum toggle), careful compile interaction (in-place .data ops keep param identity → no recompile), standard composite run.

**Risk Assessment**: (a) dt per SAM step >2× (extra eager foreach + sync) → fewer epochs than modeled; watchdog rescaled (contention >33ms; abort if windowed mean >31ms early). (b) compile guard recompile on perturbation — mitigated by in-place .data ops; STARTUP/early watchdog catches a recompile storm as dt chaos. (c) SAM destabilizes the high-LR phase (ascent at lr 0.4) — literature trains SAM from scratch at standard LRs, low risk; NaN guard armed. (d) Worst case: clean run, plateau ~96.2–96.5 → sharpness axis closed with a number.

### 2. Full SAM every step (canonical ρ=0.05)
**Summary**: The literature-exact recipe: every step pays the double pass; dt ~46ms → ~67 epochs.

**Reasoning**: Strongest per-step evidence, but the deficit arithmetic is unaffordable: −65 epochs ≈ −0.9 by the measured linear law vs literature gain ≤ +0.8 under weaker augmentation than ours — modeled net negative before any shrinkage. Only correct if the linear deficit law badly overestimates at large deficits.

**Sources**: arXiv 2010.01412; EXP-026/021 deficit arithmetic.

**Estimated Effort**: low-medium.

**Risk Assessment**: High prior of a −0.3 to −0.9 result whose interpretation is confounded (mechanism vs deficit) — weak information value compared to Candidate 1.

### 3. H2D prefetch overlap (CUDA-stream prefetcher moving transfer off the charged critical path)
**Summary**: Overlap batch N+1's H2D with batch N's compute; honest throughput gain of ~1–1.5ms/step → ~+7 epochs.

**Reasoning**: Real systems optimization, but the conversion law is already measured at this exact magnitude: EXP-021 bought +10 epochs (compile max-autotune) and converted to ZERO; EXP-031's +46 epochs also converted to zero. Throughput is a closed axis at plateau.

**Sources**: EXP-021, EXP-031; max-statistic law.

**Estimated Effort**: medium.

**Risk Assessment**: Near-certain within-noise outcome; also touches the timed-region accounting, which demands extreme care to stay clearly on the honest side — cost/benefit dominated by Candidate 1.

## Idea Evaluation

Candidate 3 is eliminated by measured precedent: its entire upside (epochs) is a currency twice measured to convert to zero at the plateau. Candidate 2 vs 1 is a dose decision on the same new axis: full-dose SAM has the cleanest literature anchor but the wall-clock arithmetic (−65 epochs ≈ −0.9) exceeds even the optimistic gain estimate, making a negative result uninterpretable (deficit? mechanism?). Candidate 1's k=5 dose puts the deficit (−24 epochs ≈ −0.33) inside the range the published gains can plausibly overcome, is directly supported by LookSAM's finding that periodic ascent retains most of the benefit, and produces an interpretable closure either way (dt and epochs near baseline → any plateau delta is the mechanism's). Evidence strength: high for the mechanism class, medium for the k=5 transfer to our regime. Mechanism clarity: high — explicit flat-minima pressure targeting the diagnosed boundary brittleness, orthogonal to every bracketed axis. Risk: graceful. Candidate 1 wins.

## Chosen Idea
**Selected**: Periodic SAM — full ascent-descent every 5th step (ρ=0.05, BN-protected second pass) (Candidate 1)

**Why this idea**:
It opens the single remaining unprobed level-mechanism class (explicit loss-landscape geometry) with top-venue evidence on this exact dataset/architecture family, at a dose whose wall-clock cost (−~24 epochs) sits below the literature gain it must beat — the only SAM variant whose arithmetic is not pre-lost. It composes with the certified recipe (nothing else changes), respects every measured law (normalization constants protected, noise axis untouched on 4/5 steps, anneal completes), and fails gracefully into a measured axis closure.

**Hypothesis**:
Applying SAM's ascent-descent on every 5th step (ρ=0.05, BN stats frozen on the perturbed pass) at dt ≈ 27ms / ~115 epochs converges to a FLATTER minimum whose plateau LEVEL exceeds the baseline mean by more than the ~0.33 epoch-deficit cost — predicting best_test_acc ≥ 96.81. Falsified by: windowed dt >31ms early (cost model wrong, kill); or a clean converged plateau < 96.81 — in particular a plateau at/below the baseline band closes the sharpness axis (implicit SGD+augmentation noise already saturates flatness at this scale).
