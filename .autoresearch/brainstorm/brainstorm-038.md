# Brainstorm EXP-038
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- No new external searches — the EXP-037 lesson (project-insights, EXP-035/036/037 entry) is that out-of-regime literature evidence is currently the WEAKEST predictor available: three consecutive mechanisms with strong fixed-epoch/light-aug publications returned exact-deficit nulls. The binding screen is now IN-REGIME evidence, and the only in-regime evidence source is this project's own 38-experiment record.
- Standing in-regime sources used below: EXP-029 (knowledge: BN running stats are calibrated parameters; eval sensitivity to normalization constants measured at −10.9 when substituted), EXP-027 (run-level σ ≈ 0.16; plateau eval-to-eval scatter; max harvests ≤0.03 over plateau median), EXP-036 (audit method: hunt UNMEASURED constants rather than new mechanisms).
- PyTorch BatchNorm semantics (framework fact, not transferred evidence): `running = (1−m)·running + m·batch` per step, default m=0.1 — an EMA with ~10-step horizon; `train.py` never sets it (confirmed by grep), so every eval normalizes with constants estimated from roughly the last 10 batches of that epoch.

## Experimental History Review

- 39 experiments, 6 improvements (last EXP-006); baseline 96.71 @ 1990397 (recipe mean ≈96.57, σ ≈0.16); bar 96.81; **32 consecutive misses**, the last three (SAM, LS-dose, SE) being EXACT-deficit nulls that established the heavy-augmentation absorption law and the in-regime-evidence screen.
- **Audit state**: every EXPLICIT constant in train.py is dosed (EXP-036 closed the last). Applying EXP-036's audit method one level deeper — implicit framework defaults that are functionally constants — finds exactly one undosed, zero-dt, eval-relevant dial: **BN momentum (default 0.1)**. EXP-035 froze it on SAM's perturbed passes (engineering); EXP-029 replaced the stats' SOURCE DISTRIBUTION (clean vs augmented — inverted, −10.9); no experiment has dosed the ESTIMATOR SMOOTHNESS on the unchanged augmented stream.
- Screens any candidate must pass: deferral (no learning displaced from heat), dt (numerics-preserving; +1ms ≈ −6 epochs ≈ −0.08), max-statistic (only converged plateau LEVEL pays), noise optimum (don't touch batch/β/aug), absorption (no out-regime module imports), σ (effect must plausibly reach +0.3… or be zero-cost so a null is cheap).
- Mechanism inventory says everything else law-screened: capacity/depth/head/init/shortcut/activation/attention closed; schedule/optimizer/noise bracketed; regularization dose-response peaked both sides; data order/curriculum has no in-regime evidence and time-varying pressure loses both directions; throughput is at the numerics-equivalent floor (EXP-021).

## Candidate Ideas

### 1. BN running-stat momentum 0.1 → 0.02 (smoother normalization constants at every eval)
**Summary**: One constant applied to all 19 BN layers at construction (`m.momentum = 0.02` loop after model build, or pass momentum in `nn.BatchNorm2d` calls). Zero dt (the EMA update is the same fused op), zero VRAM, zero noise/schedule change — execution signatures byte-identical to baseline; ~139 epochs.

**Reasoning**: The metric is max over 139 evals, and every eval's accuracy is computed through BN constants that are a ~10-batch EMA snapshot (m=0.1). EXP-029 measured the eval's extreme sensitivity to these constants (−10.9 when their distribution was changed) — in-regime evidence that the constants are load-bearing. The estimator has variance (10-batch sample of augmented activations) and bias-under-motion (lags the weights during hot phases). At the converged plateau, weights are quasi-static, so a 50-batch horizon (m=0.02) cuts estimator variance ~5× with negligible lag cost — strictly better-estimated constants exactly where the max-statistic harvests. Jensen-type argument: accuracy is locally concave in normalization-constant error, so noise in the constants costs mean accuracy; smoothing recovers that cost. Honest effect-size accounting: the recoverable mean gain is plausibly +0.05–0.15 (sub-bar alone), partially offset by reduced eval scatter trimming the ≤0.03 max-harvest; this is a low-ceiling but ZERO-COST probe — the only law-compliant dial left whose null is free (no deficit currency spent) and which closes the last implicit constant, completing the audit at both explicit and implicit levels.

**Sources**: EXP-029 (goal-learnings BN-constants law); EXP-027 (σ/plateau scatter); train.py grep (momentum never set); PyTorch BN docs semantics.

**Estimated Effort**: low — one-line construction-time loop, standard composite run with baseline watchdog.

**Risk Assessment**: (a) hot-phase lag could depress MID-RUN evals (cosmetic for a max taken at plateau) — divergence guard unaffected (evals stay ≫15%); (b) if plateau-period weight drift is larger than assumed, a 50-batch horizon could lag at the end too — bounded by trying 0.02 (not 0.005); (c) null outcome costs nothing and is itself the audit-closing datum.

### 2. Classifier-head damping (fc zero-init / logit temperature)
**Summary**: Zero-init fc or scale logits ×0.125 (cifar10-fast).

**Reasoning**: Re-surfaced from EXP-035/036/037 next-steps, but stands against the measured init law (EXP-018/019: nothing init-time moves a 139-epoch converged plateau) and the EXP-036 loss-target null plus the absorption law (temperature is a cifar10-fast import calibrated to ~10-epoch light-aug budgets). Expected effect ≈ 0 with no in-regime support; only its zero cost keeps it on the list.

**Sources**: brainstorm-037 Candidate 3 (screened); goal-learnings init entries.

**Estimated Effort**: low.

**Risk Assessment**: Safe failure, but two measured laws predict the null in advance — running it re-measures known optima.

### 3. Stage-3-only width widening 256 → 320 (capacity where FLOPs are cheap, all blocks kept)
**Summary**: widths (64,128,320); the one capacity move EXP-017 explicitly left untried ("width asymmetry, partially rehabilitated").

**Reasoning**: Keeps all 9 blocks (avoids EXP-017's stage-1-removal failure) and respects channel alignment (320 = 64×5). But the dt law prices it at ≈ +2.5ms (FLOPs ×1.19) → deficit ≈ −0.20, and the width-magnitude curve is measured SATURATED at 4× (6× lost), so the marginal capacity gain at stage 3 alone is plausibly < +0.2 gross — net negative by the project's own arithmetic before any absorption discount.

**Sources**: EXP-017 insight; EXP-005/034 hardware law; EXP-001/002/005 width curve.

**Estimated Effort**: low-medium.

**Risk Assessment**: Pays real deficit currency against a sub-screen expected gain; the asymmetric-conversion law (extra capacity must clear its dt cost at plateau LEVEL) predicts a loss.

## Idea Evaluation

Evidence strength: Candidate 1 is the only one with IN-REGIME mechanistic support — EXP-029 measured (at −10.9 magnitude) that eval accuracy routes through exactly the constants this dial controls, and the framework-default 10-batch horizon is objectively a high-variance estimator of them. Candidates 2 and 3 are each predicted null/negative by two or more measured project laws; running them would re-measure known optima at (for 3) real deficit cost. Mechanism clarity: Candidate 1's causal path is short and quantitative (estimator variance → normalization-constant error → concave accuracy penalty at the plateau where the max is taken). Expected impact: honestly low-ceiling (+0.05–0.15 plausible, +0.3 only if BN-stat noise is a larger share of run-level σ than estimated) — but it is the ONLY remaining dial whose cost is exactly zero in every closed currency, making its information/cost ratio unbounded; after three deficit-paying nulls, the correct risk posture is a free probe. Risk: the safest profile available (byte-identical signatures; null closes the implicit-constant audit). Feasibility: one line. Candidate 1 wins; 2 and 3 are recorded as law-screened so future loops skip the re-derivation.

## Chosen Idea
**Selected**: BN running-stat momentum 0.1 → 0.02 (Candidate 1)

**Why this idea**:
It is the last unmeasured constant in the program — implicit rather than explicit, found by the same audit method that produced EXP-036 — and the only law-compliant candidate with in-regime evidence (EXP-029's measured sensitivity of eval accuracy to BN constants). It is free in every closed currency (dt, heat, noise, numerics of the training path), so even its null outcome is pure information: it completes the constant audit at both levels and measures, for the first time, how much of the run-level σ/plateau level is BN-estimator noise.

**Hypothesis**:
Smoothing the BN running-stat EMA from a ~10-batch to a ~50-batch horizon (momentum 0.02) reduces normalization-constant estimation error at the converged plateau, raising the plateau LEVEL — predicting best_test_acc ≥ 96.81 if BN-stat noise costs the mean ≥ +0.25 (the falsifiable strong form). Execution signatures byte-identical to baseline (dt 22.4ms, ~139 epochs, params 4,286,026). Falsified by a clean plateau within the baseline band (96.4–96.7) → BN-estimator noise contributes <0.1 to the level, the implicit-constant audit closes, and mid-run eval shape (lag signature: depressed early/hot-phase evals recovering at plateau) is recorded as the mechanism diagnostic either way.
