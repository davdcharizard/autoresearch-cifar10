# Brainstorm EXP-071
**Created**: 2026-06-10
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **BatchNorm eps** (Ioffe & Szegedy 2015; PyTorch `nn.BatchNorm2d` default `eps=1e-5`): the numerical floor in `y = (x − μ)/sqrt(σ² + eps)·γ + β`. For channels with large activation variance (σ² ≫ eps) it is negligible; for low-variance/near-dead channels a larger eps shrinks the normalized output, a very mild smoothing/regularization. On a well-trained BN net it is the canonical "last static knob" — near-certain null at this budget. Standard "high-eps" values used elsewhere are 1e-3 (TensorFlow default) to 1e-2.
- No new external sources warranted — this loop is plateau-mapping a known-exhausted recipe; the relevant evidence is internal (goal-learnings + project-insights).

## Experimental History Review

Current best = **EXP-054 = 96.45** (`RandomApply([AugMix()], p=0.5)` + GPU Cutout16). Bar = 96.55. **16 consecutive no-improvements since EXP-054.**

The project is comprehensively mapped. High-importance closures (project-insights):
- **Augmentation** (the ONLY lever that ever lifted top-1) — FULLY exhausted on every sub-axis incl. internal mixing-distribution alpha (EXP-069). CPU and GPU delivery both closed.
- **"Convergence-bound, not overfit-bound" (line 68, High)** — ADDING regularizers fails: WD↑ (EXP-005), Mixup (EXP-011), CutMix (EXP-018), dropout (EXP-022), GhostBN (EXP-047) all regressed/nulled. The net is NOT overfitting, so neither adding NOR removing regularization helps.
- **"Polish-vs-top1" (line 61, High)** — weight-averaging, init/WD, optimizer family+grad/objective, LS-retune, GC, PolyLoss, cosine-geometry all move loss/calibration, NOT top-1. A lower test loss is a near-certain false-positive for top-1 here.
- **Classifier-head/readout** — closed from THREE angles: feature-aggregation (EXP-032), scoring-geometry (EXP-039), spatial-pooling-statistic (EXP-070, −9.45pp; also taught that dt-neutral ≠ optimization-safe — the recipe is brittle to logit-scale perturbations).
- **Capacity** (k/depth/realloc), **schedule** (peak-LR/warmup/SGDR/cooldown), **normalization** (GhostBN, BN-momentum↓, clean-BN), **batch** — all closed.
- **BN-estimator axis (line 181)**: "Only **BN eps (untested)** and momentum-UP (near-certain null) remain — effectively closed."
- **Scalar-knob pattern (EXP-067 insight)**: "every scalar/static-knob retune lands −0.2 to −0.6pp, confirming 96.45 is the k=4/300s ceiling."

**Genuinely UNTESTED cells**: BN eps (the explicitly-flagged last static knob), SGD momentum coefficient, Nesterov on/off. All near-null or contraindicated.

## Candidate Ideas

### 1. BatchNorm eps 1e-5 → 1e-3
**Summary**: Add `eps=1e-3` to all four `nn.BatchNorm2d(...)` constructions in train.py (BasicBlock bn1/bn2, the downsample-shortcut BN, and the stem bn1). Single static change, everything else byte-identical to EXP-054. (Note: `_make_layer`/BasicBlock build BNs internally, so this is ~4 call sites or a small helper — all in train.py.)

**Reasoning**: This is the ONE genuinely-untested cell on the BN-estimator axis (goal-learnings line 181 explicitly: "Only BN eps (untested)… remain"). It is OPTIMIZATION-STABLE — unlike the momentum coefficient (which scales the effective LR) and unlike the EXP-070 readout (which perturbed logit scale), eps only nudges the BN denominator and cannot destabilize training. Its failure mode is a clean within-noise NULL (benign), so it completes the BN-estimator axis closure cleanly without risk. Mechanistically, a larger eps mildly dampens low-variance channels — a tiny smoothing that *could* (very low probability) help calibration-limited borderline cases, but almost certainly lands within the ±0.25pp noise band.

**Sources**: goal-learnings line 179-181 (BN-estimator axis, eps untested); BN paper; project-insights line 61/68 (polish/regularizer closures — eps is neither a strong regularizer nor a logit-scale change, so it dodges both failure mechanisms but also offers little upside).

**Estimated Effort**: low (add one kwarg to ~4 BN sites; ~590s run).

**Risk Assessment**: Near-certain within-noise null (most likely 96.2–96.45). No destabilization risk (eps is the safest possible knob — strictly numerical, optimization-neutral, logit-scale-neutral, dt-neutral, params unchanged). Worst case a small −0.1 to −0.2pp if the dampening very mildly under-utilizes low-variance channels. Honest assessment: this is plateau-mapping with near-zero upside — its value is COMPLETING the axis map (a documented untested cell), not a real bid for +0.1pp.

### 2. SGD momentum 0.9 → 0.95
**Summary**: `MOMENTUM = 0.9` → `0.95` (single constant).

**Reasoning**: The one untested optimizer scalar.

**Sources**: train.py L25; goal-learnings line 181 ("momentum-UP near-certain null").

**Estimated Effort**: low.

**Risk Assessment**: **Contraindicated / likely a REGRESSION, not a clean null.** With Nesterov SGD the effective step size scales ~1/(1−m): raising m 0.9→0.95 roughly DOUBLES the integrated effective LR (10→20). The peak-LR axis is closed with 0.2 optimal and 0.3 (1.5×) already −0.45pp (EXP-016); a ~2× effective-LR increase would overshoot the tuned cosine anneal → likely a meaningful regression. Worse failure mode than Idea 1, no upside. Reject.

### 3. Weight decay 1e-4 → 5e-4 — clean re-test on the AugMix recipe
**Summary**: `WEIGHT_DECAY = 1e-4` → `5e-4`, de-confounding EXP-005 (which tested 5e-4 on the OLD pre-AugMix recipe AND was throughput-confounded — only 65 ep).

**Reasoning**: EXP-005's 5e-4 lowered eval loss (0.204→0.196) but its top-1 read was muddied by 65-ep underfit; a clean 91-ep run on the current recipe was never done.

**Sources**: TSV EXP-005; project-insights line 68 (High).

**Estimated Effort**: low.

**Risk Assessment**: **Contraindicated by a High-importance closure.** project-insights line 68: the net is convergence-bound (not overfit-bound), and EVERY add-regularizer move (incl. WD↑ EXP-005) nulled/regressed. EXP-005's loss-down/top1-flat is the textbook polish-vs-top1 signature → a clean re-test would almost certainly reproduce "loss down, top-1 flat" = null. Lower information than Idea 1 (which closes a genuinely-untested axis vs re-confirming a closed one). Reject.

## Idea Evaluation

All three are dt-neutral, wall-safe, single-variable, and — honestly — near-null-to-negative; the project is at its comprehensively-mapped k=4/300s ceiling (16 straight misses; every High-importance axis closed). The decision is purely about which is the most *useful* plateau-mapping probe with the *safest* failure mode, applying EXP-070's hard-won lesson (prefer probes that do NOT perturb logit scale OR effective LR):

- **Idea 2 (momentum)** ≈ doubling the effective LR → contraindicated by the closed LR axis → likely a real regression, not a clean null. Worst pick.
- **Idea 3 (WD↑)** re-tests a direction a High-importance insight already closed (add-regularizer fails on a convergence-bound net) → low information, predictable null.
- **Idea 1 (BN eps)** is the ONLY genuinely-untested cell (explicitly flagged in goal-learnings), is the SAFEST possible knob (numerically-floor-only: optimization-stable, logit-scale-neutral, dt-neutral, param-neutral — cannot destabilize like EXP-070, cannot over-LR like Idea 2, cannot over-regularize like Idea 3), and its clean-null failure mode COMPLETES the BN-estimator axis map.

Given there is no positive-EV experiment remaining (the honest state of a comprehensively-mapped ceiling), the right move under the NEVER-STOP directive is the cleanest, safest axis-closer that fills a documented untested cell — Idea 1. Expected impact is ~nil; its purpose is map-completion, transparently.

## Chosen Idea
**Selected**: Idea 1 — BatchNorm eps 1e-5 → 1e-3

**Why this idea**:
After 16 consecutive misses every lever is closed, and the only genuinely-UNTESTED, optimization-SAFE cell remaining is BN eps — explicitly flagged in goal-learnings as the last untouched BN-estimator knob. It is the safest possible change (numerical-floor-only: it cannot destabilize training like the EXP-070 readout, cannot inflate the effective LR like a momentum change, cannot over-regularize like a WD/dropout change), and its clean within-noise-null failure mode cleanly completes the BN-estimator axis map. It is honestly plateau-mapping, not a real bid for the bar — but it is the correct disciplined NEVER-STOP probe: fill the documented gap with the lowest-risk knob rather than re-test a closed direction or risk a destabilizing change.

**Hypothesis**:
Raising BN eps 1e-5→1e-3 will leave best_test_acc within the ±0.25pp noise band of baseline (most likely 96.2–96.45, NOT ≥ 96.55), confirming that the BN numerical floor is inert on this well-conditioned net and closing the last untested cell of the BN-estimator axis. The (very low probability) upside is that mildly dampening low-variance channels nudges a few borderline test cases and clears the bar; the overwhelmingly likely outcome is a clean null that completes the map and reinforces that 96.45 is the k=4/300s ceiling.
