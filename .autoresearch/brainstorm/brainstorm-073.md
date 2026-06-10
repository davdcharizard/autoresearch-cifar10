# Brainstorm EXP-073
**Created**: 2026-06-10
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- No new external sources warranted. This loop is plateau-mapping a comprehensively-exhausted recipe (73 experiments, 19 consecutive misses since EXP-054). The relevant evidence is entirely internal (goal-learnings + project-insights). The one remaining untested axis (the SGD momentum/Nesterov internals) is well-understood from first principles (Sutskever et al. 2013 on Nesterov momentum; the effective-step ≈ 1/(1−m) scaling) — no literature gap to fill.

## Experimental History Review

Current best = **EXP-054 = 96.45** (`RandomApply([AugMix()], p=0.5)` + GPU Cutout16). Bar = 96.55. **19 consecutive no-improvements since EXP-054.** The project is exhaustively mapped (73 experiments, 8 improvements all ≤ EXP-054).

**The ONLY lever that ever lifted top-1 — augmentation diversity — is now CLOSED from every angle**: policy family (TA/RA/AA/AugMix, EXP-012/014/060), chain-count/width (EXP-055), magnitude/severity (EXP-053), mix-alpha (EXP-069), coverage (EXP-055/057), delivery path (CPU/GPU, EXP-056/057/059), AND op-set composition (EXP-072: all_ops=False → 96.43, −0.02pp virtual-tie with improved loss — the photometric ops are not load-bearing). EXP-072's −0.02pp near-tie is the strongest evidence yet that 96.45 is a robust optimum.

**Every non-augmentation axis is closed**: capacity (k up/down EXP-004/009/058, depth EXP-044, realloc EXP-038), schedule (peak-LR EXP-016/017, warmup EXP-062, SGDR EXP-029, cooldown EXP-033/034/063), normalization (GhostBN EXP-047, BN-momentum EXP-067, clean-BN EXP-061, BN-eps EXP-071 — all 3 BN-estimator angles), readout/head (EXP-032/039/070), weight-averaging (EMA/SWA/Lookahead EXP-006/019/020/068), optimizer FAMILY (AdamW EXP-043) + dynamics (SAM EXP-036, GC EXP-030/031, grad-clip EXP-064 — gradient/weight-dynamics polish family explicitly closed, project-insights line 61), regularizers (WD/Mixup/CutMix/dropout/LS — convergence-bound not overfit-bound), batch (EXP-025/050), resolution (EXP-066), LayerScale/zero-γ (EXP-051/026). The net is **generalization-bound at fixed capacity** (project-insights line 85).

**Scalar-knob pattern (High)**: every static/scalar retune lands −0.2..−0.6pp (EXP-062/064/065/067/071). EXP-072 op-set was the rare near-tie (−0.02pp).

**Genuinely UNTESTED cells (the entire remaining frontier)**: the SGD optimizer-internal knobs — **Nesterov on/off** and the **momentum coefficient (0.9)**. Both near-null-to-contraindicated. Nothing with positive expected value remains.

## Candidate Ideas

### 1. Nesterov momentum ON → OFF (vanilla heavy-ball SGD)
**Summary**: `optim.SGD(..., nesterov=True)` → `nesterov=False` (vanilla momentum), keeping momentum=0.9, everything else byte-identical to EXP-054. Single-flag change. CPU/GPU-neutral, throughput-neutral (Nesterov vs heavy-ball is the same kernel cost), param-neutral.

**Reasoning**: This is the SAFEST genuinely-untested cell — the last untouched optimizer boolean. Nesterov applies the gradient at the look-ahead point (θ + m·v) rather than at θ; on a well-conditioned BN+warmup recipe it gives a slightly more responsive update with marginally better convergence, which is WHY it is the tuned default (validated since EXP-000). Turning it OFF tests how much Nesterov's look-ahead contributes vs plain heavy-ball. It cannot destabilize training (it does NOT change the effective step magnitude ~1/(1−m), only the point of gradient evaluation — unlike a momentum-coefficient change), and its failure mode is a clean within-noise null or small regression. Its value is COMPLETING the optimizer-internal axis map (the last untested optimizer knob) with the safest possible failure mode.

**Sources**: train.py L200-206 (`nesterov=True`); goal-learnings EXP-043 (SGD+Nesterov beats AdamW — Nesterov is part of the tuned winner); project-insights line 61 (optimizer-dynamics polish closed); Sutskever et al. 2013 (Nesterov momentum in deep nets).

**Estimated Effort**: low (one flag; ~590s run).

**Risk Assessment**: Cannot destabilize (same effective LR; only the gradient-eval point shifts — no logit-scale/LR perturbation, unlike EXP-070/071). Near-certain within-noise null OR small −0.1..−0.3pp regression (removing the tuned look-ahead). Honest assessment: plateau-mapping with ~nil upside — its purpose is closing the last untested optimizer cell, transparently, not a real bid for the bar.

### 2. SGD momentum coefficient 0.9 → 0.95
**Summary**: `MOMENTUM = 0.9` → `0.95` (single constant), Nesterov kept on.

**Reasoning**: The one untested optimizer SCALAR.

**Sources**: train.py L25; goal-learnings line 184 ("momentum-UP near-certain null/regression").

**Estimated Effort**: low.

**Risk Assessment**: **Contraindicated / likely a REGRESSION, not a clean null.** With Nesterov the effective step ≈ 1/(1−m): raising m 0.9→0.95 roughly DOUBLES the integrated effective LR (10→20). The peak-LR axis is CLOSED with 0.2 optimal and 0.3 (1.5×) already −0.45pp (EXP-016); a ~2× effective-LR increase would overshoot the finely-tuned cosine anneal → likely a meaningful regression (worse failure mode than Idea 1, no upside). Reject as the lead.

### 3. SGD momentum coefficient 0.9 → 0.85 (lower)
**Summary**: `MOMENTUM = 0.9` → `0.85` — the symmetric untested cell to Idea 2 (lower effective LR).

**Reasoning**: If 0.9 is slightly past the momentum optimum, 0.85 (≈0.67× effective LR) could marginally help; mirrors the LR-LOWER probe (EXP-017).

**Sources**: train.py L25; EXP-017 (peak-LR 0.15 lower → −0.64pp).

**Estimated Effort**: low.

**Risk Assessment**: Near-certain regression. Lower momentum ≈ lower effective LR, and the LR-LOWER direction is closed (EXP-017 0.15 → −0.64pp, the largest schedule regression). 0.85 would under-drive the cosine anneal → likely −0.2..−0.5pp. Low information (re-confirms the closed LR-lower direction via the momentum proxy).

## Idea Evaluation

The honest state: the project is at a comprehensively-mapped ceiling (19 straight misses; the only top-1 lever, augmentation, is now exhausted including op-set EXP-072; every other axis closed). There is NO positive-EV experiment remaining — all three candidates are near-null-to-negative. The decision is purely which plateau-mapping probe is the most USEFUL with the SAFEST failure mode (applying the EXP-070/071 lesson: prefer probes that do NOT perturb effective-LR or logit-scale):

- **Idea 2 (momentum↑0.95)** ≈ doubles effective LR → contraindicated by the closed LR axis → likely a real regression, not a clean null. Worst failure mode.
- **Idea 3 (momentum↓0.85)** ≈ lowers effective LR → the closed LR-LOWER direction (EXP-017) → near-certain regression, low information.
- **Idea 1 (Nesterov off)** is the ONLY remaining cell that does NOT change the effective-LR magnitude (it shifts only the gradient-evaluation point), so it is the SAFEST possible probe — it cannot overshoot/undershoot the tuned anneal like the momentum-coefficient changes, cannot destabilize like the EXP-070/071 architectural probes, and its clean-null failure mode cleanly closes the last untested optimizer cell (the optimizer-boolean axis).

Idea 1 wins on safety and information value (closes a genuinely-untested cell vs re-confirming the closed LR axis via a momentum proxy).

## Chosen Idea
**Selected**: Idea 1 — Nesterov momentum ON → OFF (vanilla heavy-ball SGD)

**Why this idea**:
After 19 straight misses the augmentation lever (the only one that ever lifted top-1) is exhausted from every angle and every other axis is closed; no positive-EV experiment remains. The only genuinely-untested, optimization-SAFE cell is the Nesterov flag — the last untouched optimizer boolean. Unlike the momentum-coefficient changes (Ideas 2/3, which scale the effective LR into the closed LR axis and likely regress) and unlike the EXP-070/071 architectural probes (which perturbed logit-scale/numerics), toggling Nesterov off changes ONLY the gradient-evaluation point at fixed effective step — it cannot destabilize training, and its clean within-noise-null failure mode cleanly completes the optimizer-internal axis map. It is the correct disciplined NEVER-STOP probe: fill the documented last gap with the lowest-risk knob, transparently as plateau-mapping rather than a real bid for the bar.

**Hypothesis**:
Turning Nesterov off leaves best_test_acc within the ±0.25pp noise band of 96.45 (most likely 96.2–96.45, NOT ≥ 96.55), confirming that on this well-conditioned BN+warmup recipe the Nesterov look-ahead contributes only a marginal convergence refinement (consistent with it being the tuned default that beat AdamW, EXP-043), and closing the last untested optimizer cell. The overwhelmingly likely outcome is a clean null/small regression that reinforces 96.45 as the robust k=4/300s ceiling; any movement would itself be informative about how load-bearing the look-ahead update is here.
