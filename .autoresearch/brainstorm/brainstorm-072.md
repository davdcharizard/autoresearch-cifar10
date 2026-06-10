# Brainstorm EXP-072
**Created**: 2026-06-10
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **AugMix (Hendrycks et al., ICLR 2020)** — torchvision `transforms.AugMix` source (verified in-env, tv 0.24.1): the op pool has TWO tiers. The base 9 ops are geometric + lossless-photometric: ShearX/Y, TranslateX/Y, Rotate, Posterize, Solarize, AutoContrast, Equalize. When `all_ops=True` (torchvision's DEFAULT) it ADDS 4 magnitude-photometric ops: **Brightness, Color, Contrast, Sharpness** (each linspace(0,0.9)). The ORIGINAL AugMix paper deliberately EXCLUDES exactly these 4 (and the overlap with ImageNet-C corruption types is its stated reason — a test-set-integrity concern specific to the *robustness* benchmark, NOT to clean accuracy). KEY IMPLICATION: EXP-054's `transforms.AugMix()` silently runs `all_ops=True` (all 13 ops) — the photometric ops ARE active. Whether those 4 color/brightness ops help or hurt CLEAN CIFAR-10 top-1 has never been isolated here.
- No new external source needed beyond the verified torchvision source + the project's own augmentation map; this is an internal op-set-composition probe of the one proven lever.

## Experimental History Review

Current best = **EXP-054 = 96.45** (`RandomApply([AugMix()], p=0.5)` + GPU Cutout16). Bar = 96.55. **18 consecutive no-improvements since EXP-054.** The project is comprehensively mapped (72 experiments).

**The ONLY lever that has EVER lifted top-1**: augmentation DIVERSITY — Cutout (EXP-002/003 +1.1pp cumulative), TrivialAugment (EXP-012 +0.22), AugMix multi-chain (EXP-052 +0.12, EXP-054 +0.11). Every non-augmentation axis is closed:
- Optimizer (SGD-mods, AdamW EXP-043, SAM EXP-036, Lookahead EXP-068, GC EXP-030/031, grad-clip EXP-064), schedule (peak-LR EXP-016/017, warmup EXP-062, SGDR EXP-029, cooldown EXP-033/034/063), capacity (k up/down EXP-004/009/058, depth EXP-044, realloc EXP-038), normalization (GhostBN EXP-047, BN-momentum EXP-067, clean-BN EXP-061, BN-eps EXP-071), readout/head (EXP-032/039/070), weight-averaging (EMA/SWA/Lookahead), regularizers (WD/Mixup/CutMix/dropout/LS — convergence-bound not overfit-bound), batch (EXP-025/050), resolution (EXP-066). **Scalar-knob pattern**: every static retune lands −0.2..−0.6pp.

**AugMix sub-levers already mapped**: chain-COUNT w2→w3 (EXP-052/054, w3 best; w4 EXP-055 hurts), magnitude/severity 3→6 (EXP-053 hurts), mix-distribution alpha 1→2 (EXP-069 hurts), coverage p (35% EXP-055 hurts, 50% EXP-054 best, 100% EXP-057 hurts — true interior optimum). Policy FAMILY: TrivialAugment (EXP-012, 96.22), RandAugment (EXP-014, 96.19), AutoAugment-CIFAR10 (EXP-060, 96.22) all < AugMix-p0.5 (96.45).

**GENUINELY UNTESTED AugMix cell**: the **op-SET composition** (`all_ops` flag). Every prior AugMix experiment varied how-many-chains / how-strong / how-often, but never WHICH OPS are in the menu. Notably, the three policy families that include color/photometric ops by default (TA, RA, AA) all tied at ~96.22, BELOW AugMix's 96.45 — leaving open whether AugMix's edge is its mixing despite (or because of) its photometric ops.

## Candidate Ideas

### 1. AugMix `all_ops=False` — drop the 4 photometric ops (geometric-only AugMix)
**Summary**: Change EXP-054's `transforms.AugMix()` → `transforms.AugMix(all_ops=False)`, restricting the AugMix op pool from 13 ops to the original-paper 9 (ShearX/Y, TranslateX/Y, Rotate, Posterize, Solarize, AutoContrast, Equalize) — i.e., REMOVE Brightness, Color, Contrast, Sharpness. Single-variable; everything else byte-identical to EXP-054 (w3, chain_depth -1, severity 3, alpha 1.0, p=0.5, GPU Cutout16, cosine peak0.2/Nesterov/WD1e-4/LS0.1, batch128, seed42, compile reduce-overhead). CPU-delivered → FREE w.r.t. the Σdt=300s budget, throughput-/param-neutral (the op pool is sampled inside the dataloader workers).

**Reasoning**: This is the one genuinely-untested dimension of the only lever that works (augmentation op-set composition). Two competing mechanisms make it a REAL bidirectional probe, not a foregone null: (a) the 4 photometric ops (Color/Brightness/Contrast at severity 3) distort class-discriminative COLOR cues on CIFAR-10 (where hue can separate classes), so removing them could yield a cleaner, more label-preserving diversity → a top-1 gain; (b) conversely they add diversity, and diversity is the proven lever, so removing them could mildly regress. The circumstantial evidence leans toward (a) being worth testing: TA/RandAugment/AutoAugment all INCLUDE color ops and all underperformed AugMix here — consistent with photometric ops being net-unhelpful on this net. Geometric-only AugMix isolates the mixing+geometric-diversity benefit from the photometric distortions.

**Sources**: torchvision AugMix source (verified in-env); AugMix paper (op-exclusion rationale); TSV EXP-012/014/060 (color-op policies all 96.22 < 96.45), EXP-052/053/054/055/069 (AugMix sub-lever map); goal-learnings augmentation entries.

**Estimated Effort**: low (one kwarg; ~590s run).

**Risk Assessment**: Cannot destabilize (CPU op-menu change only — no logit-scale/effective-LR/graph perturbation, unlike EXP-070/071 architectural probes; the safest failure mode). Worst case a mild −0.1..−0.3pp if the removed ops were net-beneficial diversity. Most-likely outcomes: small + (photometric ops were mildly harmful) or small − (they were mildly helpful); a real shot at the bar only if color-op removal helps more than the diversity loss costs.

### 2. AugMix coverage p=0.5 → 0.6 (single scalar)
**Summary**: `RandomApply([AugMix()], p=0.5)` → `p=0.6`. More images get AugMix, fewer stay clean.

**Reasoning**: Coverage is the proven-most-sensitive AugMix knob (50% beat both 35% and 100%). 0.6 is an untested interior point between the 50% optimum and the 100% over-regularized failure.

**Sources**: TSV EXP-054 (50%), EXP-055 (35% hurts), EXP-057 (100% hurts).

**Estimated Effort**: low.

**Risk Assessment**: Near-certain scalar-knob null/regression. Coverage is already bracketed with 50% as a clear interior optimum (35% AND 100% both worse) → 0.6 almost certainly lands −0.1..−0.3pp (the convergence-bound net under-trains as effective aug rises toward the EXP-057 failure). Low information — re-confirms a near-settled axis rather than opening a new one.

### 3. AugMix severity 3 → 2 (lower per-op magnitude)
**Summary**: `transforms.AugMix(severity=2)` — the symmetric untested cell to EXP-053 (severity 6, hurt).

**Reasoning**: EXP-053 only probed HIGHER magnitude (6, regressed). Lower (2) is untested; if default 3 is slightly past the per-op-strength optimum, 2 could help marginally.

**Sources**: TSV EXP-053 (severity 6 hurts, "magnitude interior-optimal at default 3").

**Estimated Effort**: low.

**Risk Assessment**: Near-certain null. EXP-053 already concluded "magnitude is interior-optimal at default 3"; severity 2 is a small step off a declared optimum → likely −0.1..−0.2pp, a scalar-knob axis-filler with little upside.

## Idea Evaluation

All three are CPU-delivered (free w.r.t. Σdt), single-variable, wall-safe, throughput-/param-neutral, and on the proven augmentation lever — none can destabilize training (the safe-failure class, unlike the EXP-070/071 architectural probes). The decision is which is the most USEFUL probe with a genuine (not foregone) upside:

- **Idea 2 (coverage 0.6)** and **Idea 3 (severity 2)** are both scalar retunes off ALREADY-BRACKETED interior optima (coverage 50% beat 35%&100%; severity 3 beat 6). The project's hard-won "every scalar/static-knob retune lands −0.2..−0.6pp" pattern applies squarely → near-certain nulls that re-confirm settled axes. Low information.
- **Idea 1 (all_ops=False)** is the ONLY genuinely-untested DIMENSION — op-set composition, orthogonal to all four already-mapped AugMix scalar sub-levers (count/magnitude/alpha/coverage). It has a real bidirectional mechanism (photometric ops may distort CIFAR color cues OR add useful diversity), and the circumstantial evidence (color-op policies TA/RA/AA all underperformed AugMix) makes the "removing color ops helps" hypothesis non-trivial. It is the disciplined NEVER-STOP move: probe the one untested cell of the one lever that has ever worked, rather than re-confirm a bracketed scalar.

Idea 1 wins on mechanism clarity and information value; ties the others on safety and cost.

## Chosen Idea
**Selected**: Idea 1 — AugMix `all_ops=False` (geometric-only AugMix, drop the 4 photometric ops)

**Why this idea**:
After 18 straight misses every axis is closed EXCEPT the augmentation op-SET composition — a genuinely-untested dimension of the only lever (augmentation diversity) that has ever lifted top-1 on this net. EXP-054's `transforms.AugMix()` silently includes 4 photometric ops (Brightness/Color/Contrast/Sharpness) that the original AugMix paper excludes; these distort class-relevant CIFAR-10 color cues, and the three color-op-inclusive policy families (TA/RA/AA) all underperformed AugMix here. Restricting AugMix to its 9 geometric/lossless ops isolates the mixing+geometric diversity from the photometric distortion — a clean, single-variable, Σdt-free, throughput-neutral test that cannot destabilize training. It is the highest-information remaining probe with a real (if uncertain) shot at the bar.

**Hypothesis**:
Setting `all_ops=False` removes the 4 photometric ops and yields a geometric-focused AugMix. PREDICTION: best_test_acc moves within ±0.3pp of 96.45; the upside case (photometric ops were net-harmful to CIFAR-10 color discrimination → cleaner diversity clears 96.55) is a genuine ~25-35% possibility rather than a foregone null, while the most-likely outcome is a small change either direction that maps the op-set-composition axis. A clear improvement would newly establish that AugMix's photometric ops are counterproductive for clean CIFAR-10 accuracy; a regression would confirm the full 13-op diversity is load-bearing.
