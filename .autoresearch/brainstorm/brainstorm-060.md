# Brainstorm EXP-060
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Pervasive Label Errors in Test Sets (Northcutt, Athalye, Mueller — NeurIPS 2021)** (https://arxiv.org/abs/2103.14749; https://github.com/cleanlab/label-errors)
  CIFAR-10 is among the CLEANEST audited benchmarks (~0.54% label-error rate; cf. CIFAR-100 ~5.85%, QuickDraw ~10%). Assuming a similar train-set rate, only ~250–500 of 50,000 train labels are wrong — this CAPS the upside of any label-noise-removal candidate at well under the project's +0.3pp one-draw detection threshold (and LS 0.1 already bounds the wrong-label gradient pull). Used here as an effect-size SCREEN, killing an otherwise-novel candidate.
- **Confident Learning (Northcutt, Jiang, Chuang — arXiv 1911.00068; cleanlab)** (https://l7.curtisnorthcutt.com/confident-learning)
  Methodology for detecting label errors via predicted-probability statistics; gains from retraining without detected errors on CIFAR-10 are small at low noise rates — consistent with the screen above.
- **CutMix (Yun et al. — ICCV 2019, arXiv 1905.04899)** (known reference; distill to knowledge/ in plan phase if chosen)
  Occlusion-with-SIGNAL: the cut region is filled with the same-position patch from another training image and labels mix proportionally to area. On CIFAR ResNet/PyramidNet at fixed epochs (crop+flip baselines), CutMix beats Cutout/erasing by +0.5–1.0pp — the largest published raw effect among legal constructions never tested on this recipe in ANY form. Project's mixup datum (EXP-009, −0.46) was STACKED as a 4th regularizer, not substituted.
- **Smith et al. 2018 batch-as-decay — REFUTED in-project** (knowledge/papers/dont-decay-lr-increase-batch.md, EXP-059)
  Closed the last documented frontier corner (noise schedule); this brainstorm therefore operates beyond the catalogued axis list.

## Experimental History Review

- **Current best**: 96.71 @ 1990397 (EXP-006 recipe); family mean ≈ 96.57, σ ≈ 0.16 (EXP-027); bar = 96.81 = mean+1.5σ; TRUE effects need ≥ +0.3pp for one-draw detection. 60 experiments, 6 improvements (all ≤ EXP-006), 53 consecutive closures since.
- **The documented frontier is EMPTY as of EXP-059** (reports/exp-report-059.md § Next Steps): heat (both directions), noise (level AND schedule), loss geometry (4 pathways incl. fc-WD bracket EXP-057/058), numerics (bf16/fp16/max-autotune), structure (width lattice {64,128,256} hardware-closed, depth-width plane, shortcuts, attention, pooling, head routing, block order), data order/coverage, tail distribution (two-sided pressure law), averaging (weight- AND function-space), regularization DOSE (both sides), per-layer constants, init (both directions), optimizer geometry, step-time engineering (99.3% kernel floor) — all measured-closed.
- **Standing laws every candidate must pass**: absorption (external transfer 0-for-16 — published gains under weak-aug fixed-epoch regimes absorb under TA+RE + completed anneal); deferral (+1ms ≈ −7 ep ≈ −0.08pp); two-sided tail pressure (nothing — data or parameters — may stop moving before budget end); max-statistic (plateau LEVEL is the metric); gradient-noise optimum (bidirectional); numerics equivalence; epoch-boundary law (epoch ends must not be structurally special).
- **Open seams explicitly left by prior closures**: (a) EXP-009's insight — "substitution (swap RE for mixup) … exists but expected value is low": regularizer TYPE at constant dose was never measured; the dose closure does not formally cover it. (b) Data-quality composition (label-error excision) — never tested, but now effect-size-screened by the Northcutt rate (above). (c) Early-phase aug warmup — not formally closed, but interpolates EXP-031's negative lesson (early-phase distribution advantages are transit-only and decay to zero at the converged plateau).
- **Adjacent negative priors**: EXP-009 mixup STACKED −0.46 (over-regularization at 4 regularizers); EXP-013 any crop-diversity increase = pressure; EXP-051 per-sample confidence suppression −7.8σ (low-p views are load-bearing genuine signal); absorption killed SE (+0.5–1.2 published → 0.00), SAM, whitening, zero-γ, projection shortcuts, pre-act order.

## Candidate Ideas

### 1. CutMix substituted for RandomErasing at matched dose (p=0.5, α=1.0, GPU-side, compile-static masks)
**Summary**: Remove RandomErasing from the CPU transform pipeline and apply CutMix per-batch in the charged loop instead, at the same application probability (p=0.5): with prob 0.5 per step, sample λ ~ Beta(1,1), cut a √(1−λ)-scaled box at a random center, fill it with the same-position patch from a batch permutation, and train on the area-weighted mixed loss `lam*CE(logits, y) + (1−lam)*CE(logits, y[perm])`. Implementation is compile-safe via full-size binary masks built from broadcast coordinate comparisons (no dynamic slicing — static shapes, one graph), ~0.1–0.2ms of elementwise work per step; the second CE reuses the same logits (trivial). The regularizer COUNT stays at 3 (LS + TA + CutMix) and the occlusion dose stays at p=0.5 — this isolates regularizer TYPE as the single variable.

**Reasoning**: The dose-response closure (EXP-003/004/009/013/015) established the recipe sits at the dose PEAK, but every point on that curve used occlusion-with-NOISE (RE's random fill). CutMix's mechanism is qualitatively different: the occluded region carries real image statistics (keeps activations in-domain) and the label supervises the actual mixed content — the model learns to allocate confidence proportional to visible evidence. Published effect on CIFAR ResNets is +0.5–1.0 over Cutout at fixed epochs — the largest raw published effect of ANY untested legal construction; even after the absorption discount (0-for-16), this is the rational falsification probe of the "measured ceiling" hypothesis: if the highest-upside untested move reads family-band, the ceiling conclusion strengthens maximally per run. EXP-009's mixup datum does NOT cover this: mixup was global alpha-blending (ghosting, off-manifold inputs) STACKED as a 4th regularizer on top of RE; CutMix is local patch replacement (in-domain inputs) SUBSTITUTED for RE.

**Sources**: arXiv 1905.04899 (CutMix); EXP-009 insight (substitution left open); EXP-003 (RE +0.83 — occlusion is the single most valuable regularizer on this recipe, so its TYPE is the highest-leverage type variable); goal-learnings § reg dose-response; absorption law.

**Estimated Effort**: low-medium — ~30 lines in train.py (remove RE from transforms; batched mask CutMix + mixed loss in the loop); compile warmup must cover BOTH branches (CutMix-on and CutMix-off steps); GPU probe to confirm dt unchanged.

**Risk Assessment**: (a) Absorption — TA may already supply the in-domain-diversity signal CutMix adds over RE; published gains are crop+flip-baseline; family-band read likely a priori. (b) Label mixing interacts with LS 0.1 (mixed-and-smoothed targets) — DeiT-class recipes do exactly this; low risk. (c) The stochastic branch (p=0.5 per step) must not recompile — both branches need warmup; mask path is shape-static by construction. (d) Worst case: over-regularization à la EXP-009 (−0.4-class read) — informative either way (closes the type seam from above). No crash/invalid pathway; verdict lands cleanly in a pre-registerable branch.

### 2. Train-set label-error excision via persistent per-sample EMA confidence (bottom ~0.5%, flagged at p=0.5, excluded thereafter)
**Summary**: Wrap the train dataset to also yield sample indices; during the charged loop maintain a per-sample EMA of p_true (one vectorized gather per step, ~zero cost); at p=0.5 of the budget, permanently zero the loss weight of the ~250 samples (0.5%) with the lowest EMA — samples the model persistently fails to fit across ~70 epochs of diverse augmentations are the label-error suspects (noisy labels are memorized late or never).

**Reasoning**: Data-quality composition is the one axis never measured in 60 experiments. Binary per-SAMPLE excision of the extreme persistent tail differs in granularity (sample vs view), timing (one-time flag vs continuous), and dose (0.5% vs all-sample reweighting) from EXP-051's failed continuous suppressor.

**Sources**: arXiv 2103.14749 (Northcutt — ~0.54% CIFAR-10 error rate); arXiv 1911.00068 (confident learning); EXP-051 (adjacent closure — the risk anchor).

**Estimated Effort**: medium (index plumbing through the loader, masked-mean loss in the compiled graph, EMA buffer).

**Risk Assessment**: FAILS THE EFFECT-SIZE SCREEN: with only ~0.5% wrong labels and LS 0.1 already capping the wrong-label pull, the honest upside is ≤ +0.1pp — below the +0.3 one-draw detection threshold, so even a TRUE positive effect would read as noise. EXP-051's −7.8σ shows the downside if the flagged tail is actually hard-genuine signal. Asymmetric risk, sub-detectable reward.

### 3. Augmentation-strength warmup (crop+flip only for p < 0.2, full TA+RE thereafter)
**Summary**: Run the first 20% of the charged budget with light augmentation (crop+flip), switching to the full TA+RE pipeline at p=0.2 — cleaner early gradients during warmup/peak heat, full pressure (per the tail laws) for the remaining 80%.

**Reasoning**: The tail-pressure closures (EXP-025/033) only measured LIGHTENING the END; the START was never dosed. Anti-deferral framing: lighter early aug = faster early learning during the hottest phase.

**Sources**: EXP-025/033 (tail closures — do not formally cover the head); EXP-031 (the damning precedent).

**Estimated Effort**: medium (per-epoch transform switching through persistent workers — needs the EXP-041 shared-memory epoch-tensor pattern; loader stall risk at the switch).

**Risk Assessment**: EXP-031 measured exactly this mechanism shape on the resolution axis: the early-phase advantage is transit-speed only and decays to zero at the converged plateau (max-statistic law). Expected family-band with extra plumbing risk. Weakest mechanism of the three.

## Idea Evaluation

**Effect-size screen first** (goal-learnings: candidates need TRUE ≥ +0.3pp for one-draw detection): Idea 2 is killed here — Northcutt's ~0.54% error rate bounds its upside at ≤ +0.1pp regardless of mechanism truth, and EXP-051 shows the downside tail is severe (−7.8σ class). A candidate whose BEST case is sub-detectable is not runnable under the honest protocol. Idea 3 survives the screen arithmetically (early-dynamics effects can be large) but its mechanism shape — early-phase distribution advantage converting to plateau level — is precisely what EXP-031 measured at zero conversion on the resolution axis, and it carries the most infrastructure risk (mid-run transform switching through persistent workers).

**Idea 1 wins on every remaining criterion.** Evidence: largest published raw effect (+0.5–1.0 on CIFAR ResNets) of any untested legal construction; the project's own occlusion datum (RE +0.83, the single most valuable regularizer in the recipe) says occlusion TYPE is the highest-leverage type variable available. Mechanism: clear and distinct from the closed dose axis — occlusion-with-signal + area-proportional supervision vs occlusion-with-noise; the EXP-009 mixup negative explicitly does not cover it (global blending, stacked 4th regularizer vs local patching, substituted 3rd). Law compliance: dose held at p=0.5 (reg-dose closure respected), dt ≈ unchanged pending probe (deferral law), constant pressure through the last step (tail laws), no eval-side contact (no reward-hacking surface), static-shape masks (single batch shape — the EXP-059 dynamic-shapes pitfall does not even arise). Risk profile: fails gracefully into the family band or an EXP-009-class over-regularization read; both pre-registerable. Strategic value: as the highest-upside untested move, a family-band read is also the strongest single-run CONFIRMATION of the measured-ceiling hypothesis available — informative in both directions.

## Chosen Idea
**Selected**: CutMix substituted for RandomErasing at matched dose (p=0.5, α=1.0, GPU-side, compile-static masks)

**Why this idea**:
It is the only surviving candidate that passes the effect-size screen with a mechanism not already inside a measured closure. The regularization axis was closed in DOSE with every point measured at occlusion-type = noise-fill; CutMix changes the TYPE while holding dose, count, throughput, noise, heat, and numerics constant — a single-variable probe of the last open seam in the recipe's most valuable component (occlusion, +0.83). It simultaneously serves as the maximal falsification probe of the measured-ceiling conclusion: no remaining legal construction has a larger published prior.

**Hypothesis**:
Replacing RandomErasing's noise-fill occlusion with CutMix's signal-fill occlusion + area-proportional label supervision will raise best_test_acc to ≥ 96.81 IF occlusion type (not just dose) carries information under heavy augmentation — i.e., if training the model to ignore out-of-context patches while allocating confidence proportional to visible evidence improves decision boundaries beyond what noise-occlusion robustness provides. If instead the absorption law extends to augmentation TYPE (TA already saturating the in-domain-diversity channel), the run reads in the family band [96.41, 96.73], closing the regularization axis in type as well as dose and materially strengthening the measured-ceiling conclusion. An EXP-009-class read (< 96.41) would instead indicate mixed-label supervision is over-pressure even at constant dose.
