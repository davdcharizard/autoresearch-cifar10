# Brainstorm EXP-017
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- No new search this loop — the relevant literature was reviewed in brainstorm-016 and is sufficient. Key
  references (already cited): linear-scaling rule / warmup (Goyal et al. 2017, arXiv:1706.02677); SGDR cosine
  schedule (Loshchilov & Hutter 2017, arXiv:1608.03983); CutMix (Yun et al. ICCV 2019, arXiv:1905.04899).
- Standing fact: the textbook SGD peak LR for batch-128 WRN/ResNet on CIFAR is ~0.1. Our baseline runs 0.2.
  EXP-016 just established that 0.3 is WORSE than 0.2 → the optimum is ≤ 0.2; the open question is whether it
  lies BELOW 0.2 (toward the textbook 0.1) or is essentially AT 0.2.
- Knowledge base (`knowledge/README.md`): only `papers/trivialaugment.md`. No LR-specific entry needed.

## Experimental History Review
- **Current best 96.22%** (EXP-012): k=4 WRN-style + Cutout(16) + TrivialAugment + compile, loss 0.195, ~91 ep.
  18 experiments, 6 improvements.
- **LR axis — now partially mapped** (EXP-016, this session): raising peak 0.2 → 0.3 REGRESSED to 95.77
  (−0.45pp, below the compiled-k4 null band ~95.92, loss 0.2018 > 0.195). Conclusion recorded in goal-learnings:
  peak 0.2 is at/ABOVE optimal; the axis has a **sign pointing DOWN** → the next probe is a LOWER peak (0.1–0.15).
  This experiment completes that probe. Note: a lower peak is NOT guaranteed to help — 0.2 could already be the
  peak of the curve (both 0.1 and 0.3 worse); the probe disambiguates "0.2 is optimal" from "optimum < 0.2."
- **Closed / saturated axes**: capacity (width ≥k5, depth), block ordering (pre-act, EXP-015), activation (SiLU),
  channel attention (SE), EMA/SWA, weight-decay, more-epochs-alone, aug POLICY (RA≈TA), shrinking-Cutout, weak Mixup.
- **Genuinely untried**: LOWER peak LR (this loop); CutMix (label-mixing aug, Mixup-cousin risk); batch size
  (blind, couples LR); schedule SHAPE (warmup fraction, decay form — secondary to peak magnitude per EXP-016).
- **Dominant constraint**: 300s epoch wall; sub-~0.2pp deltas are noise (goal-learnings High Importance). A
  compute-neutral hyperparameter change is a perfectly fair test (no throughput confound — EXP-015 lesson).

## Candidate Ideas

### 1. Lower peak LR 0.2 → 0.15 (sign-corrected LR probe)
**Summary**: Change the single constant `PEAK_LR` 0.2 → 0.15, keeping the 5% warmup → cosine-to-0 shape. Everything
else unchanged (k=4, batch 128, Nesterov, WD 1e-4, LS 0.1, Cutout(16), TrivialAugment, compile, seed 42).
**Reasoning**: EXP-016 established the LR optimum is ≤ 0.2 (0.3 regressed). 0.15 is the maximum-likelihood location
of the optimum if it lies below 0.2 — a modest step from the known-good 0.2 toward the textbook batch-128 peak (0.1),
balancing two failure modes: too-high overshoots/over-explores (0.3, EXP-016), too-low under-progresses within the
fixed ~84–91-epoch budget. Mechanism: a slightly gentler peak under heavy TA+Cutout regularization may settle into a
better minimum than 0.2 within the budget. Compute-free → perfectly fair test, and it cleanly completes the LR-axis
map regardless of sign (improve / 0.2-is-optimal / settled).
**Sources**: train.py L23 (`PEAK_LR`), L35-41 (`lr_at_fraction`); EXP-016 (0.3 regressed → optimum ≤0.2);
goal-learnings (LR axis sign points down); Goyal 2017; Loshchilov 2017.
**Estimated Effort**: trivial (one constant).
**Risk Assessment**: Modest ceiling and the optimum may simply BE 0.2 (→ null). Lower-LR could also under-progress
in the budget (→ slight regression). Graceful failure either way (baseline 96.22 holds). If 0.15 ≈ 0.2 or worse, the
LR-peak axis is settled (both directions mapped) and we pivot off it for good.

### 2. CutMix (regional label-mixing augmentation), GPU-vectorized per batch
**Summary**: Add CutMix (Yun 2019) as a per-batch GPU op alongside Cutout: with some probability, paste a random
rectangular patch from a shuffled copy of the batch and set the target to the area-weighted label mix (soft-target
cross-entropy). Implemented like `cutout_batch` (vectorized, no CPU sync) to preserve throughput.
**Reasoning**: The aug-strength axis broke the plateau once (TA, EXP-012); CutMix is the strongest evidenced aug
mechanism not yet tried — regional (spatially local like Cutout) with label interpolation, distinct from TA's
photometric transforms and Cutout's pure occlusion.
**Sources**: Yun et al. ICCV 2019 (arXiv:1905.04899); train.py L44-57 (`cutout_batch` template), L223; goal-learnings
(aug-strength axis; CutMix listed as remaining move).
**Estimated Effort**: medium (vectorized patch-paste + label-mix + soft-target loss; train.py-only, GPU op).
**Risk Assessment**: Real null/regression risk: it is the same label-mixing FAMILY as the weak Mixup that already
nulled (EXP-011) on this regularization-saturated net, and label-mixing augs characteristically need MANY epochs
(papers use 200–300) to pay off — we fit only ~84–91, so it may underfit. Higher ceiling than Idea 1 IF it works,
but lower probability. Test loss will rise (soft-target artifact) — judge on acc only.

### 3. Longer LR warmup (WARMUP_FRAC 0.05 → 0.15)
**Summary**: Extend the linear warmup from 5% to 15% of the budget, keeping peak 0.2 and cosine-to-0.
**Reasoning**: A schedule-SHAPE lever orthogonal to peak magnitude; a longer warmup can stabilize early training
under the noisy TA+Cutout gradients and let the high-LR phase be used more productively.
**Sources**: train.py L24 (`WARMUP_FRAC`), L38-39; Goyal 2017 (warmup rationale).
**Estimated Effort**: trivial (one constant).
**Risk Assessment**: Low ceiling — EXP-016 found peak MAGNITUDE is the first-order LR knob; warmup shape is
second-order. A longer warmup also eats budget at low LR → fewer productive high-LR steps → possible slight
regression. Best deferred until the peak magnitude is pinned.

## Idea Evaluation
EXP-016 made the LR-peak axis the most *informative* cheap probe available: it has a known sign (optimum ≤ 0.2) and
one modest step (0.15) completes the map — so **Idea 1** is the natural, highest-evidence next move, is compute-free
(a perfectly fair test, the cleanliness that EXP-015 lacked), and resolves the axis regardless of outcome. Idea 3 is
the same axis but a second-order knob (warmup shape) that EXP-016 implicitly deprioritized relative to peak magnitude
— do peak first. Idea 2 (CutMix) has the higher ceiling but is the riskier bet (Mixup-cousin null precedent +
epoch-budget underfit) and is better held as the fallback once the LR axis is fully mapped. Evidence (known sign) +
mechanism clarity + fair-test cleanliness + low cost select **Idea 1**; CutMix (Idea 2) is the next loop's lead if
0.15 does not improve.

## Chosen Idea
**Selected**: Lower peak LR 0.2 → 0.15 (sign-corrected LR probe)

**Why this idea**:
EXP-016 established the LR optimum is ≤ 0.2 (peak 0.3 regressed to 95.77). 0.15 is the single most informative,
lowest-cost, perfectly-fair test to complete the LR-peak map — a modest step toward the textbook batch-128 peak.
It either finds a better minimum (improvement) or settles the LR-peak axis (both directions then mapped), at zero
throughput risk.

**Hypothesis**:
Lowering `PEAK_LR` 0.2 → 0.15 (same 5% warmup, cosine-to-0) will let the heavily-augmented k=4 model settle into a
slightly better-generalizing minimum within the ~84–91-epoch budget, lifting `best_test_acc` above the 96.32 bar
(expected ~96.3–96.4) with final_test_loss ≤ 0.195. If acc is within noise of 96.22 or lower, then peak 0.2 is
already optimal and the LR-peak axis is settled (0.3 worse above, 0.15 not better below) — pivot to a different
axis (CutMix) next.
