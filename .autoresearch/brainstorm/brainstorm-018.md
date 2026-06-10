# Brainstorm EXP-018
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- Knowledge base (`knowledge/README.md`): `papers/trivialaugment.md` (the EXP-012 win). No CutMix entry yet —
  will add one in planning/analysis if pursued.
- **CutMix (Yun et al., ICCV 2019, arXiv:1905.04899)**: cut a random rectangular region from one image and paste
  the corresponding patch from another, mixing the labels in proportion to the patch AREA. A strong regional
  augmentation+regularizer: unlike Mixup's global pixel blend, the pasted region is spatially coherent (real local
  features), and unlike Cutout's zeroing it wastes no pixels. Reports consistent CIFAR/ImageNet gains. Standard
  recipe: λ ~ Beta(α=1.0, α=1.0), box of area (1−λ), loss = λ·CE(out, y_a) + (1−λ)·CE(out, y_b), applied with some
  per-batch probability p.
- **Caveat from the literature**: CutMix's benefit is realized over LONG schedules (the paper trains 200–300
  epochs on CIFAR). At our ~84–91-epoch budget the regularizer may not have "warmed up," risking underfit.
- **Mixup vs CutMix**: Mixup (global blend) nulled here as a *weak* (α=0.2) variant (EXP-011); CutMix is regional
  and typically stronger/more effective on CIFAR — but it is the same label-mixing FAMILY, so the precedent is a
  yellow flag, not a green light.

## Experimental History Review
- **Current best 96.22%** (EXP-012): k=4 WRN-style + Cutout(16) + TrivialAugment + compile, loss 0.195, ~91 ep.
  19 experiments, 6 improvements.
- **LR-peak axis SETTLED this session** (EXP-016/017): sweep 0.15→95.58, **0.2→96.22 (optimum)**, 0.3→95.77 —
  0.2 is a clean interior optimum; axis closed. Block-ordering (pre-act, EXP-015) also closed (+ throughput confound).
- **Augmentation history** (most relevant to CutMix): TrivialAugment (strong, diverse, INPUT-SPACE) gained +0.22pp
  (EXP-012); RandAugment ≈ TA (policy saturated, EXP-014); shrinking Cutout under-regularized (EXP-013, sweet spot
  ≥16); **weak Mixup (label-mixing, α=0.2) nulled** (EXP-011, ~88 ep) — the regularization-saturated net gave
  diminishing returns to a second weak aug. Corrected lesson: *strong, diverse* aug can still gain.
- **Closed/saturated axes**: capacity (k≥5), block-ordering, activation (SiLU), SE attention, EMA/SWA, weight-decay,
  more-epochs-alone, aug-POLICY (RA≈TA), shrinking-Cutout, LR-peak.
- **Genuinely untried**: CutMix (this loop); batch size (blind, couples LR — though LR now mapped); DropBlock /
  stochastic-depth (regularizers; shallow net limits stochastic-depth); LR-schedule SHAPE (low EV, deprioritized).
- **Dominant constraint**: 300s epoch wall (~84–91 ep); sub-~0.2pp deltas are noise; compute-neutral fair tests
  preferred (EXP-015 confound lesson — but CutMix as a per-batch GPU op is throughput-neutral like Cutout).

## Candidate Ideas

### 1. CutMix (regional label-mixing augmentation), GPU-vectorized per batch
**Summary**: Add CutMix as a per-batch GPU op. With probability p (≈0.5) per batch: draw λ~Beta(1,1), choose a
random box of area (1−λ), paste that box from a randomly shuffled copy of the batch (`x[perm]`), recompute λ from
the true box area, and compute the loss as λ·CE(out, y) + (1−λ)·CE(out, y[perm]) (each with label smoothing).
Keep the existing per-batch Cutout and TrivialAugment (validated recipe). Implemented vectorized on-GPU (like
`cutout_batch`, no CPU sync) so throughput/epoch count are preserved → fair same-budget test.
**Reasoning**: The augmentation-strength axis is the one that broke the plateau (TA, EXP-012). CutMix is the
strongest evidenced aug mechanism not yet tried — regional (real local features, spatially coherent, distinct from
TA's photometric transforms and Cutout's occlusion) with label interpolation. It directly targets the residual
generalization gap that bounds this fixed-capacity net.
**Sources**: Yun et al. ICCV 2019 (arXiv:1905.04899); train.py L44-57 (`cutout_batch` GPU-vectorized template),
L223 (per-batch aug site), L232-236 (loss site — needs the two-target mix); goal-learnings (aug-strength axis;
CutMix the remaining mechanism).
**Estimated Effort**: medium (vectorized box sampling reusing the `cutout_batch` coordinate math + batch-shuffle
paste + two-term soft-target loss; ~20–30 lines, train.py-only, all on-GPU).
**Risk Assessment**: (a) Same label-mixing FAMILY as the weak Mixup that nulled (EXP-011) on this
regularization-saturated net; (b) CutMix characteristically needs MANY epochs (papers 200–300) to pay off — at
~84–91 ep it may UNDERFIT and slightly regress; (c) stacking CutMix + Cutout + TA may over-regularize within the
budget. Mitigations: moderate prob p=0.5 (half the batches keep the plain recipe), α=1.0 (standard). Test loss will
rise (soft-target artifact) — judge on acc only. Higher ceiling than any remaining knob IF it works; graceful
no-improvement otherwise (baseline 96.22 holds).

### 2. Batch size 128 → 256 + linear LR scaling (peak 0.2 → 0.4)
**Summary**: Double `BATCH_SIZE` and scale `PEAK_LR` by 2× (linear-scaling rule), keeping warmup/cosine.
**Reasoning**: Now that the LR-peak is mapped at batch 128 (optimum 0.2), the linear-scaling target (0.4) for
batch 256 is principled rather than blind. On a launch-bound net a larger batch may raise img/s (amortized launch
overhead) and lower gradient variance.
**Sources**: Goyal 2017 (arXiv:1706.02677); project-insights (launch-bound k=4, VRAM headroom ~0.5/98 GB); train.py L22-23.
**Estimated Effort**: low (two constants).
**Risk Assessment**: Epochs already saturated (more img/s won't help, EXP-007); large-batch generalization gap is
real and the linear-scaling rule is only approximate (may need the LR re-mapped at the new batch → couples two
unknowns). Blind-ish; medium-low confidence. Defer behind CutMix.

### 3. DropBlock regularization in the conv stages
**Summary**: Add DropBlock (structured spatial dropout — drop contiguous feature-map regions) to the residual
stages, a regularizer mechanism distinct from Cutout (input) and label smoothing (output).
**Reasoning**: Targets the generalization gap via feature-space structured noise; complementary to input/label aug.
**Sources**: Ghiasi et al. 2018 (DropBlock); train.py BasicBlock.
**Estimated Effort**: medium (DropBlock module + keep_prob schedule; train.py-only).
**Risk Assessment**: Adds 2 hyperparameters (block_size, keep_prob) that need tuning; on a shallow 20-layer net the
effect is typically small and DropBlock shines more in deeper/wider nets; the net is already regularization-saturated
(Cutout+TA+LS+WD), echoing the EXP-008/010/011 nulls. Lower evidence and higher tuning burden than CutMix.

## Idea Evaluation
With the LR-peak and block-ordering axes now closed this session, the productive move is a genuinely new
*mechanism*, not another knob. **CutMix (Idea 1)** has by far the strongest external evidence (consistent published
CIFAR gains), the clearest mechanism (regional, label-mixing regularizer orthogonal to TA's photometric and
Cutout's occlusion), and the highest ceiling of anything left — and it is implementable as a throughput-neutral
per-batch GPU op (fair test). Its risks are real and must be stated honestly (Mixup-cousin null precedent +
epoch-budget underfit), but they are graceful failure modes, and a clean CutMix result — win or null — is the
decisive test of whether the augmentation axis has any remaining headroom beyond TA. Idea 2 (batch size) is
blind-ish and fights the saturated-epochs finding; Idea 3 (DropBlock) is lower-evidence, adds tuning burden, and
leans into the regularization-saturation nulls. Evidence + mechanism + ceiling + fair-test feasibility select
**Idea 1**. If CutMix nulls, the augmentation axis (and likely the 96.22 plateau at fixed k=4 capacity) is settled.

## Chosen Idea
**Selected**: CutMix (regional label-mixing augmentation), GPU-vectorized per batch

**Why this idea**:
With LR and architecture axes now closed, CutMix is the last well-evidenced untried *mechanism* — the strongest
remaining augmentation lever (regional, label-mixing, distinct from the photometric TA and occlusion Cutout already
in the recipe). It is implementable throughput-neutrally on-GPU (fair same-budget test) and either breaks the
plateau or decisively settles the augmentation axis.

**Hypothesis**:
Adding CutMix (p=0.5, α=1.0) on top of the TA+Cutout recipe will reduce the residual generalization gap and lift
`best_test_acc` above the 96.32 bar (expected ~96.3–96.6). Because CutMix is a label-mixing aug that benefits from
long schedules, the main downside risk is underfit within the ~84–91-epoch budget → a null/slight regression near
96.0, with final_test_loss rising (soft-target artifact, judged on accuracy only). A null would, together with the
weak-Mixup null (EXP-011), settle the augmentation axis: TA is the ceiling for this recipe/budget.
