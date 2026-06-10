# Brainstorm EXP-037
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

<!-- Ideation only. Metric/direction/constraints/verification live in the goal file;
     baseline (96.22, commit 6c417a4) lives in experiment-indices/improve-cifar10-test-accuracy.tsv. -->

## Web Search & Literature Review

No new external search. Grounding from standard practice + project knowledge base:
- **Standard CIFAR-10 normalization**: per-channel mean `(0.4914, 0.4822, 0.4465)` AND per-channel std `(0.2470, 0.2435, 0.2616)`. The current code (train.py L152-155) subtracts the mean but uses `std=(1,1,1)` — i.e. NO variance normalization. Unit-variance inputs are the assumption behind Kaiming init (He et al. 2015, already used at conv1) and standard input whitening.
- Project-insights Medium explicitly lists **"input normalization"** as an untried *convergence-neutral* lever (the class that, unlike regularizer-adds, doesn't cost epochs).
- `knowledge/papers/sam.md`, `gradient-centralization.md`, `swa.md`: the optimizer/flat-minima generalization levers — all CLOSED here.

## Experimental History Review

**Current best / baseline**: 96.22% (EXP-012, 6c417a4); bar 96.32 (+0.1). 36 experiments; plateau extremely well-confirmed.

**The plateau is now mapped across ~28 axes** — diagnosis: generalization-bound at fixed k=4 capacity, convergence-saturated (~77 ep), bounded by two walls: (1) compute/epoch wall (any FLOP add → fewer epochs → regress, High); (2) polish-vs-top1 (compute-neutral *optimization* polish → loss not top-1, Medium).

**CLOSED**: capacity (k>4); ALL augmentation (TA+Cutout ceiling; policy/Mixup/CutMix/Cutout-size/cooldown/cooldown-reheat); ENTIRE LR-schedule; all regularizer-adds; architecture (SE/SiLU/preact/ResNet-D/BlurPool/multi-scale-head/large-batch); optimizer gradient-dynamics (GC) AND objective (SAM, EXP-036); weight-averaging (EMA/SWA).

**Untried, compute-neutral, convergence-neutral gap**: **input normalization** — the code never normalizes input variance (`std=(1,1,1)`). This is the last clean compute-neutral lever explicitly flagged by project-insights and never tested. It is NOT a regularizer-add (no convergence penalty) and NOT optimization-polish-on-a-converged-recipe in the usual sense (it changes the data the net sees, not the optimizer). Whether it moves top-1 is genuinely open, though the expected effect is bounded because conv1 is immediately followed by BatchNorm.

## Candidate Ideas

> **DISCARDED (infeasible) — input std-normalization**: the original lead (set train `std=(1,1,1)→(0.2470,0.2435,0.2616)`) is INFEASIBLE. The frozen `Eval.evaluate()` (prepare.py L13) hardcodes `mean,std = (0.4914,0.4822,0.4465),(1,1,1)` — the test set is normalized with std=1 and CANNOT be changed (prepare.py is protected). Any train-side std change would make the model train on unit-variance inputs while eval feeds ~4× larger-scale inputs → train/test distribution MISMATCH → catastrophic regression classified `invalid`, not a real test. Input normalization is therefore PINNED by the frozen eval; the axis is closed by infeasibility. (Verified: `grep Normalize prepare.py`.)

### 1. RandomCrop reflect-padding (padding_mode 'constant'→'reflect')
**Summary**: Change the train-only `transforms.RandomCrop(32, padding=4)` (train.py L158) to `RandomCrop(32, padding=4, padding_mode="reflect")`. Default padding is zero ('constant') → the 4-px border added before the random crop is black; reflect-padding mirrors the edge pixels instead. Train-only (eval/test_tf does NOT crop — prepare.py uses full 32×32 images), so there is NO train/eval mismatch risk. Compute-neutral (CPU PIL op, no GPU sync, no FLOP/epoch change).
**Reasoning**: This improves the QUALITY of the existing geometric augmentation rather than its strength: zero-padding injects artificial black borders into ~12% of each padded image's area, so translated crops contain unnatural black wedges the test images never have; reflect-padding produces translated views whose borders match natural image statistics, tightening the train/test distribution match for the one augmentation applied every step. It is NOT a regularizer-ADD (same crop, same strength → dodges the convergence-penalty/underfit pattern of EXP-018/022) and NOT a policy swap (distinct from the closed RandAugment-vs-TA axis, EXP-014). Compute-neutral → no epoch wall.
**Sources**: standard CIFAR practice (reflect-padding common in strong recipes); project-insights (compute-neutral convergence-neutral levers); train.py L156-167; eval-no-crop confirmed in prepare.py L15-19.
**Estimated Effort**: low — one-argument change. One run.
**Risk Assessment**: **Likely within-noise** — the net is regularization-saturated and the border artifact affects only a thin frame; BN + the existing strong aug may already absorb it. Safe: train-only, compute-neutral, no crash/mismatch risk. Worst case: clean within-noise null. Modest, defensible upside (better-matched augmented distribution).

### 2. ResNeXt-style grouped convolution at fixed FLOPs (cardinality)
**Summary**: Replace the 3×3 convs in BasicBlock with grouped convs (cardinality C) + a width bump to hold FLOPs ≈ constant, adding "cardinality" (ResNeXt, Xie et al. 2017) as a capacity dimension orthogonal to width/depth.
**Reasoning**: ResNeXt beats ResNet at equal FLOPs on ImageNet; cardinality is the one capacity axis untried here (width=closed, depth=untouched-but-compute-walled).
**Sources**: ResNeXt (Xie et al. 2017); capacity axis (EXP-004/009).
**Estimated Effort**: medium — restructure BasicBlock convs + width math.
**Risk Assessment**: HIGH — (a) grouped convs are often LESS hardware-efficient on GPU → likely RAISES dt → epoch wall (compute-confound, the dominant failure mode here); (b) the "deep/large-image tricks don't transfer to shallow 32×32 CIFAR" Medium insight has killed every architectural restructure tried (SE/preact/ResNet-D/BlurPool/multi-scale-head); (c) changes params. Low confidence, high compute-confound risk.

### 3. Stronger augmentation: two independent TrivialAugment draws per image
**Summary**: Apply `TrivialAugmentWide()` twice per image (two random ops) for stronger, more diverse augmentation, compute-neutral (CPU PIL ops, no GPU sync).
**Reasoning**: The High insight says "test the strongest, most diverse variant before concluding" an aug axis closed.
**Sources**: project-insights High (strong-diverse-aug); TrivialAugment (EXP-012); RandAugment(2,9) (EXP-014).
**Estimated Effort**: low.
**Risk Assessment**: **Likely null/regression** — EXP-014 already showed RandAugment(2 ops)≈TA(1 op) (policy/op-count doesn't matter once strong aug present), and the regularizer-add pattern (EXP-018/022) shows stronger aug can underfit at the short budget. Two-draw TA ≈ RA(2,9) → expected within-noise-to-negative. Low value.

## Idea Evaluation

The original lead (input std-normalization) is dead on feasibility (frozen eval pins std=1). Of the survivors, the plateau is bounded by the compute wall and the polish pattern; the only moves that avoid both are compute-neutral AND not-already-closed.

- **Compute-neutrality**: #1 (reflect-pad) and #3 (two-draw TA) are compute-neutral; #2 (ResNeXt) very likely trips the compute wall (grouped-conv GPU inefficiency raises dt) — its dominant outcome is a confounded regression, the exact trap that closed SE/preact/ResNet-D/BlurPool. #2 dominated.
- **Novelty / non-redundancy**: #1 is genuinely untried and targets a DIFFERENT lever than any closed aug axis — augmentation *quality* (border statistics), not strength (Cutout-size EXP-013/021) or policy (RA-vs-TA EXP-014) or label-mixing (EXP-011/018). #3 is effectively a re-test of the closed aug-op-count axis (EXP-014, RA(2 ops)≈TA) → low information and risks over-aug underfit.
- **Mechanism / expected impact**: #1 has a clean, defensible mechanism (remove the black-border artifact that zero-padding injects into every cropped training image → augmented distribution closer to the artifact-free test images) and is feasible + safe. Honest ceiling is modest (regularization-saturated net, thin border). #3's mechanism is known-null. #2 could add real capacity but is overwhelmingly likely to regress on compute + shallow-net-transfer grounds.
- **Risk**: #1 safest (one-arg, train-only → no compute/crash/eval-mismatch risk). #3 safe but near-certain null/underfit. #2 high-risk.

#1 wins: compute-neutral, genuinely untried, feasible, low-risk, with a distinct (augmentation-quality) mechanism not covered by any closed aug axis. A clean null closes the crop-padding-mode sub-lever; an unlikely gain breaks the plateau.

## Chosen Idea
**Selected**: RandomCrop reflect-padding (`padding_mode` default 'constant' → 'reflect')

**Why this idea**:
After the input-std-normalization lead was killed by the frozen eval (std pinned to 1), reflect-padding is the best feasible remaining lever: compute-neutral, train-only (no train/eval mismatch — eval doesn't crop), genuinely untried, and mechanistically distinct from every closed augmentation axis (it improves the *quality*/border-statistics of the one geometric aug applied every step, not its strength or policy). Zero compute-wall risk and a clean failure mode. It is the honest next probe given ~28 closed axes; its expected impact is modest but its cost and risk are near-zero.

**Hypothesis**:
Reflecting (instead of zero-filling) the 4-px crop border will remove the artificial black wedges that zero-padding injects into translated training crops, tightening the train/test distribution match and marginally improving best_test_acc — tested against the bar 96.32. Most likely outcome (honest): within-noise (~96.1–96.3), since the net is regularization-saturated and BN may absorb the thin-border effect — which would close the crop-padding-mode sub-lever. Throughput-neutral (~91 ep, params 4,299,866 unchanged); no eval-mismatch risk (train-only augmentation).
