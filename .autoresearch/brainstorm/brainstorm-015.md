# Brainstorm EXP-015
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- **Identity Mappings in Deep Residual Networks / pre-activation ResNet (He et al., ECCV 2016)**
  (https://arxiv.org/abs/1603.05027): reordering each residual block to **BN→ReLU→conv** (pre-activation) instead of
  the original conv→BN→ReLU (post-activation, ResNet-v1) gives a cleaner identity path, easier optimization, and
  better generalization at ~zero extra params/FLOPs. The shortcut becomes a bare projection (no BN), and a single
  final BN→ReLU is added before global pooling.
- **Wide Residual Networks (Zagoruyko & Komodakis, BMVC 2016)** (https://arxiv.org/abs/1605.07146): the canonical
  WRN — which our model is described as ("WideResNet-style k=4") — uses **pre-activation** blocks. Our current
  `BasicBlock` is actually the **post-activation ResNet-v1** ordering (conv→BN→ReLU, ReLU after the residual add).
  So we are running a "WRN-style" width on v1 blocks — converting to true pre-activation WRN blocks is the canonical,
  evidenced architectural upgrade we have not yet tried.
- Note: pre-activation's gain is largest for very deep nets; on a shallow ResNet-20 it is smaller but generally
  non-negative, and it is essentially free in compute.

## Experimental History Review
- **Current best 96.22%** (EXP-012): k=4 WRN-style (post-act v1 blocks) + Cutout(16) + TrivialAugment + compile, loss 0.195.
- **Augmentation axis now MAPPED** (goal-learnings): adding strong diverse aug gained (TA, EXP-012); reducing it lost
  (EXP-013); swapping the auto-aug *policy* is saturated (RA≈TA, EXP-014). → STOP tuning augmentation; open a new axis.
- **Closed axes**: width ≥k5 (epoch wall), weight-decay, EMA/SWA (cosine-to-0), more-epochs-alone, SE/channel-attn,
  SiLU, weak-Mixup, auto-aug-policy, shrinking-Cutout.
- **Untried axes**: (a) **block micro-architecture / activation ordering** (pre-act vs post-act) — never touched;
  (b) LR-schedule (peak LR/warmup) — only WD swept (EXP-005, old recipe); (c) different aug *mechanism* (larger
  Cutout, Mixup/CutMix on TA).
- **Dominant constraint**: 300s epoch wall — any change must keep throughput (pre-act is compute-neutral; LR tuning
  is free).

## Candidate Ideas

### 1. Pre-activation (true-WRN) BasicBlocks: BN→ReLU→conv, bare-conv shortcuts, final BN→ReLU
**Summary**: Convert the post-activation `BasicBlock` (conv→BN→ReLU ×2, ReLU after add) to the canonical
pre-activation form: `out = conv1(relu(bn1(x)))`, `out = conv2(relu(bn2(out)))`, `out += shortcut`, **no** ReLU after
the add. The downsample/projection shortcut becomes a bare 1×1 conv applied to the *pre-activated* input
`relu(bn1(x))` (no shortcut BN). The stem becomes just `conv1` (drop the stem BN→ReLU; the first block's bn1 now
provides it), and a final `BN→ReLU` is added after layer3 before global avg-pool. This is the true WideResNet
formulation (our model is currently "WRN-style width on ResNet-v1 blocks").
**Reasoning**: Pre-activation gives a cleaner identity/gradient path and better generalization at ~zero compute cost
(He 2016; the standard WRN uses it). It is the canonical architecture our "WRN-style" model is NOT using — the most
evidence-backed *structural* lever left now that augmentation is mapped. Compute-neutral → throughput and epoch count
stay ~unchanged (still launch-bound, compile applies), so it's a fair test at the same budget.
**Sources**: He et al. 2016 (arXiv:1603.05027); Zagoruyko & Komodakis 2016 (arXiv:1605.07146); train.py BasicBlock
L88-92 (current post-act ordering); goal-learnings (augmentation mapped → open architecture axis).
**Estimated Effort**: medium (restructure BasicBlock forward + shortcut, drop stem BN→ReLU, add final BN→ReLU in
ResNet.forward; ~15 lines, train.py-only).
**Risk Assessment**: (a) num_params shifts by a few hundred (shortcut BNs removed, one final BN added) — expected
for an architecture change, so the "params unchanged" check no longer applies (note expected ≈4,299,466). (b) On a
shallow ResNet-20 the pre-act gain may be small → possible noise-scale null. (c) Implementation-bug risk (wrong
shortcut placement / missing final BN) → guard with abort criteria (NaN, acc not tracking ≥ ~94% by mid-run, clean
compile). Worst case graceful no-improvement; baseline 96.22 holds. Corroborate any gain with final_test_loss.

### 2. LR-schedule micro-tuning (peak LR 0.2 → 0.1, the textbook batch-128 WRN value)
**Summary**: Lower `PEAK_LR` 0.2→0.1 (standard SGD peak for batch 128 WRN), keeping cosine-to-0 + 5% warmup.
**Reasoning**: Peak 0.2 was set heuristically in EXP-000 and never re-tuned after width/aug changes; with TA's noisier
gradients a lower peak may converge to a better minimum. Trivial, compute-free, orthogonal to augmentation.
**Sources**: train.py L23 (PEAK_LR=0.2); EXP-000 (heuristic peak); EXP-005 (only WD swept).
**Estimated Effort**: low (one constant).
**Risk Assessment**: Blind direction — at only ~91 epochs a lower peak could *underfit* (too little exploration in
the budget) just as easily as help. Low confidence, low ceiling; best kept as a fallback probe.

### 3. Larger Cutout (20px) under TA
**Summary**: `CUTOUT_SIZE` 16→20 — a different aug *mechanism* than the saturated policy axis; EXP-013 bounded the
occlusion sweet spot below at 16.
**Reasoning**: 8<16 (EXP-013) → sweet spot ≥16; test >16.
**Sources**: EXP-013; DeVries & Taylor 2017.
**Estimated Effort**: low.
**Risk Assessment**: 16px is the textbook CIFAR-10 optimum and TA already adds strength → 20px likely over-regularizes
(underfit, loss↑). Low ceiling.

## Idea Evaluation
With the augmentation axis mapped (Ideas 3 and the policy axis are low-ceiling), the productive move is a NEW axis.
Idea 1 (pre-activation) has the strongest **evidence** (canonical WRN/pre-act ResNet, the formulation our model
*should* use) and the clearest **mechanism** (cleaner identity path + better generalization at zero compute cost),
with the highest ceiling of the three. Its costs — a few-hundred param shift and implementation care — are
manageable and the change is compute-neutral so it's a fair same-budget test. Idea 2 (LR) is trivial but blind/low-
ceiling; Idea 3 (larger Cutout) is low-ceiling and likely over-regularizes. Idea 1's failure mode is graceful
(no-improvement, baseline holds) and even a null is a clean, valuable architectural data point.

Evidence + mechanism + highest ceiling + compute-neutral fairness select **Idea 1**. Idea 2 is the natural fallback
if pre-activation nulls.

## Chosen Idea
**Selected**: Pre-activation (true-WRN) BasicBlocks: BN→ReLU→conv, bare-conv shortcuts, final BN→ReLU

**Why this idea**:
The augmentation axis is now mapped (TA wins; policy saturated; Cutout sweet-spot ≥16), so the productive next move
is to open a new, evidence-backed axis. Our model is described as "WideResNet-style" but actually runs post-activation
ResNet-v1 blocks; converting to the canonical pre-activation WRN formulation is the most literature-supported
*structural* improvement available, at essentially zero compute cost (so it stays a fair same-budget, ~91-epoch test).
It is train.py-only and fails gracefully.

**Hypothesis**:
Converting to pre-activation blocks (BN→ReLU→conv, bare-conv shortcuts, final BN→ReLU) will improve the
identity/gradient path and generalization, lifting `best_test_acc` above the 96.32 bar (expected ~96.3–96.6%) with a
corroborating final_test_loss ≤ 0.195, at unchanged throughput/epochs. If acc is within noise of 96.22 with loss
≈ 0.195, pre-activation's benefit is negligible on this shallow k=4 ResNet-20 and the block-ordering axis is settled.
