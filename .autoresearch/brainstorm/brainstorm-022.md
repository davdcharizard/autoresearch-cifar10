# Brainstorm EXP-022
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Wide Residual Networks — Zagoruyko & Komodakis, BMVC 2016 (arXiv:1605.07146)**: this is the paper our k=4 widened
  ResNet directly descends from (WideResNet-style {16k,32k,64k} stages). Its central *regularization* recommendation
  is **dropout inserted between the two 3×3 convolutions of each residual block** (after the first ReLU, before the
  second conv) — NOT the usual "dropout in the classifier". The paper shows this consistently improves WRN on
  CIFAR-10/100 (e.g. WRN-28-10 CIFAR-10 error 4.00→3.89 with dropout), attributing the gain to the wide regime having
  more co-adaptation/redundancy for dropout to regularize. They use p≈0.3 for their long (200-epoch) schedules.
  This is the architecture's *own* recommended regularizer and we have never tested it. (No external fetch needed
  beyond repo knowledge; relevant for the plan to add a `knowledge/papers/wrn-dropout.md` entry.)
- **Knowledge base** (`.autoresearch/knowledge/README.md`): has trivialaugment.md, cutmix.md, swa.md — all augmentation/
  averaging axes now closed. No dropout entry exists yet.

## Experimental History Review

Current best = **96.22%** (EXP-012, commit 6c417a4). 22 experiments; ~14 axes closed. Binding constraint:
generalization at fixed k=4 capacity in 300s (~92 epochs). The model is generalization-bound.

**Closed axes (do NOT revisit):** capacity k>4 (epoch wall, EXP-004/009), LR-peak (0.2 interior optimum,
EXP-016/017), block-order/pre-act (EXP-015), activation/SiLU (EXP-010), SE channel-attention (EXP-008),
weight-decay (EXP-005), more-epochs alone (EXP-007), auto-aug policy TA≈RA (EXP-014), occlusion-strength /
Cutout-size (16 is interior optimum, EXP-013/021), label-mixing aug Mixup/CutMix (EXP-011/018), weight-averaging
EMA/SWA (EXP-006/019/020).

**The decisive meta-insight (project-insights High Importance, EXP-012):** *"Regularization saturated" is
mechanism-specific — do NOT declare an axis closed from weak-variant nulls; test the strongest, most
mechanistically-DISTINCT variant.* TrivialAugment broke the plateau precisely because it was a strong, diverse,
distinct mechanism after WD/Mixup read as "saturated".

**Genuinely untested regularization MECHANISMS** (distinct loci, not yet probed):
- **Dropout between block convs (WRN-recommended)** — intermediate-FEATURE regularization. Distinct from input-space
  aug (TA/Cutout), weight-space (WD), label-space (LS, mixing), and trajectory (SWA). NEVER tested. The WRN paper
  explicitly recommends it for exactly this architecture. This is the strongest-evidenced untested mechanism.
- Label-smoothing VALUE (0.1 fixed since EXP-000, never swept) — label-space, uncertain direction.
- Per-channel input std-norm (std=(1,1,1)→true std) — input-scaling; expected null (BN absorbs it).

## Candidate Ideas

### 1. Dropout in the WideResNet residual blocks (WRN-style, between the two convs)
**Summary**: Add a `nn.Dropout2d(p)` (or `nn.Dropout`) in `BasicBlock.forward` between the first ReLU and the second
conv — i.e. `out = F.relu(bn1(conv1(x))); out = dropout(out); out = bn2(conv2(out))` — exactly the placement
Zagoruyko & Komodakis 2016 prescribe for Wide ResNets. Add a `DROPOUT_P` constant (first probe **p=0.1**) and a
`self.dropout` module in `BasicBlock`. Everything else identical to the EXP-012 baseline (k=4, TA+Cutout(16),
PEAK_LR 0.2 cosine-to-0, compile, seed 42). Params unchanged (dropout has none); compute near-neutral (one cheap
elementwise mask per block).

**Reasoning**: The model is generalization-bound and ALL closed regularization axes are either input-space (aug),
weight-space (WD), label-space (LS/mixing), or trajectory (SWA). **Intermediate-feature dropout is a distinct,
untested locus** — and it is the regularizer the WRN paper specifically recommends for wide residual nets, citing the
wide regime's extra redundancy/co-adaptation as what dropout regularizes. Per the project's High-Importance
meta-insight, "saturated" verdicts from weak/redundant regularizers (WD, mild Mixup) do NOT close a distinct strong
mechanism — TrivialAugment proved this. Dropout-in-WRN is the best-evidenced such mechanism remaining. Clean
attribution (params unchanged), throughput-neutral, fair same-budget test.

**Sources**: Zagoruyko & Komodakis 2016 (WRN, arXiv:1605.07146) § dropout; project-insights High Importance
("regularization saturated is mechanism-specific", EXP-012); train.py L65-92 (`BasicBlock`).

**Estimated Effort**: low — one constant + ~2 lines in `BasicBlock` (`__init__` adds `self.dropout`, `forward`
applies it).

**Risk Assessment**: The recipe is heavily regularized (TA+Cutout+LS+WD) and at 92 epochs (vs WRN's 200) strong
dropout risks UNDER-fitting — the exact failure mode that sank CutMix (EXP-018, strong aug, short budget). Mitigation:
first probe at a MILD **p=0.1** (not the paper's 0.3) so it adds feature regularization without strongly slowing
convergence; if it gains, sweep up (0.2/0.3) in a follow-up. Fails gracefully (no-improvement / mild loss rise if it
under-fits). `Dropout2d` (channel-wise) vs `Dropout` (elementwise): WRN uses plain elementwise dropout between convs;
use `nn.Dropout` to match the paper.

### 2. Lower label smoothing (LABEL_SMOOTHING 0.1 → 0.05)
**Summary**: Reduce LS from 0.1 to 0.05 (one constant, train.py L27). LS has been fixed at 0.1 since EXP-000 and
never swept.

**Reasoning**: With TA + Cutout + WD already providing strong regularization, 0.1 LS may over-soften targets and
slightly cap top-1; reducing it could let the model sharpen decision boundaries. A genuinely untested knob, clean
and fair (no compute/param change).

**Sources**: train.py L27 (LS fixed 0.1 since EXP-000); project pattern "validated recipe … label-smoothing(0.1)".

**Estimated Effort**: low — one constant.

**Risk Assessment**: The recipe reads as regularization-saturated (WD-up null, smaller-Cutout hurt), so REDUCING a
regularizer may instead increase overfitting and hurt — direction genuinely uncertain. LS top-1 effects are usually
small (mainly calibration). Likely within noise. Weaker evidence and mechanism than Idea 1.

### 3. Per-channel input std-normalization (std (1,1,1) → true CIFAR std)
**Summary**: Normalize inputs by true per-channel std (≈(0.247,0.243,0.261)) instead of (1,1,1) (mean-only), train.py
L152-155.

**Reasoning**: Standard preprocessing; the one untried input-side knob. Cheap, definitively closes the
input-normalization axis.

**Sources**: train.py L152-155 (the `std=(1,1,1)` comment flags this); standard CIFAR practice.

**Estimated Effort**: low — one tuple.

**Risk Assessment**: First layer is Conv→BatchNorm; BN almost certainly absorbs a per-channel input rescale →
expected NULL. Low ceiling; an axis-closer, not a real lead.

## Idea Evaluation

**Evidence strength**: Idea 1 is strongest — it is the regularizer the WRN paper *specifically recommends for this
exact architecture*, and it targets a regularization LOCUS (intermediate features) that no prior experiment has
touched. The project's own High-Importance meta-insight (don't close an axis from weak-variant nulls; a strong
distinct mechanism can still gain — proven by TrivialAugment) directly endorses probing it. Idea 2 is a never-swept
knob but with weaker, direction-ambiguous evidence. Idea 3 is an expected null.

**Mechanism clarity**: Idea 1 — clear: dropout reduces feature co-adaptation in the wide layers, a distinct
generalization mechanism. Idea 2 — ambiguous direction (less reg could help OR hurt). Idea 3 — almost certainly
nulled by BN.

**Expected impact**: Idea 1 highest (distinct untested mechanism on the binding generalization constraint, with
architecture-matched literature support). Idea 2 low-medium. Idea 3 ≈ 0.

**Risk profile**: All fail gracefully. Idea 1's specific risk is under-fitting at the short budget, mitigated by the
mild p=0.1 first probe. All are compute-neutral, params-unchanged (Idea 1 adds no params), single-knob — clean fair
tests.

**Feasibility**: Idea 1 ~2 lines + a constant; Ideas 2/3 one constant/tuple. All trivial.

Conclusion: **Idea 1 (WRN dropout, p=0.1)** is the lead — the best-evidenced untested regularization mechanism, on
the exact architecture the recommending paper studied, targeting a locus no prior experiment probed, directly
endorsed by the project's "test the strong distinct variant" meta-insight. Idea 2 (LS sweep) is the natural
follow-up; Idea 3 is a cheap axis-closer for a later loop.

## Chosen Idea
**Selected**: Dropout in the WideResNet residual blocks (WRN-style, p=0.1, between the two convs)

**Why this idea**:
The model is generalization-bound and every closed regularization axis lives at a different locus (input/weight/
label/trajectory). Intermediate-FEATURE dropout is the one distinct, untested regularization mechanism — and it is
exactly what the Wide ResNet paper recommends for this architecture, citing the wide regime's redundancy. The
project's strongest meta-insight says not to declare regularization closed from weak-variant nulls but to test the
strongest mechanistically-distinct variant (as TrivialAugment proved). Dropout-in-WRN is that variant. It is a clean,
fair, param-neutral, throughput-neutral single-knob test.

**Hypothesis**:
Adding WRN-style dropout (p=0.1) between the block convs will reduce feature co-adaptation and the residual
generalization gap, lifting best_test_acc above the 96.32 bar. If instead acc falls / test-loss rises (dropout
under-fits at the ~92-epoch budget, à la CutMix), feature-dropout does not help at this short budget and the
mechanism is closed (with a possible higher-p or longer-budget caveat noted for the record).
