# Brainstorm EXP-001
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
Grounded in canonical CIFAR-classification literature (heavily-trodden ground; citations are standard):

- **Zagoruyko & Komodakis, "Wide Residual Networks" (2016)** (arXiv:1605.07146)
  Widening a ResNet (more channels per layer) is more parameter- and compute-efficient than making it
  deeper/thinner: a shallow-but-wide WRN matches or beats very deep thin ResNets at equal or lower
  wall-clock, because wide layers parallelize better on GPUs and converge in fewer epochs. WRN-16-4 / -28-10
  reach ~95-96% on CIFAR-10 with full training. Directly relevant: our budget is wall-clock-bound and the
  model is capacity-bound, so trade some depth/epochs for width.
- **He et al., "Deep Residual Learning" (2015)** (arXiv:1512.03385) — ResNet-56/110 are the deep CIFAR
  baselines (~93.0/93.6% full training); projection (1×1 conv) shortcuts vs identity/zero-pad.
- **DeVries & Taylor, "Cutout" (2017)** (arXiv:1708.04552) — random square erasing; +~0.5-1 pp on CIFAR-10
  ResNets, complementary to capacity.
- **EXP-000 recipe (now baseline)**: bf16 AMP + channels_last + time-fraction cosine LR + Nesterov +
  label smoothing (reports/exp-report-000.md).

## Experimental History Review
Source: experiment-indices/improve-cifar10-test-accuracy.tsv, goal-learnings, project-insights, exp-report-000.

- **Current best / baseline**: **92.06%** (EXP-000, commit de14baa) — the modern training recipe.
- **EXP-000 (improvement, 91.73→92.06)**: bf16+channels_last+cosine+Nesterov+label-smoothing. Fit 109
  epochs / 42,156 steps (vs baseline 90 / 34,861).
- **Key learning (High Importance, goal-learnings)**: **model capacity — not training budget — is now the
  ceiling**: +21% steps gave only +0.33 pp. → prioritize wider/deeper nets + projection shortcuts +
  stronger augmentation over more epochs.
- **Project insight (High)**: VRAM headroom is enormous (164 MB / 98 GB) — capacity scaling is essentially
  free of memory cost; the binding budget is the 300s wall-clock, so favor *throughput-efficient* capacity adds.
- **Project insight (Med)**: bf16 autocast + channels_last is the safe, fast default on H20.
- **Untried gaps**: any architecture change (width, depth, shortcuts, stem), any extra augmentation
  (Cutout/mixup), recipe hyperparameter tuning (peak LR / WD / label smoothing). All architecture/aug is
  fresh ground.

## Candidate Ideas
First principles: the metric is now bounded by model capacity at a fixed 300s wall-clock. The lever with
the best capacity-per-wall-clock is **width** (WRN result: wide layers are GPU-efficient and converge in
fewer epochs), not depth (sequential, less efficient). All ideas keep the EXP-000 recipe and edit only train.py.

### 1. Widen the ResNet (WideResNet-style) + projection shortcuts
**Summary**: Increase the channel widths of the existing ResNet-20 from {16,32,64} by a width multiplier
k — e.g. k=4 → {64,128,256} (a WRN-style capacity jump, ~16× params but tiny in absolute terms), keeping
depth at 20. Replace the channel-padding identity downsample with 1×1 conv (projection) shortcuts, which
suit wider stages and are standard for WRN. Keep the full EXP-000 recipe (bf16, channels_last, cosine,
Nesterov, label smoothing). Planning to pick k by the budget/epoch tradeoff (start k=4; fall back k=2 if
epochs in 300s are too few to converge).

**Reasoning**: Directly attacks the capacity ceiling EXP-000 exposed. WRN evidence: wide-shallow beats
deep-thin at equal/again-lower wall-clock and converges in fewer epochs; wide convs use the H20 far better
than the current tiny model (which barely used the GPU — ~20k img/s on a 270k-param net). VRAM is free.
The throughput hit from widening is sublinear in practice due to better GPU utilization.

**Sources**: WRN (1605.07146); ResNet (1512.03385) for projection shortcuts; goal-learnings High-importance
capacity finding; project-insights VRAM/throughput insights; reports/exp-report-000.md.

**Estimated Effort**: medium (parametrize channel widths via a multiplier; swap shortcut to 1×1 conv +
BN; all within train.py's ResNet/BasicBlock classes).

**Risk Assessment**: Low–medium. Main risk: too-wide → too few epochs in 300s → underfit (graceful
no-improvement). Mitigations: moderate k, the time-fraction cosine already anneals fully regardless of
step count, and an early-run check of epochs/throughput can downscale k. No crash risk; VRAM ample.

### 2. Go deeper: ResNet-56
**Summary**: Increase depth (NUM_BLOCKS 3→9 → ResNet-56) keeping widths {16,32,64}, with the EXP-000 recipe.

**Reasoning**: ResNet-56 is the canonical strong CIFAR ResNet (~93% full training) and adds capacity.

**Sources**: ResNet (1512.03385); goal-learnings capacity finding.

**Estimated Effort**: low (change NUM_BLOCKS).

**Risk Assessment**: Medium. Deeper = more sequential layers, lower GPU efficiency, and more epochs needed
to converge — at a 300s budget it likely undertrains relative to a wide net (WRN paper's central point).
Higher underfit risk than widening.

### 3. Add Cutout augmentation to the current model
**Summary**: Add Cutout (one random ~8–16px square erased per image) to the train transform, keep the
EXP-000 model/recipe.

**Reasoning**: Cheap, well-evidenced +0.5–1 pp on CIFAR ResNets; reduces overfitting.

**Sources**: Cutout (1708.04552).

**Estimated Effort**: low (add a transform / in-collate erase).

**Risk Assessment**: Low. But analysis says capacity is the binding ceiling, so augmentation alone likely
yields a smaller gain and can even slow convergence at a short budget. Best composed *with* added capacity later.

## Idea Evaluation
All three respect hard constraints (train.py only, no new deps, single GPU/300s, eval once/epoch, no seed hacking).

- **Evidence strength**: Idea 1 (widen) has the most directly-applicable evidence — the WRN result is
  precisely "wide beats deep at fixed wall-clock," and our own EXP-000 proved capacity (not budget) is the
  ceiling with VRAM free. Idea 2's evidence (ResNet-56 ~93%) assumes full-length training we don't have.
  Idea 3 is well-evidenced but addresses overfitting, a secondary lever here.
- **Mechanism clarity**: Idea 1 — more channels → more capacity, and wide convs are GPU-efficient so the
  epoch cost is sublinear; crisp. Idea 2 — more depth → more capacity but worse compute-efficiency at this
  budget. Idea 3 — less overfit → better generalization, but bounded by current capacity.
- **Expected impact**: Idea 1 highest (targets the proven bottleneck with the most compute-efficient knob).
  Idea 2 moderate but budget-risky. Idea 3 modest.
- **Risk profile**: 1 and 3 fail gracefully; 2 most likely to undertrain.
- **Feasibility**: all in train.py; 1 is a moderate parametrization, 2/3 are smaller, but 1's higher
  expected impact dominates.

Idea 1 wins: best evidence, clearest mechanism, highest expected impact, and aligns with both High-importance
learnings (capacity ceiling; VRAM-free throughput-efficient scaling). Cutout (Idea 3) is the natural
follow-up to compose once capacity is increased.

## Chosen Idea
**Selected**: Idea 1 — Widen the ResNet (WideResNet-style) + projection shortcuts, on top of the EXP-000 recipe.

**Why this idea**:
EXP-000 proved the model is capacity-bound (a 21% step increase barely moved accuracy) while VRAM is
essentially free. The Wide ResNet result shows width is the most compute-efficient way to add capacity
under a wall-clock budget — wide layers parallelize well on the H20 (which the tiny 270k-param baseline
barely utilized) and converge in fewer epochs than deep-thin nets. Widening + projection shortcuts keeps
the proven training recipe intact and changes only the model definition in train.py.

**Hypothesis**:
A WideResNet-style widening (start k=4 → channels {64,128,256}) with 1×1-conv projection shortcuts, trained
with the EXP-000 recipe under the 300s budget, will raise best_test_acc meaningfully above the 92.06%
baseline (target ~93%+), because the added capacity addresses the binding ceiling and wide convs remain
throughput-efficient enough to fit sufficient epochs. If k=4 leaves too few epochs to converge, k=2
({32,64,128}) is the fallback.
