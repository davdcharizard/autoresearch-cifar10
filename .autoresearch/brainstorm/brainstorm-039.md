# Brainstorm EXP-039
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

<!-- Ideation only. Metric/direction/constraints/verification live in the goal file;
     baseline (96.22, commit 6c417a4) lives in experiment-indices/improve-cifar10-test-accuracy.tsv. -->

## Web Search & Literature Review

No new external search. Grounding from project knowledge base + standard practice:
- **Cosine / normalized-softmax classifiers** (e.g. "Large-Margin"/CosFace-style, and normalized-softmax
  used in several strong image recipes): L2-normalize both the penultimate features and the classifier
  weight rows, score by scaled cosine similarity `s·cos θ`. Removes the norm degrees of freedom from the
  decision rule → purely angular class boundaries; reported small, consistent top-1/robustness gains and
  better-conditioned boundaries on balanced classification, at ~zero compute and no convergence penalty.
- knowledge/papers: all optimizer/flat-minima/polish/aug papers — CLOSED here.

## Experimental History Review

**Current best / baseline**: 96.22% (EXP-012, 6c417a4); bar 96.32 (+0.1). 39 experiments; plateau
extremely well-confirmed.

**The plateau is bounded by THREE established walls (project-insights High/Medium):**
1. **Compute/epoch wall** — ANY non-trivial FLOP add (and even FLOP-neutral but wall-clock-heavier
   changes, EXP-038) → fewer epochs → under-train → regress (EXP-004/009/015/024/036/038).
2. **Polish-vs-top1** — compute-neutral OPTIMIZATION polish (EMA/SWA/GC/LS/bag-of-tricks) lowers
   loss/flatness not top-1.
3. **Regularizer-adds underfit at 300s** — dropout (EXP-022, −1.37), CutMix (EXP-018, −1.08), Mixup
   (EXP-011) all underfit the convergence-bound recipe. ADDING regularization fails here.

**CLOSED axes (~30)**: capacity BOTH ways (uniform widening EXP-004/009 + FLOP-neutral reallocation
EXP-038); ALL augmentation (strength/policy/mixing/cooldown/border, through EXP-037); ENTIRE LR
schedule; regularizer-adds; architecture (SiLU/preact/ResNet-D/BlurPool/multi-scale-HEAD EXP-032);
optimizer dynamics (GC EXP-030/031) + objective (SAM EXP-036); weight-averaging (EXP-006/019/020);
input std-norm INFEASIBLE (frozen eval; also a near-no-op because conv1→BN absorbs input scale).

**Untried gap**: the **classifier-head decision GEOMETRY**. The head is a plain `nn.Linear(w3,10)` with
bias + softmax-CE (train.py L107, L133). EXP-032 closed the multi-scale *feature-aggregation* head
(concat layer2+layer3 → broke the coarse-to-fine hierarchy, −1.5pp), but the SCORING geometry of the
final projection — normalizing features+weights to an angular (cosine) decision rule — has never been
tested. It is convergence-neutral (no epoch penalty → dodges walls #1 and #3) and not optimization
polish in the EMA/SWA sense (it changes WHERE the decision boundaries sit → affects top-1, not just
loss → dodges wall #2's mechanism). It is the one lever that sidesteps all three walls.

## Candidate Ideas

### 1. Cosine / normalized-softmax classifier head (angular decision geometry)
**Summary**: Replace the plain linear head's scoring with a normalized-cosine rule. Keep
`self.fc = nn.Linear(w3, 10, bias=False)`; in `forward`, after global-avg-pool, compute
`feat = F.normalize(pooled, dim=1)`, `w = F.normalize(self.fc.weight, dim=1)`,
`logits = scale * F.linear(feat, w)` with a fixed `scale` (≈16; large enough that softmax+LS can
saturate over 10 classes). Everything else unchanged. Compute-neutral (two L2 norms on tiny tensors),
convergence-neutral, params −10 (drop fc bias).
**Reasoning**: The net is generalization-bound at fixed capacity; loss is already well-minimized
(polish can't lift top-1) and capacity/regularizer/augmentation are walled. The classifier's SCORING
geometry is the one untouched degree of freedom that directly affects which test points land on the
correct side of the boundary (top-1) WITHOUT adding compute, capacity, or a convergence-slowing
penalty. Normalizing features+weights removes norm-based slack and forces angular separation, which
can regularize the boundary and improve margin/generalization. Distinct from EXP-032 (that bypassed
the feature hierarchy; this preserves it, touching only the final projection's scoring).
**Sources**: normalized-/cosine-softmax literature (CosFace/normalized-softmax); project-insights
(top-1 needs non-polish, non-capacity generalization); train.py L107/L131-133.
**Estimated Effort**: low — head forward change + one `scale` constant. One run.
**Risk Assessment**: MEDIUM. Clean failure mode (no crash, compute-neutral → no epoch confound). Main
risks: (a) `scale` mistuning — too small → softmax can't saturate (under-confident, high loss, regress);
too large → over-confident. Mitigate with scale≈16 (standard for ~10 classes) or a learnable scale
(init 16). (b) Label-smoothing interacts with bounded cosine logits — may slightly de-tune LS, but LS
0.1 is mild. Honest expectation: within-noise on balanced CIFAR-10 (the gain is usually small/robustness-
oriented), but a genuine, mechanism-backed, wall-dodging probe of the last unmapped lever.

### 2. Stochastic Depth (Huang 2016) — mild linear-decay residual drop (p_L≈0.8)
**Summary**: Per-sample drop each BasicBlock's residual f(x) with linear-decay survival 1.0→0.8 over
the 9 blocks (keep the skip path); scale by survival at test. Vectorized masked impl (multiply f(x) by
a Bernoulli mask) to stay compile-friendly.
**Reasoning**: The canonical ResNet regularizer (implicit depth-ensemble) and the last standard recipe
ingredient untested here.
**Sources**: Deep Networks with Stochastic Depth (Huang 2016); project-insights regularizer-underfit.
**Estimated Effort**: medium — masked residual in BasicBlock + per-block survival; cudagraph-RNG care
(reduce-overhead may not advance torch.rand per replay → would need masks passed in from the eager loop).
**Risk Assessment**: HIGH-ish — SD is a convergence-SLOWING regularizer-add, squarely in the wall-#3
pattern that killed dropout (EXP-022, −1.37) and CutMix; at the 300s convergence-bound budget it most
likely under-converges → regress, even at mild p_L. Plus the cudagraph-RNG implementation risk. Lower
EV than #1 and fights the strongest established wall.

### 3. Repeated (batch) augmentation — fewer unique images, more augmented copies per batch
**Summary**: Build each batch from B/2 unique images, each with 2 independent augmentations (Hoffer
2020 "augment your batch"), keeping batch size 128. Compute-neutral.
**Reasoning**: Trades data diversity per step for augmentation diversity; sometimes improves
generalization.
**Sources**: Hoffer et al. 2020.
**Risk Assessment**: HIGH — halves unique images/epoch → effectively fewer image-epochs at the short
budget → under-train (same family as the regularizer/aug-strength failures); also overlaps the closed
augmentation axis. Likely null/negative. Low value.

## Idea Evaluation

After ~30 closed axes the only moves with any remaining shot are those that DODGE all three walls
(compute, polish, regularizer-underfit). Mechanism + wall-avoidance is the deciding lens:

- **#1 (cosine head)** is the only candidate that sidesteps ALL three walls: compute-neutral (no epoch
  hit), convergence-neutral (not a regularizer-add → not wall #3), and it changes the decision geometry
  (top-1-affecting, not loss-only polish → not wall #2). Genuinely unmapped (the head-SCORING axis,
  distinct from the closed feature-aggregation head). Clean, low-effort, clean failure mode. Honest
  ceiling is modest (balanced CIFAR-10 gains from cosine heads are usually small) but it is the
  best-positioned untried lever.
- **#2 (stochastic depth)** has the strongest generalization pedigree but is a convergence-slowing
  regularizer-add → directly in the wall-#3 failure pattern (count ≥2, High) → most likely under-trains
  and regresses at 300s; plus cudagraph-RNG implementation risk. Fighting the best-established wall with
  weak justification → deprioritized.
- **#3 (repeated aug)** reduces unique-images/epoch → under-train at the short budget AND overlaps the
  closed augmentation axis → low value.

**#1 wins**: it is the single remaining lever that targets top-1 (decision geometry) while dodging every
established wall, at near-zero cost and risk. A clean null closes the classifier-scoring sub-lever; a
small gain breaks a 39-experiment plateau.

## Chosen Idea
**Selected**: Cosine / normalized-softmax classifier head — L2-normalize penultimate features and the
final linear weights, score by scaled cosine similarity (`scale≈16`), everything else unchanged.

**Why this idea**:
Every capacity, augmentation, schedule, optimizer, regularizer, and weight-averaging axis is closed, and
the plateau is bounded by three walls (compute, polish-vs-top1, regularizer-underfit). The classifier-head
SCORING geometry is the one untouched, convergence-neutral, compute-neutral degree of freedom that changes
WHERE decision boundaries sit (top-1-affecting, not loss-only) — so it is the only remaining lever that
dodges all three walls. It is low-effort, has a clean failure mode, and is mechanistically distinct from
the closed feature-aggregation head (EXP-032).

**Hypothesis**:
Replacing the plain linear+bias softmax head with a scaled-cosine (feature+weight L2-normalized,
`scale≈16`) head will impose an angular decision geometry that better-conditions class boundaries and
improves margin/generalization, lifting best_test_acc above the bar 96.32 — at throughput-neutral ~91 ep,
params essentially unchanged. Honest most-likely outcome: within-noise (~96.0–96.3), since cosine heads
give only small gains on balanced data and conv1→BN-style normalization already conditions the network;
a clean null then closes the classifier-scoring sub-lever. Key risk to verify: `scale` must be large
enough for softmax+LS to saturate (else under-confidence → regression) — check final_test_loss is not
inflated relative to baseline 0.195.
