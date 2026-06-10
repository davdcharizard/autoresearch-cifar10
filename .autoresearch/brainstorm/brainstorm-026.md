# Brainstorm EXP-026
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **He et al. — "Bag of Tricks for Image Classification with CNNs" (CVPR 2019)** (knowledge/papers/bag-of-tricks.md; arXiv:1812.01187)
  Two FREE (compute-neutral, param-neutral) convergence tricks remain untried on our recipe: **zero-init residual γ** (initialize the last BN's γ to 0 in each residual block so the block starts as an identity map → easier early optimization/signal propagation) and **no-bias-decay** (apply weight decay ONLY to conv/linear weights, never to BN γ/β or biases). Both give small consistent gains on ResNets; the paper's compute-ADDING tricks (ResNet-B/C/D downsample, mixup, large-batch LR scaling) are either already tried-and-failed here (EXP-011/018/025) or epoch-wall-risky (EXP-024).

## Experimental History Review

Current best **96.22%** (EXP-012, commit 6c417a4); bar = 96.32 (+0.1pp). 25 experiments; the plateau is exhaustively mapped (~18 axes closed):

- **All scalar knobs bracketed interior optima**: LR-peak 0.2 (EXP-016/017), Cutout 16px (EXP-013/021), label-smoothing 0.1 (EXP-023), weight-decay (EXP-005), and now **batch size 128** (EXP-025).
- **Augmentation family closed**: policy saturated (TA≈RA), label-mixing underfits (Mixup/CutMix), occlusion optimal at 16.
- **Adding regularizers closed**: WD↑/Mixup/CutMix/dropout all regress — recipe is **convergence-bound, not overfit-bound** (project-insight Medium).
- **Compute-adding structural changes closed**: k≥5 width, pre-act, BlurPool all hit the **epoch wall** (project-insight High, count 4).
- **Throughput/batch closed**: EXP-025 — k=4 is launch-bound ONLY at batch 128; batch 256 is compute-bound, collapsed updates 61%, −2.38pp. New project-insight: launch-bound is batch-dependent.
- **Weight-averaging closed** (EMA/SWA, count 3).

**Untried gaps:** the remaining moves are all COMPUTE-NEUTRAL convergence-quality levers (the only safe class for a convergence-bound recipe with no epoch-wall risk): (1) **Bag-of-Tricks free levers** — zero-init residual γ + no-bias-decay (NEVER tried; distinct from scalar-knob tuning and from adding regularizers); (2) **per-channel input std-norm** (expected BN-absorbed null); (3) **gradient-norm clipping** (untried convergence-stability lever).

Codebase state (confirmed this loop): `BasicBlock.bn2` (train.py L75) is the last BN in each block; `ResNet.__init__` applies `_weights_init` (kaiming) at L108; optimizer is a single-group `optim.SGD(model.parameters(), weight_decay=1e-4, nesterov=True)` (L192-198) — so WD currently hits BN γ/β and the fc bias too.

## Candidate Ideas

### 1. Bag-of-Tricks free convergence bundle (zero-init residual γ + no-bias-decay)
**Summary**: Two compute-neutral, param-neutral best-practice changes (He et al. 2019). (a) **Zero-init residual γ**: after `self.apply(self._weights_init)` in `ResNet.__init__`, loop the modules and `nn.init.zeros_(m.bn2.weight)` for every `BasicBlock`, so each residual branch outputs 0 at init and the block starts as identity (the 6 identity-shortcut blocks become exact identity; the 3 projection-shortcut blocks become relu(BN(proj(x)))). (b) **No-bias-decay**: rebuild the SGD optimizer with two param groups — weight_decay=1e-4 for conv/linear weights (ndim≥2), weight_decay=0 for BN γ/β and biases (ndim≤1). Everything else byte-identical (LR 0.2, batch 128, TA+Cutout(16), cosine-to-0, 91 epochs).

**Reasoning**: The recipe is convergence-bound, so convergence-quality levers that add NO compute are the correct class — they keep the 91-epoch baseline and carry no epoch-wall / update-collapse risk (the two failure modes that killed EXP-004/009/015/024 and EXP-025). Both tricks are documented, standard, and genuinely untried here. Zero-γ eases early optimization (cleaner residual signal at init); no-bias-decay stops the small but real mistake of L2-penalizing BN scale/shift. Bundling the two synergistic free tricks maximizes the chance of clearing the +0.1pp bar in one fair run.

**Sources**: He et al. "Bag of Tricks" (knowledge/papers/bag-of-tricks.md); train.py L75/L108/L192-198; project-insights (convergence-bound; compute-neutral is the safe class); goal-learnings (scalar knobs all bracketed → pivot off scalar tuning).

**Estimated Effort**: low (a small init loop + a param-group split; one 300s run).

**Risk Assessment**: ResNet-20 is shallow (9 residual blocks) so zero-γ's benefit (largest for deep nets) may be marginal; no-bias-decay's effect is small because WD is only 1e-4. Combined gain may land below the ~0.2pp noise floor → no-improvement. Bundling muddies attribution if it gains (mitigated: a gain triggers a one-run ablation next loop). No regression risk beyond noise; compute-neutral so epochs stay ~91 (verify) and no confound.

### 2. Per-channel input std-normalization
**Summary**: Change `std` in `transforms.Normalize` from `(1,1,1)` to the true CIFAR-10 per-channel std `(0.2470, 0.2435, 0.2616)` (mean already subtracted), giving unit-variance per-channel inputs.

**Reasoning**: The only untouched scalar in the input pipeline; closes the input-normalization axis cleanly. Compute-neutral, zero confound.

**Sources**: train.py L152-155 (the `std=(1,1,1)` comment flags it); project-insights/memory (flagged as the last cheap probe, expected BN-absorbed null).

**Estimated Effort**: low (one-line edit; one run).

**Risk Assessment**: Expected NULL — the stem conv + first BatchNorm absorb any global per-channel affine input rescaling, so accuracy should be unchanged within noise. Low expected value; primarily an axis-closer.

### 3. Gradient-norm clipping
**Summary**: Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` (e.g. max_norm=5.0) between `loss.backward()` and `optimizer.step()`, clipping the global grad norm to stabilize the occasional large-gradient step that strong TrivialAugment hard examples can produce.

**Reasoning**: Compute-neutral, untried convergence-stability lever; can slightly improve the optimization trajectory under strong augmentation. Distinct from every closed axis.

**Sources**: train.py L237-238 (backward/step site); standard practice for stabilizing SGD with aggressive augmentation.

**Estimated Effort**: low (one line; one run).

**Risk Assessment**: With BN + warmup + bf16, gradients are already well-behaved, so clipping at a loose norm likely no-ops (null); a too-tight norm could slow convergence (mild regression). Low expected value; mainly a probe. A `.item()`-free op so no throughput hit.

## Idea Evaluation

All three respect the hard constraints (train.py-only, no new deps, single GPU, ≤1 eval/epoch, no seed hacking, <600s) and none retries a logged failed approach — zero-γ, no-bias-decay, input-std, and grad-clip are all untried.

- **Evidence strength**: Idea 1 has the strongest literature (canonical Bag-of-Tricks levers validated across ResNets) AND fits the project's hardest-won insight (convergence-bound → compute-neutral is the only safe class). Idea 2 is an expected null. Idea 3 has weak evidence it helps a well-behaved BN+warmup run.
- **Mechanism clarity**: Idea 1 is crisp (identity-init blocks ease early optimization; stop L2-penalizing BN scale). Idea 2 predicts no change (BN absorbs it). Idea 3's mechanism only bites if large gradients actually occur, which is doubtful here.
- **Expected impact**: Idea 1 highest — it's the only remaining lever with a documented positive effect AND no epoch-wall/update-collapse risk. Ideas 2-3 are closers/probes with expected ~null.
- **Risk profile**: All fail gracefully to no-improvement; all compute-neutral (epochs stay ~91, no confound). Idea 1 is the safest high-information bet.
- **Feasibility**: All low-effort. Idea 1 is slightly more code (init loop + param groups) but trivial.

Idea 1 dominates on evidence, mechanism, and expected impact, and is precisely the compute-neutral class the convergence-bound diagnosis points to. Ideas 2 and 3 are reserved as the final axis-closers.

## Chosen Idea
**Selected**: Bag-of-Tricks free convergence bundle (zero-init residual γ + no-bias-decay)

**Why this idea**:
After EXP-025 closed batch size, every scalar knob is bracketed and every compute-adding or regularizer-adding change has failed — leaving compute-NEUTRAL convergence-quality levers as the only class with a defensible positive mechanism and zero epoch-wall risk. The two Bag-of-Tricks free levers are the canonical, well-evidenced, genuinely-untried members of that class. They keep the fair 91-epoch baseline (no throughput confound) and target the binding convergence constraint directly.

**Hypothesis**:
Initializing each BasicBlock's `bn2.γ` to 0 (identity-init residual branches) and excluding BN γ/β + biases from weight decay improves early optimization and the effective regularization of the convergence-bound recipe, lifting `best_test_acc` above the 96.32 bar at an unchanged ~91 epochs / 4,299,866 params / <600s wall-clock. Falsifiable: if the bundle lands within ±0.2pp of 96.22 (noise), the free tricks are too marginal on this shallow net and the input-norm/grad-clip closers are next; a regression would indicate identity-init or the WD-group change is actively harmful here.
