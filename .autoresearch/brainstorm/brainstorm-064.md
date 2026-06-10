# Brainstorm EXP-064
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- (none consulted — no new external technique. After 64 experiments every literature-sourced lever (auto-aug families, SAM, SWA/EMA, BlurPool/ResNet-D, PolyLoss, GhostBN, LayerScale, AdamW, pre-activation, bag-of-tricks) is closed. This loop probes the last genuinely-untested in-recipe knobs. Gradient-norm clipping is standard, well-understood practice; no high-signal source adds beyond the project's own high-LR experiments EXP-016/017.)

## Experimental History Review
- **Current best: 96.45 (EXP-054)** = k=4 WideResNet-20 + AugMix-p0.5 + GPU Cutout16 + cosine peak0.2/warmup0.05/Nesterov/LS0.1/WD1e-4 + compile, 91 ep, dt 8ms.
- **64 experiments, 8 improvements. The plateau is mapped across EVERY major lever** (project-insights High): augmentation strength/policy/coverage/delivery (CPU+GPU all closed), capacity ×4 directions, optimizer family+gradient-dynamics+objective, LR peak+shape+warmup (EXP-062), normalization-as-regularizer (GhostBN), eval-BN (EXP-061), residual scaling, head, batch (128 optimal), activation, regularizers, weight-averaging, throughput→epochs (saturated ~91). **Both near-miss combinations also closed**: EXP-049 (cooldown+GC), EXP-063 (cooldown-on-AugMix).
- **The box**: the net is simultaneously CAPACITY-bound (can't widen — epoch wall from all 4 directions) AND EPOCH-saturated (~91 ep, more doesn't help — EXP-045/046). The only lever that ever lifted top-1 was more-diverse CPU augmentation (free w.r.t. the Σdt budget), now fully mapped.
- **Genuinely UNTESTED knobs remaining** (all low-ceiling): gradient-norm clipping (never tried); BN momentum/eps (never tuned); **per-channel std normalization — train.py L152-155 uses `std=(1,1,1)`, i.e. data is mean-subtracted but NEVER divided by std** (a real, deliberate code oddity, never tested).

## Candidate Ideas

### 1. Gradient-norm clipping at a permissive threshold
**Summary**: Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)` between `loss.backward()` and `optimizer.step()` (train.py L245-246). Single line; threshold a constant. Clips only outlier gradient spikes, leaving normal steps untouched.

**Reasoning**: This is the cleanest genuinely-untested optimizer-adjacent knob with a concrete (if weak) mechanism. AugMix distorts ~50% of images via multi-chain mixing; heavily-distorted batches produce large-loss → large-gradient spikes, and the recipe holds a high peak LR (0.2) for a long plateau (EXP-016/017 showed 0.2 is at/above the stable optimum — i.e. the net runs near its stability edge). A spike during the high-LR phase can knock converging weights off a good trajectory; a permissive clip bounds exactly those spikes while leaving the ~99% of normal steps unchanged → potentially a marginally better basin. "Trains stably" (no divergence) ≠ "trains optimally" — clipping can still smooth the trajectory near the stability edge.

**Sources**: train.py L245-246 (backward/step); EXP-016/017 (high-LR regime near stability edge); standard practice.

**Estimated Effort**: Trivial (one line). cudagraph-safe — the clip operates on eager `.grad` tensors AFTER backward, OUTSIDE the compiled forward (EXP-042 only broke on branches INSIDE the compiled graph). Throughput-neutral (one extra norm reduction/step, negligible).

**Risk Assessment**: Low, but low-evidence. This net never diverged in 64 runs, so harmful spikes may not exist → most likely an exact null (project-insights: optimizer/gradient-dynamics polish is closed). A too-low threshold would under-step and regress; 2.0 is permissive (clips only true outliers). Worst case a small regression. Near-noise ceiling.

### 2. Per-channel std normalization (std (1,1,1) → CIFAR-10 channel std)
**Summary**: Change train.py L152-155 normalization `std` from `(1,1,1)` to the standard CIFAR-10 per-channel std `(0.2470, 0.2435, 0.2616)`, applied to BOTH the train and eval-feeding transforms (via the shared `mean,std`). Single-variable.

**Reasoning**: The data is currently mean-subtracted but NOT std-normalized — a deliberate code oddity (flagged in the README per the L155 comment). conv1 has NO preceding BatchNorm, so it sees raw inputs with per-channel std ≈ 0.25; Kaiming init assumes unit-variance inputs, so conv1's output variance is ~0.06× expected and its weight-gradient scale is suppressed → conv1 effectively trains at a reduced rate. Proper std normalization restores unit-variance inputs and conv1's intended learning dynamics.

**Sources**: train.py L152-155 (normalization), L102 (conv1, no preceding BN), L110-115 (Kaiming init); standard CIFAR-10 preprocessing.

**Estimated Effort**: Trivial (one tuple). Compute-neutral, throughput-neutral, params unchanged. NOTE: the same normalization feeds eval (prepare.py/Eval uses the test transform built from the same stats — must confirm the eval path uses the train.py-defined normalization, not a frozen separate one; if eval normalization is frozen/separate, a train-only std change would create a train↔eval input mismatch and must be discarded).

**Risk Assessment**: Low-to-moderate. The effect is confined to one 3→16 conv (bn1 immediately renormalizes the forward pass downstream), so the impact is marginal → likely near-null. KEY RISK: if the eval transform is frozen and independent of train.py's `mean,std`, changing train std alone introduces a train/eval distribution mismatch (→ regression / invalid). Must verify eval normalization is shared before running.

### 3. BN momentum reduction (0.1 → 0.03)
**Summary**: Set `momentum=0.03` on all `BatchNorm2d` constructors (longer EMA window for running mean/var) so eval-time running stats average over more batches. Single-variable.

**Reasoning**: Under heavy AugMix, per-batch BN stats are noisy; a longer EMA window could give running stats that better represent the steady-state distribution the eval forward uses.

**Sources**: train.py BN constructors (L71/75/83/103); EXP-061 (BN-stat operating point).

**Estimated Effort**: Trivial (momentum kwarg).

**Risk Assessment**: Low-to-moderate, low-evidence, and EXP-061 weakly contraindicates: it showed the augmented running stats ARE the correct trained-in operating point. With a cosine-annealing LR the distribution shifts through training; a longer EMA window LAGS that shift → running stats at eval could be staler → mild regression possible. Near-noise.

## Idea Evaluation
- **Evidence strength**: all three are low-evidence probes on an exhausted plateau (this is the honest state at experiment 64). Idea 1 has the clearest mechanism (high-LR stability edge + AugMix spikes) and is the cleanest untested optimizer knob. Idea 2 targets a real code oddity but is confined to one tiny layer and carries an eval-mismatch risk. Idea 3 is weakly contraindicated by EXP-061.
- **Mechanism clarity**: Idea 1 clear-but-weak (spike-taming near the stability edge); Idea 2 clear-but-tiny (one conv's early dynamics) + a sharp downside risk; Idea 3 plausible-but-EXP-061-argues-against.
- **Expected impact**: all near-noise. Idea 1 affects all params during the long high-LR phase (broadest mechanism); Ideas 2/3 are narrow.
- **Risk profile**: Idea 1 safest — throughput-neutral, cudagraph-safe, no eval-path interaction, cleanest no-improvement failure mode. Idea 2 has a real invalid/regression risk (eval normalization mismatch) unless verified. Idea 3 mild-regression-leaning.
- **Feasibility**: all trivial.
- **Conclusion**: Lead with **Idea 1 (gradient-norm clipping)** — the cleanest genuinely-untested knob, broadest mechanism, safest failure mode, zero eval-path risk. Honest expectation: near-noise null on a deeply-mapped plateau (this net trains stably), run per NEVER-STOP to definitively close the gradient-clipping lever. If null, Idea 2 (after verifying the eval normalization path) is the next probe, then a documented radical-architecture attempt.

## Chosen Idea
**Selected**: Gradient-norm clipping at a permissive threshold (max_norm=2.0).

**Why this idea**: It is the cleanest genuinely-untested knob remaining, with the broadest and clearest mechanism of the three (it acts on every parameter during the long high-LR plateau where EXP-016/017 placed the recipe at/above its stability edge, taming AugMix-induced gradient spikes), the safest failure mode (throughput-neutral, cudagraph-safe, no interaction with the frozen eval path — unlike Idea 2's eval-mismatch risk), and trivial effort. On a plateau where every major lever and both near-miss combinations are closed, definitively closing the last clean optimizer knob is the most defensible next probe.

**Hypothesis**: Clipping gradient norm at a permissive 2.0 will tame the largest AugMix-induced gradient spikes during the high-LR plateau and smooth convergence to a marginally better basin, raising best_test_acc to ≥ 96.55 (baseline 96.45 + 0.1pp). Given the net never diverged across 64 runs (so harmful spikes may be rare) and optimizer/gradient-dynamics polish is a closed family, the most likely outcome is a within-noise null, but the probe is trivial, throughput-neutral, and the cleanest untested knob.
