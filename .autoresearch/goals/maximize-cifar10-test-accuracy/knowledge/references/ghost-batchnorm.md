# Ghost Batch Normalization (regularizing activation-statistic noise) for fast CIFAR-10

Standing reference for GhostBN — a regularization MECHANISM distinct from input-aug (occlusion/mixing/transform), weight-decay, label-smoothing, and loss-geometry (all saturated on this goal by EXP-015).

## What it is
Instead of computing BN statistics over the full minibatch, split the batch into smaller "ghost" sub-batches (size g) and normalize each by its OWN mean/var during TRAINING. Smaller g → noisier per-group stats → regularizing noise injected into activations. Origin: Hoffer et al. 2017 "Train longer, generalize better" (arXiv:1705.08741). A KEY trick of David Page's "How to train your ResNet" / DavidNet fast-CIFAR recipe — the lineage this goal's net descends from. Typical g: 32–128 within a 512 batch.

## Why it's a fresh axis here
Our net runs batch 512 with FULL-batch `nn.BatchNorm2d` = the LOW-noise BN regime. GBN is the documented way to add the missing large-batch regularization noise. It is mechanistically orthogonal to every saturated regularizer, so the same-axis diminishing-returns law (which closed the input-aug lane after 3 mechanisms) does not directly apply. Composes with the existing weight-EMA (EMA denoises GBN's noisier iterates → captures benefit, cancels eval variance).

## Implementation on THIS harness (the load-bearing correctness points)
- **Drop-in for the BN in `conv_bn`** (7 sites). Module must expose `running_mean`/`running_var`/`weight`/`bias`/`num_batches_tracked` with the SAME names/shapes as `nn.BatchNorm2d` so `AveragedModel(use_buffers=True)` EMA and `channels_last` work unchanged.
- **TRAIN**: reshape [N,C,H,W]→[G,g,C,H,W], normalize each ghost by its own (g,H,W) stats (the regularization noise), then affine γ/β.
- **EVAL-BUFFER UPDATE — critical**: update `running_mean/var` from the **FULL-batch moments** (correct population estimate over all 512), NOT the noisy per-ghost stats. Otherwise EMA averages polluted stats into eval (use_buffers=True). This decouples train-noise from eval-stat-quality.
- **EVAL**: standard BN with the clean (EMA-averaged) running stats — identical path to `nn.BatchNorm2d.eval()`.
- **Equivalence smoke**: g=512 (one ghost) MUST be numerically equal (~1e-3) to `nn.BatchNorm2d` in both train and eval — unit-test before the official run.
- **bf16**: compute stats in fp32 (`.float()`) like BN under autocast.
- **drop_last=True** guarantees N=512 → G=512/g integer for g∈{64,128,256}; assert divisibility, else fall back to full-batch.
- **Throughput**: reshape + grouped stats is near-free → num_epochs should stay ~150 (no under-anneal tax, unlike compute-adding blocks). Verify num_epochs ~142–155.

## Strength
Start g=128 (4 ghosts, mild), then g=64 (8 ghosts). AVOID g=32 first unless ep25 stays healthy (too-small g can over-regularize/destabilize at 150ep). Watch ep25 + full-anneal.

## Status on this goal
Tested **EXP-016** → **no-improvement, but a POSITIVE-signal near-miss**. layer3-only g=128 hit 96.38 @133ep vs same-session standard-BN control 96.14 @149ep (+0.24pp DESPITE −16 epochs; ep25 healthy; fully annealed) — the FIRST regularization mechanism to beat its control on this goal. Missed the 96.48 bar only because of throughput.
**CRITICAL implementation finding (overrides the "near-throughput-free" note above):** ghosting on this harness is NOT throughput-free — it breaks cuDNN's single fused channels_last BN kernel. Measured ~50% slower at ALL sites (manual fp32 AND fused `F.batch_norm` over the ghost-folded view both ~50%; per-site BN forward 4–9× slower, worst at large spatial maps). All-site → ~75ep (halved) → throughput-disqualified at a 300s budget. Only a layer-restricted version (`GHOST_MIN_CH`: 512=layer3-only +12%, 256=layer2+3 +27%) keeps epochs in band; layer3-only is the only config that stayed near 150ep-equivalent.
**Re-entry path:** fund layer3-only GBN with the EXP-014 torch.compile +12% throughput recipe (run ghost noise at ~150ep), or build a fused/compiled ghost kernel that preserves channels_last; then test if the +0.24pp signal clears 96.48 epoch-neutral. See `experiments/016/04-analysis.md`.

## UPDATE — axis CLOSED by EXP-017 (read this before any re-entry)
**EXP-017** tested the GhostBN mechanism THROUGHPUT-FREE via a `NoisyBN` surrogate (fused full-batch BN + per-(sample,channel) post-norm jitter, β-untouched), bypassing the kernel-break tax entirely. Calibrated the layer3 ghost(g=128)-equivalent noise (σ_add*≈0.025, σ_mul*≈0.033, corrected form σ_full/σ_ghost−1 over 6 real batches) → σ_cal=0.033. Result: **cA (σ=0.033) tied the same-session control EXACTLY, 96.14=96.14** @140ep; cB (σ=0.083, 2.5×) only +0.09pp (96.23, within noise, < 96.48). Decisively, **this session's c0=96.14 == EXP-016's c0=96.14** → the ~96.14 same-host control is reproducible and **EXP-016's "+0.24pp" was a low-control-draw artifact**, not a real BN-noise benefit. The mechanism is REDUNDANT with the existing regularization stack at full epochs. **Do NOT re-run GhostBN or NoisyBN at any σ/group-size.** The compile-funded faithful GhostBN is now LOW priority (the surrogate already spanned 1× and 2.5× with no signal); only worth it if a backbone pivot also stalls AND the structured group-shared/data-dependent form is hypothesized to matter beyond magnitude. See `experiments/017/04-analysis.md`.

## Sources
- Hoffer et al. 2017 https://arxiv.org/abs/1705.08741 ; David Page "How to train your ResNet" (myrtle.ai) / DavidNet.
