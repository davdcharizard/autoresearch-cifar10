# Proposal idea-01: Throughput-free BN-affine noise (cheap GhostBN surrogate)

## Core change (train.py only)
Wrap the BN in `conv_bn` with a `NoisyBN` that, during TRAINING only, applies standard FULL-batch fused BN (no speed loss) and then perturbs the normalized activations with per-(sample,channel) Gaussian noise:
`y = BN(x) * (1 + σ_mul·ε1) + σ_add·ε2`, ε1,ε2 ~ N(0,1) drawn with shape [N,C,1,1] (broadcast over H,W). Env `BN_NOISE` (=σ, default 0 = exact EXP-008 baseline). Eval path = identical to `nn.BatchNorm2d` (noise off when `not self.training`); EMA(use_buffers) and running stats untouched.

## Mechanism — why this is the SMART version of EXP-016
EXP-016 produced the FIRST positive regularization signal on this goal: layer3-only GhostBN beat its same-session control by +0.24pp (96.38 vs 96.14) DESPITE 16 fewer epochs. But GhostBN's grouping breaks cuDNN's fused channels_last BN kernel (~50% slower all-site → halved epochs → under-anneal — the #1 failure mode here, project-insights High). GhostBN's regularization IS exactly a per-(sample,channel) perturbation of the normalization: normalizing a sample by a noisy sub-batch's mean μ̃ and std σ̃ instead of the full-batch μ,σ is algebraically `BN(x)·(σ/σ̃) + (μ−μ̃)/σ̃` — a per-channel multiplicative + additive jitter. `NoisyBN` injects that jitter DIRECTLY as a single elementwise op on the BN output, keeping the fused kernel → ~150 epochs (no under-anneal). This decouples the regularization mechanism (which showed signal) from the throughput tax (which capped it).

## Why it targets the limiter
The limiter is the ~96.3–96.5 generalization ceiling (project-insights High, EXP-014). EXP-016 showed BN-statistic noise is the ONE regularization axis not yet saturated (input-aug/wd/LS/loss-geometry all tied). This tests whether that axis clears the bar when applied at full epochs — the decisive throughput-free experiment EXP-016 could not run.

## Design — SAME-SESSION multi-cell (REFINED layer3-first per cross-model review)
Add `BN_NOISE_MIN_CH` (apply noise only at BN sites with num_features ≥ this; default 0 = all sites) to mirror EXP-016's proven layer3 (C=512) signal site.
- c0: `BN_NOISE=0` (standard BN, full-speed control ~150ep) — the same-session anchor (stored 96.38 too weak at the noise floor; EXP-016's c0 drew low at 96.14).
- cA: `BN_NOISE=0.10, BN_NOISE_MIN_CH=512` (layer3-only, mild) — the FAITHFUL reproduction of the EXP-016 site, now epoch-neutral. PRIMARY cell.
- cB: `BN_NOISE=0.20, BN_NOISE_MIN_CH=512` (layer3-only, stronger) OR a mild all-site arm (`BN_NOISE=0.10, BN_NOISE_MIN_CH=0`) — pick the second operating point in the plan; an all-site arm only if the throughput smoke confirms ~150ep.
Each a separate `train.py` process, `CUDA_VISIBLE_DEVICES=1 timeout 600`, nvidia-smi logged. Expect num_epochs ~149 (throughput-free — VERIFY; reject <142). Optional refinement: calibrate σ from the measured full-batch-vs-ghost(g=128) per-channel stat ratio at layer3 instead of a fixed σ.

## Correctness / EMA
- Train-only: noise disabled in eval (`self.training` guard) → eval bit-identical to standard BN.
- EMA: the existing `AveragedModel(use_buffers=True)` averages clean running stats (updated by the fused BN as usual; noise is applied AFTER, not to the buffers) → eval stats clean, and EMA further averages the noisy iterates (denoises, captures benefit) — same synergy GhostBN relied on.
- bf16/channels_last: elementwise multiply/add preserve memory format; draw noise in the activation dtype.
- σ=0 ⇒ no-op ⇒ exact baseline (regression guard, smoke).

## Verification
- Best noise cell ≥ **96.48** AND > same-session c0 by >0.1pp (clear of noise floor). Hairline (<0.15pp) → confirmation re-run.
- num_epochs ~142–155 (throughput-free check); ep25 within ~0.5pp of c0 + fully annealed (over-reg watch at σ=0.2).
- Integrity: train.py-only; prepare.py byte-unchanged; ≤1 eval/epoch; seed 42; σ=0≡baseline smoke.
- ON A WIN: bake winning σ as default.

## Hypothesis
Per-(sample,channel) BN-affine Gaussian noise at σ≈0.1–0.2, composed with the existing EMA, injects the EXP-016 ghost-stat regularization at FULL ~150 epochs and lifts best_test_acc ≥96.48 over the same-session control. If all cells tie at healthy epochs/ep25, BN/activation-statistic noise is redundant with the existing stack at full epochs → the EXP-016 +0.24pp was a low-c0-draw artifact, and the ceiling is not noise-movable (strengthens the backbone-pivot mandate).

## Effort: low. Risk: noise form is a surrogate (post-BN affine jitter ≈ but ≠ exact ghost-stat perturbation); could tie (mechanism redundant) or under-fit at σ=0.2. Throughput-free so no under-anneal confound. Fallback if it ties: the faithful-but-costly compile-funded layer3 GhostBN (idea uses EXP-014 compile to recover throughput).
## Sources: experiments/016/04-analysis.md (+0.24pp signal, throughput tax); knowledge/references/ghost-batchnorm.md; project-insights High (under-anneal #1 failure, generalization ceiling); Hoffer 2017 arXiv:1705.08741.
