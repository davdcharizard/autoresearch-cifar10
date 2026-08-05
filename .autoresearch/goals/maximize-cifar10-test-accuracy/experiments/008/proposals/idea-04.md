# Proposal idea-04: GELU (smooth-activation) recipe alignment

## One-line summary
Replace `nn.ReLU(inplace=True)` with `nn.GELU()` in `conv_bn` (the activation used by
every conv block in the net), aligning the activation with the fast-CIFAR lineage
(hlb-CIFAR10 / airbench, arXiv:2404.00498, whose ConvGroup is
`Conv → MaxPool → BatchNorm → GELU → Conv → BatchNorm → GELU`). This is a
single-variable activation swap on the EXP-004 baseline (96.00%).

**Honest verdict up front:** this is a *defensive / low-expected-value* probe, not a
high-conviction win. On a net that is already saturated/regularization-bound at ~96.0%,
the documented effect of ReLU→GELU is typically well under the ~0.1pp noise floor, and
the swap carries a *real* per-step throughput cost that can push the run into the
recurring under-anneal failure mode (EXP-005, EXP-007) and turn the result *negative*.
The value of running it is mostly to (a) close the "we never tried the lineage's
activation" loop with a measurement, and (b) cheaply produce a GELU-baseline that a
future capacity win can ride on. If the loop wants a >0.1pp swing, a milder capacity
step (256→320) is the better bet; this idea is the safe, low-effort alternative.

## Target metric and limiter
- Metric: CIFAR-10 `best_test_acc` (%, higher better), fixed 300s training budget.
- Baseline: **96.00%** (EXP-004, commit ae31206). Bar: ≥96.10% AND clearly above the
  ~0.1pp noise floor.
- Named limiter (diagnosis): the net is **saturated / regularization-bound** — it fits
  ~142 epochs in 300s vs airbench96's 37 for the same ~96%, so epochs are *not* the
  scarce resource in the abstract, but **per-step throughput drives an epoch-count
  jitter that is itself the dominant noise source** (EXP-006: identical code fit 142 vs
  150 epochs; noise floor ~0.1pp; Protocol Findings High). The recurring concrete
  failure is **under-annealing**: any change that cuts epochs truncates the low-LR tail
  where most accuracy is set (EXP-005 142→131, EXP-007 150→94, both negative). This
  idea's mechanism must therefore clear two bars at once: a positive activation effect
  *and* a throughput cost small enough not to trigger under-anneal.

## Mechanism (causal chain, honest)
ReLU is `max(0,x)`; GELU is `x·Φ(x)` (Φ = standard-normal CDF), a smooth gated
activation. Two candidate mechanisms by which GELU could lift final accuracy:

1. **Smoother gradients / non-zero gradient for small-negative pre-activations.**
   ReLU has a hard kink at 0 and exactly-zero gradient for `x<0`; GELU passes a small
   gradient through slightly-negative units, which can improve optimization of the
   convex-ish low-LR tail and slightly reduce dead-unit pathologies. On a fully-annealed
   one-cycle this is a second-order effect.
2. **Mild implicit regularization.** GELU's soft gate behaves like a smoothed
   ReLU+dropout-ish nonlinearity; on a regularization-bound net this *could* nudge the
   generalization floor. But this net is already heavily regularized (Cutout-8, LS-0.2,
   wd-5e-4, EMA), so the marginal regularization headroom is small.

The causal chain to the metric: GELU → marginally better-optimized / slightly-better-
regularized iterate at the same epoch count → higher annealed test accuracy. The chain
is real but **thin**: published ReLU→GELU deltas on small conv nets are typically
0.0–0.2pp, and the lineage that *chose* GELU also co-tuned init, LR, and width around
it, so the isolated single-variable effect on a ReLU-tuned recipe is most likely
sub-noise here. I am not claiming a confident +0.1pp; I am claiming a measurement with a
plausible-but-small upside and a quantifiable downside.

## Concrete code change (in THIS codebase)
Single edit to `conv_bn` at `train.py:101-106`:

```python
def conv_bn(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.GELU(),                       # was: nn.ReLU(inplace=True)
    )
```

That is the entire functional change. `conv_bn` is the sole activation site that
matters — every block in the net is built from it: `self.prep` (`train.py:148`), all of
`layer1/2/3` (`train.py:149-151`, including the `Residual`/`GatedResidual` inner
`c1/c2`). The only activation-bearing module outside `conv_bn` is nothing — the
`whiten` conv, `MaxPool2d`, `pool`, and `fc` have no activation. So one line covers the
whole net. The final pre-`fc` path is just `pool(x).flatten(1) → fc → *scale_out`
(`train.py:177-178`); there is no activation there to change.

Recommended variant: use `nn.GELU(approximate='tanh')` (the tanh approximation) rather
than the default exact erf form, to minimize the throughput hit (see below). Pick ONE
and keep it fixed for the run; do not sweep both in a single experiment.

Notes / non-changes:
- `nn.GELU` has **no `inplace` argument**, so the activation output is a fresh tensor
  rather than an in-place overwrite. This raises activation memory modestly. VRAM is a
  non-constraint here (1635 MB used of ~98 GB at batch 512; Patterns/Medium), so this is
  irrelevant to feasibility.
- **Leave `kaiming_normal_(nonlinearity="relu")` init unchanged** (`train.py:160`).
  Rationale below — keep the experiment single-variable.
- **ReZero identity-init is unaffected.** `GatedResidual.alpha=0` (`train.py:134`) makes
  the layer2 block exact identity at init regardless of the activation inside `c1/c2`,
  so the identity-start property that made EXP-004 work is preserved. (The docstring at
  `train.py:120-128` reasons about `ReLU'(0)=0` to explain why a *zeroed-BN-γ* identity
  trick would fail and why ReZero is needed instead; that reasoning is about the BN-γ
  alternative, not about ReLU specifically, so it remains valid — ReZero still gives a
  live `∂L/∂α` gradient with GELU.)
- Everything else byte-identical: PEAK_LR=0.4, schedule, wd, LS, Cutout-8, EMA-0.998,
  flip-TTA gate, batch 512, seed 42, whitening. Clean A/B vs EXP-004.

## Throughput analysis (the load-bearing risk)
This is where the idea lives or dies, because epoch count is the dominant noise/failure
axis (EXP-005/006/007). The honest question: **does GELU cut epochs enough to
under-anneal?**

What we know from the codebase and prior runs:
- The net is **conv-bound**, not activation-bound. It has ~8M params and runs ~10 conv
  layers; the dominant cost per step is the 3×3 cuDNN convolutions and BN, under bf16
  autocast + channels_last on the H20 (EXP-004 ran ~26k img/s). Pointwise activations
  are a small fraction of wall time.
- GELU is more expensive *per activation element* than ReLU (ReLU = a single `max`;
  exact GELU = erf-based; tanh-GELU = a tanh + a few mults). But activations are applied
  to feature maps that are also produced/consumed by the far-costlier conv+BN around
  them, so the *marginal* wall-time fraction of swapping the activation kernel is small.
- Under bf16 autocast, the activation runs in low precision and is memory-bandwidth-
  bound (elementwise), so the extra arithmetic of GELU vs ReLU is partly hidden behind
  the memory traffic that ReLU already pays.

**Estimate:** I expect the per-step slowdown to be **small — order 1–4%** (exact-erf
GELU toward the high end, tanh-GELU toward the low end), translating to roughly
**142 → ~136–140 epochs** at the EXP-004 operating point. That is materially *less*
disruptive than EXP-005 (−11 epochs) or EXP-007 (−48 epochs). It is, however, **not
zero**, and on a shared host (GPU 0 busy) the measured epoch count already swings ±~8
epochs run-to-run (142↔150), so a small GELU tax can stack adversarially with host
contention into a worse-than-expected epoch count.

**Why I am not certain it's <5%:** I have not profiled `nn.GELU` vs `nn.ReLU` on this
exact net under bf16 on the H20, and PyTorch's autocast does not always fuse the
activation into the surrounding kernels, so a non-fused exact-erf GELU could be costlier
than the back-of-envelope suggests. This uncertainty is the reason to (a) prefer
`approximate='tanh'`, and (b) read `num_epochs` as the first-class falsifier (below).

## Init interaction (and why to leave it alone)
`kaiming_normal_(nonlinearity="relu")` uses gain √2, the variance-preserving gain for
ReLU. GELU's ideal gain is slightly different (GELU passes a bit less variance than ReLU
for the same input, so the variance-preserving gain is marginally larger), but:
- The net has a **BatchNorm immediately after every conv** (`conv_bn`), and the whitening
  front-end normalizes the input. BN re-normalizes activation statistics every forward,
  so the network is highly insensitive to the exact init gain — the init scale is
  washed out after the first BN. The kaiming gain mismatch is therefore second-order.
- Changing init *and* activation together would make the experiment two-variable and
  un-interpretable against the noise floor. **Keep init fixed** for a clean single-
  variable read. If GELU shows a near-bar positive signal, a gain retune is a cheap
  follow-up; if it's sub-noise, the gain would not rescue it.

## Combine or keep single-variable?
**Keep strictly single-variable.** The brainstorm history pairs "GELU + cutout12" as a
recipe-alignment bundle, but bundling two sub-noise levers makes attribution impossible
against a 0.1pp floor and risks one masking the other (e.g. cutout12 changing the
regularization point while GELU changes optimization). Run GELU alone vs the EXP-004
baseline. Do **not** also bump PEAK_LR, change init, or alter Cutout in the same run.

## Evidence
- **Lineage uses GELU (primary citation).** airbench (Keller Jordan, arXiv:2404.00498,
  "94% on CIFAR-10 in 3.29 Seconds") and hlb-CIFAR10 both use GELU in the conv block;
  the airbench ConvGroup is `Conv(3×3) → MaxPool → BatchNorm → GELU → Conv(3×3) →
  BatchNorm → GELU`, i.e. GELU directly after each BatchNorm — exactly the slot our
  `conv_bn` occupies with ReLU. Source:
  https://github.com/KellerJordan/cifar10-airbench and the arXiv paper. This is the
  motivating evidence that the swap is "recipe-aligned," but note airbench co-tuned its
  init/LR/width/whitening around GELU rather than dropping GELU into a ReLU recipe.
- **Code facts (read):** `conv_bn` is the only activation site (`train.py:101-106`);
  it feeds every block (`train.py:148-151`); `_weights_init` hardcodes `nonlinearity=
  "relu"` (`train.py:157-160`); the post-pool path has no activation (`train.py:177-178`);
  ReZero `alpha=0` gives identity-at-init independent of the activation (`train.py:130-137`).
- **Prior-experiment evidence on the risk axis:** the noise floor is ~0.1pp and is
  *driven by epoch-count jitter* (Protocol Findings High; EXP-006). Under-anneal from
  lost epochs is the recurring failure (Failed Approaches Medium; EXP-005, EXP-007).
  These say: a sub-5% throughput tax is *probably* survivable but a measured epoch drop
  below ~130 with accuracy still climbing at the tail = the kill signal.
- **What we DON'T have:** no prior in-repo measurement of GELU vs ReLU; no profiled
  throughput number. Those are exactly what this experiment produces.

## Expected magnitude vs noise floor (honest)
- Most likely outcome (my central estimate): **sub-noise**, i.e. result lands in
  [95.90, 96.05], indistinguishable from baseline → **no-improvement**. The activation
  effect on a saturated, BN-normalized, heavily-regularized net is small, and a small
  epoch loss can cancel any tiny positive.
- Upside tail: if GELU's smoother optimization buys a genuinely lower annealed loss AND
  throughput holds within ~2%, it could reach ~96.10–96.15 and clear the bar. I put this
  at maybe 15–25% probability.
- Downside tail: if the throughput tax is larger than expected (non-fused exact-erf GELU
  + host contention), epochs drop toward ~125–130 and the truncated tail yields
  ~95.8–95.9, **negative** — the EXP-005/007 failure mode again.

This is a coin-flip-toward-null probe, not a confident win. Recommend `approximate='tanh'`
specifically to shift mass out of the downside tail.

## Verification / falsification (pre-registered)
- **Primary:** `best_test_acc` ≥ 96.10 AND visibly above the ~0.1pp floor → improvement.
- **First-class diagnostic — `num_epochs`:** read it on every run.
  - If epochs ≥ ~135 and accuracy < 96.10 → the activation effect itself is sub-noise
    (clean null on GELU; the lineage's GELU advantage does not transfer to a ReLU-tuned
    recipe). Conclude GELU alone is not a lever here; do not retry standalone.
  - If epochs ≤ ~128 with the tail still monotonically climbing at budget end
    (best ≈ final, as in EXP-007) → **under-anneal**, NOT an activation verdict. The
    throughput tax was larger than projected; retry with `approximate='tanh'` if the
    first run used exact erf, else shelve.
- **Throughput check:** compare steady `img/s` and final `num_epochs` to EXP-004
  (~26k img/s, 142 epochs). This isolates "GELU didn't help" from "GELU cost too many
  epochs."
- **Scope/integrity (standard):** only `train.py` changed (diff = the one activation
  line), `prepare.py` byte-unchanged, seed 42 unchanged (no seed hacking), ≤1 eval/epoch,
  `training_seconds ≥ 295`, `total_seconds < 600` under `timeout 600`,
  `CUDA_VISIBLE_DEVICES=1`.

## Strongest risk / assumption that must hold
The single assumption that most needs to hold: **the GELU per-step throughput tax is
small enough (<~5%, ideally <~2%) that the run still fits ~135+ epochs and anneals its
tail.** If that holds, the worst case is a clean sub-noise null; if it fails (non-fused
exact GELU kernel under bf16 + shared-host contention), the experiment reproduces the
under-anneal failure and reports a false-negative on the activation itself. Mitigation:
use `nn.GELU(approximate='tanh')` and treat `num_epochs` as the gating diagnostic.

## Effort
**Low.** One-line change in `conv_bn`, no new deps (torch-only), no init/schedule retune,
no smoke-test subtleties beyond confirming shapes are unchanged (they are — GELU is
shape-preserving). One training run within the existing harness. The main "work" is
honest interpretation of `num_epochs` to separate the activation verdict from the
throughput verdict.

## Sources
- [94% on CIFAR-10 in 3.29 Seconds on a Single GPU (arXiv:2404.00498)](https://arxiv.org/abs/2404.00498)
- [KellerJordan/cifar10-airbench (GitHub)](https://github.com/KellerJordan/cifar10-airbench)
