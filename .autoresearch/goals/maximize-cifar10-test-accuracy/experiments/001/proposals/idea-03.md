# Proposal idea-03: Airbench-style whitening + GELU-ConvGroup net (path to 95-96%)

## Summary

Replace the ResNet-20 in `train.py` wholesale with the **cifar10-airbench /
hlb-CIFAR10** architecture family (Keller Jordan; tysam-code), which holds CIFAR-10
speed records (94% in ~3.8s, 96% in ~35s on an A100). The architecture is a small,
very wide 3-block conv net whose first layer is a **fixed (frozen) whitening
convolution** initialized from the eigendecomposition of training-image patches,
followed by **GELU ConvGroup** blocks (Conv -> MaxPool -> BN -> GELU, with an
optional residual pair of convs in the high-accuracy variant), a flatten-and-linear
head, trained with SGD+Nesterov, a **triangular one-cycle LR**, **label smoothing**,
bf16 autocast + channels_last, flip+translate augmentation, and **horizontal-flip
test-time augmentation done inside `forward`**.

The whitening front-end is the load-bearing mechanism: it decorrelates the input so
the network converges in ~10-40 epochs instead of the ~50-70 a ResNet-20 needs,
which is exactly what lets a higher-capacity net reach 94-96% inside a fixed time
budget. The airbench reference reaches **96.03% in 34.7s on an A100** (airbench96,
37 epochs). Our budget is **300s on an H20** — roughly an order of magnitude more
wall-clock — so even with the H20's much weaker bf16 compute throughput the
96%-recipe should fit. This is the highest-upside, highest-implementation-risk
candidate of the set.

## What it targets (limiter from the diagnosis)

The named limiter is that the **baseline ResNet-20 recipe converges too slowly to
exploit the 300s budget**: it is a thin (16/32/64-channel) net trained with a step
LR schedule that needs ~64k steps / many epochs to anneal, so within 300s it lands
at 91.57% — under-trained relative to what CIFAR-10 allows. The airbench recipe
attacks this on three fronts simultaneously:

1. **Whitening front-end** removes the low-level decorrelation work the first conv
   layers normally have to learn, so loss drops in the first epoch and the LR can be
   pushed much higher (myrtle.ai "How to Train Your ResNet" + hlb-CIFAR10 both
   attribute the bulk of their speedup to this). This converts "wall-clock budget"
   into "accuracy" far more efficiently than depth.
2. **Width over depth**: 64/256/256-channel ConvGroups are far higher-capacity than
   ResNet-20's 16/32/64 but only 3 pooling stages deep, so each step is cheap and
   the net saturates accuracy in tens (not hundreds) of epochs.
3. **One-cycle triangular LR + label smoothing + flip/translate aug** are the
   standard fast-convergence stack that fits the available step count exactly.

Net effect on `best_test_acc`: the architecture+recipe is the proven path from ~91%
to 94-96% under a tight budget. This is the only candidate with documented headroom
to ~96%.

## Exact `train.py` changes

All changes are confined to `train.py`. `prepare.py` (eval + 300s budget) is
untouched. Normalization stays mean=(0.4914,0.4822,0.4465), std=(1,1,1) for both the
train transform and the patch statistics (see "Normalization consistency").

### 1. Whitening conv initialization (verified pure torch/numpy)

The airbench algorithm is a handful of pure-torch lines (verified from
`airbench94_muon.py::init_whiten`, no new deps):

```python
def init_whitening_conv(layer, train_images, eps=5e-4):
    # layer: nn.Conv2d(3, 2*K, kernel_size=k, padding=0, bias=True), weight frozen
    # train_images: (N, 3, 32, 32) float tensor, ALREADY normalized exactly like eval
    c = train_images.shape[1]
    h, w = layer.weight.shape[2:]                      # k x k, e.g. 2x2
    # extract every k x k patch from a subset of images
    patches = (train_images.unfold(2, h, 1).unfold(3, w, 1)
                            .transpose(1, 3).reshape(-1, c, h, w).float())
    patches_flat = patches.view(len(patches), -1)      # (P, c*k*k)
    cov = (patches_flat.T @ patches_flat) / len(patches_flat)   # (c*k*k, c*k*k)
    eigvals, eigvecs = torch.linalg.eigh(cov, UPLO="U")
    scaled = eigvecs.T.reshape(-1, c, h, w) / torch.sqrt(eigvals.view(-1, 1, 1, 1) + eps)
    layer.weight.data[:] = torch.cat((scaled, -scaled))   # 2*(c*k*k) output channels
    layer.weight.requires_grad = False                    # FREEZE the whitening filters
```

Details and decisions:
- **k = 2** (airbench94 default). With c=3 -> `c*k*k = 12` eigvectors -> `whiten_width
  = 2 * 12 = 24` output channels (the `cat(scaled, -scaled)` doubles them so a
  following GELU/ReLU can use both signs). The whitening conv is
  `nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True)`; with padding=0 the 32x32
  input becomes 31x31, which the downstream MaxPools handle fine.
- **Patch subset**: use ~5000 training images (not all 50k) for the covariance — this
  is what the paper/hlb use and keeps `eigh` on a 12x12 matrix trivial (<<1s). Pull
  them from the *unaugmented* training set tensor (load CIFAR10 once with just
  ToTensor+Normalize, stack ~5000, move to GPU, run the init).
- **Freeze**: set `requires_grad = False` on the weight and exclude it from the
  optimizer param groups. In airbench the whitening *bias* is briefly trained then
  frozen (`whiten_bias_epochs=3`); for simplicity and safety I propose **keeping the
  bias trainable for the whole run** (or fixed at zero) — it is a tiny 24-vector and
  removing the schedule complexity reduces implementation risk with negligible
  accuracy cost. The eigendecomposition weight is the part that matters.
- `eps=5e-4` matches the reference; it regularizes the inverse-sqrt against tiny
  eigenvalues.

This runs once at setup (counts against startup, not the 300s training budget, since
the timer in `train.py` only accumulates `dt` inside the step loop).

### 2. Architecture (start with the airbench94 "94%" net, with a switch to the 96 net)

```python
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, ch, momentum=0.6, eps=1e-12):
        super().__init__(ch, eps=eps, momentum=1 - momentum)
        # airbench freezes the BN weight (scale) and learns only the bias;
        # SAFER default for us: leave both learnable to de-risk. Try frozen-scale as ablation.

class Conv(nn.Conv2d):
    def __init__(self, cin, cout):
        super().__init__(cin, cout, kernel_size=3, padding="same", bias=False)

class ConvGroup(nn.Module):
    # 94-net variant: conv -> pool -> bn -> gelu -> conv -> bn -> gelu
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1, self.pool = Conv(cin, cout), nn.MaxPool2d(2)
        self.norm1, self.conv2, self.norm2 = BatchNorm(cout), Conv(cout, cout), BatchNorm(cout)
        self.activ = nn.GELU()
    def forward(self, x):
        x = self.conv1(x); x = self.pool(x); x = self.norm1(x); x = self.activ(x)
        x = self.conv2(x); x = self.norm2(x); x = self.activ(x)
        return x
```

Network (`airbench94` widths):

```python
widths = dict(block1=64, block2=256, block3=256)
self.whiten = nn.Conv2d(3, 24, 2, padding=0, bias=True)   # frozen weight
self.layers = nn.Sequential(
    nn.GELU(),
    ConvGroup(24, 64),
    ConvGroup(64, 256),
    ConvGroup(256, 256),
    nn.MaxPool2d(3),
)
self.head = nn.Linear(256, 10, bias=False)

def forward(self, x):
    x = self.whiten(x)                # frozen whitening conv (applied at eval too)
    x = self.layers(x)
    x = x.flatten(1)
    x = self.head(x)
    return x / x.size(-1)             # airbench scales logits by 1/feat_dim; with label smoothing this matters
```

The `/ x.size(-1)` logit scaling and `head` bias-free linear are copied from
airbench; they interact with label smoothing and the high LR. (Note `x.size(-1)`
after flatten is the head input width via the Linear, so in airbench the division is
by the pre-head flatten dim — replicate exactly: divide logits by a fixed scalar
constant tuned to ~the head fan-in, e.g. a fixed `scale_out` constant, rather than a
runtime dim, to avoid ambiguity. Use a single tunable constant `LOGIT_SCALE`.)

**High-accuracy switch (recommended target):** add a residual conv pair inside each
ConvGroup (airbench96: "added an extra layer to each ConvBlock, added residual
connections, increased width") and run more epochs. Implement ConvGroup96 as
`conv1->pool->bn->gelu` then a residual block `x = x + gelu(bn(conv3(gelu(bn(conv2(x))))))`.
Plan: **get the 94-net working and measured first**, then flip to the 96-net only if
the 94-net already clears baseline comfortably and time remains in the budget.

### 3. Optimizer, schedule, augmentation, loss

- **Optimizer**: `torch.optim.SGD(params, lr=LR, momentum=0.85, nesterov=True,
  weight_decay=WD)`. Use **decoupled param groups** like airbench: separate group for
  BN/conv biases with a scaled LR (`bias_scaler`), and a separate (higher) LR for the
  head. To de-risk, a single group with one LR is an acceptable v1; the decoupled
  groups are a known +0.2-0.4pp refinement, add them as a second pass.
- **Schedule**: triangular one-cycle implemented with
  `torch.optim.lr_scheduler.OneCycleLR` (built into torch 2.9 — no new dep) over the
  *planned* total step count, OR a manual lambda `lr = peak * min(step/warmup,
  (total-step)/(total-warmup))`. Because our loop is **time-budgeted not
  step-budgeted**, estimate `total_train_steps` from a quick measured steps/sec on the
  first ~50 steps and the 300s budget, then build the schedule for that horizon. This
  is the single most important integration detail (see Risk).
- **Label smoothing**: `F.cross_entropy(out, targets, label_smoothing=0.2)` (airbench
  value). Replaces the current plain `F.cross_entropy` on line 173.
- **Augmentation**: keep `RandomCrop(32, padding=4)` (== airbench's translate) and
  `RandomHorizontalFlip()`, both already in the baseline `train_tf` (lines 117-124).
  Optionally add Cutout via a custom transform (airbench uses cutout in the 96-net);
  start without it to reduce risk, add as ablation.
- **Batch size**: airbench uses very large batches (500-2000) because its kernels are
  fused/compiled. On the H20 (98GB) a wide net at bs=2000 fits in VRAM, but the H20's
  bf16 compute is the bottleneck. Propose **bs=512** as a safe start (good
  GPU utilization, stable BN stats, more steps for the schedule), tune up if
  steps/sec is memory-bound.

### 4. AMP / precision

Use `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` around the forward +
loss (H20 supports bf16; bf16 needs no GradScaler, simpler and numerically safe).
Convert the model to `channels_last` (`model.to(memory_format=torch.channels_last)`
and inputs likewise) for conv throughput. **Do not** hard-cast the model to `.half()`
like airbench does (that path needs careful BN-in-fp32 handling); autocast achieves
the same speed with far less risk. The frozen whitening conv runs inside autocast
fine. `torch.compile` is allowed (in deps) and would help, but adds compile-time
(excluded from budget) and occasional first-run instability — **leave it off for v1**,
consider it only if steps/sec is the binding constraint.

### 5. TTA decision (horizontal flip inside forward)

**Recommendation: include flip TTA, gated by `self.training`.** Because frozen eval
calls `model(inputs)` directly, averaging logits of `x` and `x.flip(-1)` inside
`forward` is a legitimate, free ~+0.2-0.4pp at eval with zero training cost. Gate it
so it only activates in eval mode (otherwise it doubles training compute and corrupts
BN stats):

```python
def forward(self, x):
    out = self._forward_once(x)
    if not self.training:                       # eval (prepare.py sets model.eval())
        out = 0.5 * out + 0.5 * self._forward_once(x.flip(-1))
    return out
```

This mirrors airbench's `infer_mirror`. Translate-TTA (airbench tta_level=2) is also
possible but adds 4-9x eval forwards and more code; **flip-only** is the high
value/low risk choice. Eval time is excluded from the 300s training budget, but the
goal allows ≤1 validation/epoch and a 10-min wall-clock cap — flip TTA only doubles
eval cost, well within limits.

### 6. Normalization consistency (critical)

The frozen eval (`prepare.py` lines 13-20) feeds the model images normalized with
mean=(0.4914,0.4822,0.4465), std=(1,1,1) — i.e. **mean-subtract only**, values in
roughly [-0.49, 0.51]. The whitening eigendecomposition MUST be computed on patches in
*exactly this space*, because the frozen whitening filters are then applied to
eval images in that same space. Concretely: build the patch tensor from CIFAR10 loaded
with `ToTensor() + Normalize((0.4914,0.4822,0.4465),(1,1,1))` (identical to eval), run
`init_whitening_conv` on it, and keep the train transform's Normalize identical.
**Do not** introduce airbench's own per-channel std normalization — it would
desynchronize train/whitening from the frozen eval and silently tank accuracy. This
is the single easiest way to get a subtly-broken run, so it is called out explicitly.

## Reasoning with cited pointers

- **Whitening conv (mechanism + exact math)**: `airbench94_muon.py::init_whiten`
  (quoted verbatim above) — `torch.linalg.eigh(cov, UPLO="U")`, `eigvecs.T.reshape`,
  inverse-sqrt scaling, `cat(scaled, -scaled)`, weight frozen. Pure torch, no new
  deps. hlb-CIFAR10 `init_whitening_conv` is the same technique. Paper: "94% on
  CIFAR-10 in 3.29 Seconds on a Single GPU", arXiv 2404.00498 (Keller Jordan)
  describes the whitening layer as the key convergence accelerator.
- **Architecture (ConvGroup/widths/GELU/BN/head)**: quoted from `airbench94_muon.py`
  — `widths=dict(block1=64, block2=256, block3=256)`, `whiten_kernel_size=2`,
  `whiten_width = 2*3*k**2 = 24`, ConvGroup = conv/pool/bn/conv/bn/gelu,
  `nn.MaxPool2d(3)` then linear head, logits `/ x.size(-1)`.
- **Recipe (SGD-nesterov 0.85, triangular LR, label_smoothing=0.2, bias_scaler=64,
  whiten_bias_epochs=3, tta_level)**: airbench94 hyperparameter dict; the decoupled
  LR parametrization is from myrtle.ai "How to Train Your ResNet."
- **Headroom / budget arithmetic**: DeepWiki airbench benchmarks — airbench94 = 94.01%
  in 3.83s (≈10 ep), airbench95 = 95.01% in 10.4s (15 ep), airbench96 = 96.03% in
  34.7s (37 ep) on A100. Our budget is **300s on H20**. Even assuming the H20 is
  4-8x slower than A100 on this bf16 conv net (H20 ≈148 vs A100 ≈312 bf16 TFLOP, and
  the H20 is bandwidth-rich/compute-poor), airbench96's 37-epoch / ~35s-A100 recipe
  projects to ≈140-280s on H20 — it fits, with the 94-net (10 ep) fitting trivially
  and leaving room for extra epochs. This is the core feasibility argument: the recipe
  that documents 96% needs ~1/10th of our wall-clock on stronger hardware.
- **TTA**: airbench `infer_mirror = 0.5*net(x)+0.5*net(x.flip(-1))`, tta_level=1;
  legitimate here because frozen eval calls `model(inputs)` directly.

## Estimated effort

**High** (relative to one experiment loop). This is a full rewrite of the model and
training loop in `train.py`, not a tweak: new whitening-init setup pass, new module
classes, new optimizer param-grouping, a time-budget-aware one-cycle schedule,
autocast+channels_last plumbing, and the eval-mode TTA gate. Many small details
(padding=0 dimension bookkeeping, logit scaling, normalization-space consistency,
schedule horizon estimation) each independently break the run if wrong. Budget for
several iterations to get a clean run.

## Risk assessment

**Worst case**: a subtly-broken integration trains to <91.57% or crashes, burning the
loop with no gain. The architecture is proven, so a *correct* port should clear
baseline; the risk is almost entirely implementation, not method.

Highest-likelihood failure modes, in order:

1. **Schedule/horizon mismatch (most likely).** The loop is time-budgeted; airbench is
   step-budgeted. If the one-cycle LR is built for the wrong `total_train_steps`, LR
   either never anneals (ends high -> poor final acc) or anneals too early (under-uses
   budget). Mitigation: measure steps/sec on a short warmup, set the horizon, and rely
   on `best_test_acc` (best across epochs) so a slightly-off horizon still captures the
   peak. This is the assumption that most needs to hold.
2. **Normalization desync** (see section 6) — silently caps accuracy. Mitigation:
   compute patch stats in the exact eval normalization space; assert input range.
3. **High-LR divergence**: airbench's high LR depends on whitening + label smoothing +
   the logit `/dim` scaling + frozen-BN-scale all being present. Dropping one (e.g.
   the logit scale) while keeping the high LR can diverge. Mitigation: port the full
   recipe together; if NaN, lower peak LR first.
4. **H20 slower than projected**: if bf16 conv throughput is worse than estimated,
   only the 94-net (not 96-net) fits. That still targets ~94% >> 91.57% baseline, so
   the experiment still likely succeeds — the 96-net is upside, not the floor.
5. **BN with very wide channels + small effective batch** can give noisy stats;
   bs=512 and momentum=0.6 BN (1-momentum in torch) mitigate.

**Why it's still worth it**: it is the only candidate with a documented, reproducible
path to 95-96%, and the budget feasibility math has a large safety margin. Even a
conservative, partially-tuned port (94-net, single LR group, flip TTA) should land
~93-94%, comfortably beating the +0.1pp bar.

## Expected accuracy estimate

- **Conservative (94-net, single LR group, no Cutout, flip TTA, first clean run):
  ~93.0-94.0%.** This is the airbench94 target; we have more wall-clock than it needs,
  and flip TTA adds ~+0.2-0.4pp. Already +1.5-2.5pp over the 91.57% baseline.
- **Likely with the 96-net recipe (residual ConvGroups, more epochs, decoupled LR
  groups, Cutout, flip TTA), if it fits the budget on H20: ~95.0-96.0%.**
- **Floor (correct port, untuned): ~92.5%**, still clears the bar.

Justification: these are the documented airbench numbers on a strictly tighter
wall-clock budget than ours; the only discount is H20<A100 throughput, which trades
*reachable epochs* (hence which variant fits) rather than the *attainable accuracy per
epoch*. Best-across-epochs scoring (`best_test_acc`) further protects the estimate.

## Sources

- [KellerJordan/cifar10-airbench](https://github.com/KellerJordan/cifar10-airbench)
- [airbench neural network architectures (DeepWiki)](https://deepwiki.com/KellerJordan/cifar10-airbench/2.3-neural-network-architectures)
- [airbench performance benchmarks (DeepWiki)](https://deepwiki.com/KellerJordan/cifar10-airbench/1.2-performance-benchmarks)
- [airbench Muon optimizer / init_whiten (DeepWiki)](https://deepwiki.com/KellerJordan/cifar10-airbench/2.2-muon-optimizer)
- [hlb-CIFAR10 main.py (tysam-code)](https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py)
- ["94% on CIFAR-10 in 3.29 Seconds on a Single GPU", arXiv 2404.00498](https://arxiv.org/pdf/2404.00498)
