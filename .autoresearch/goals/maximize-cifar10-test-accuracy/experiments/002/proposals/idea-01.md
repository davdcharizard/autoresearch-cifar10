# Proposal idea-01: Frozen whitening initial conv (+ optional GELU) on the working DavidNet

## Summary

Keep the EXP-001 ResNet-9 / DavidNet and its full recipe exactly as-is (95.22%
baseline), and make ONE incremental architectural addition: prepend a **fixed
(non-learned, `requires_grad=False`) whitening convolution** as the very first layer.
Its weights come from the eigendecomposition of ~5000 normalized training-image 2×2
patches (`torch.linalg.eigh`), constructed once at setup in the EXACT frozen-eval
normalization space, producing 24 output channels via `cat(scaled, -scaled)`. Its
output (24 channels) feeds the existing `prep` conv, whose input width changes from 3
to 24. A secondary, separable lever is swapping `ReLU` → `GELU` inside `conv_bn`,
which hlb-CIFAR10 reports as a small speed/accuracy win.

This is the airbench/hlb-CIFAR10 "load-bearing convergence accelerator" ported onto a
proven base, NOT a wholesale architecture swap (that was idea-03 in EXP-001, deferred
as high-risk). Everything else — time-based triangular one-cycle (peak 0.4,
PCT_START 0.15), SGD+Nesterov (mom 0.9, wd 5e-4), label smoothing 0.2, Cutout 8,
bf16 autocast + channels_last, batch 512, scale_out 0.125, ~192 epochs/300s — stays
byte-for-byte. The mechanism: whitening decorrelates the input so the first conv
stages no longer have to learn low-level decorrelation, loss drops faster early, and
more *effective* learning fits in the fixed 300s budget → higher `best_test_acc`.

## What it targets (limiter from the diagnosis)

The named limiter from EXP-001's analysis is **convergence speed under a fixed
training-time budget**: accuracy is bounded by how much useful optimization fits in
300s. EXP-001 confirmed the bulk of the gain arrives in the low-LR tail (ep102 89.9%
→ ep187 95.05% → ep192 95.22%), i.e. the net is still improving and is
budget-limited, not capacity-limited (1.6 GB of 98 GB VRAM, 6.5M params). The
whitening front-end attacks exactly this: it removes the low-level
decorrelation/whitening work the first conv layers otherwise spend gradient steps
learning, so the same 192 epochs do strictly more high-level learning. The airbench
paper (arXiv:2404.00498) and tysam hlb-CIFAR10 both attribute the *majority* of their
speedup to this single layer. Faster per-epoch progress under an unchanged 300s
budget → a higher best-across-epochs accuracy.

## Exact `train.py` changes

All changes confined to `train.py`; `prepare.py` (eval + `TIME_BUDGET_S=300`)
untouched. Normalization stays `mean=(0.4914,0.4822,0.4465), std=(1,1,1)` everywhere.

### 1. Whitening conv initialization (pure torch, no new deps)

Add a setup helper. The algorithm is quoted from airbench `init_whiten` /
hlb-CIFAR10 `get_whitening_parameters`+`init_whitening_conv` (verified pure
torch/numpy; matches the verbatim listing in EXP-001 `proposals/idea-03.md` §1 and
the knowledge note `references/fast-cifar10-recipes.md`):

```python
@torch.no_grad()
def init_whitening_conv(layer, train_images, eps=5e-4):
    # layer: nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=False), weight FROZEN
    # train_images: (N, 3, 32, 32) float tensor, ALREADY normalized exactly like eval
    c = train_images.shape[1]                          # 3
    h, w = layer.weight.shape[2:]                      # 2, 2
    patches = (train_images.unfold(2, h, 1)
                           .unfold(3, w, 1)
                           .transpose(1, 3)
                           .reshape(-1, c, h, w).float())   # (P, 3, 2, 2)
    patches_flat = patches.reshape(patches.shape[0], -1)    # (P, 12)
    cov = (patches_flat.T @ patches_flat) / patches_flat.shape[0]   # (12, 12)
    eigvals, eigvecs = torch.linalg.eigh(cov)               # ascending eigvals
    scaled = eigvecs.T.reshape(c * h * w, c, h, w) / torch.sqrt(eigvals.view(-1, 1, 1, 1) + eps)
    layer.weight.data[:] = torch.cat((scaled, -scaled))     # (24, 3, 2, 2)
    layer.weight.requires_grad = False
```

Decisions:
- **k = 2** (airbench default). `c*k*k = 12` eigenvectors → `cat(scaled, -scaled)` →
  24 output channels (sign-doubling lets the following nonlinearity use both signs).
  Layer: `nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=False)`.
- **No bias** on the whitening conv. Airbench briefly trains then freezes a whitening
  bias (`whiten_bias_epochs=3`); I drop it entirely to remove schedule complexity —
  the BN immediately downstream (in `prep`'s `conv_bn`) absorbs any centering, so the
  bias is redundant here. This is a deliberate simplification vs airbench and de-risks
  the integration. (Fallback if accuracy disappoints: re-add a trainable 24-vector
  bias kept learnable for the whole run.)
- **Patch subset**: ~5000 *unaugmented* training images. Load CIFAR10 a second time
  in setup with `ToTensor()+Normalize(EVAL_MEAN, EVAL_STD)` only (no crop/flip/cutout),
  stack the first 5000, move to GPU, run the init. `eigh` on a 12×12 matrix is <<1s.
- **Freeze + optimizer exclusion**: after `requires_grad=False`, the whitening weight
  must NOT be handed to SGD. Change the optimizer construction (currently
  `optim.SGD(model.parameters(), ...)` at lines 155-161) to
  `optim.SGD([p for p in model.parameters() if p.requires_grad], ...)`. With
  `requires_grad=False` it is also excluded from weight decay automatically. Verify
  `num_params` print uses `sum(p.numel() for p in model.parameters() if p.requires_grad)`
  if we want the trainable count (the frozen 288-weight layer is negligible either way).
- **Timing**: this runs once in setup, BEFORE `t_start_training`, so it counts against
  startup (which the budget meter at lines 170/182-210 excludes — the timer only
  accumulates `dt` inside the step loop). The extra second of `eigh`+a second CIFAR
  load is free against the 300s budget.

### 2. Channel/spatial bookkeeping where whitening meets `prep` (CRITICAL)

The existing net (lines 84-107) starts `self.prep = conv_bn(3, 64)`. The whitening
conv changes two things the rest of the net must accommodate:

- **Channels 3 → 24**: change `prep` to consume 24 channels:
  `self.prep = conv_bn(24, 64)`. Everything downstream (layer1 64→128, etc.) is
  unchanged because `prep` still emits 64.
- **Spatial 32 → 31** (padding=0, kernel 2 shrinks each dim by 1): the input becomes
  31×31 after whitening. `prep`'s 3×3 conv uses `padding=1` (line 68) so it preserves
  31×31. The MaxPool(2) layers (lines 89-92) use floor division: 31→15→7→3, then the
  final `nn.MaxPool2d(4)` (line 92) on a 3×3 input. **This is the one real risk**:
  `MaxPool2d(4)` with `kernel_size=4 > 3` on a 3×3 feature map returns an EMPTY tensor
  (no valid window), which breaks the flatten/Linear. Two clean fixes, pick one:
  1. **Preserve 32×32 (recommended):** give the whitening conv `padding=1` so the
     2×2 kernel on a 32+2 padded input yields 33×33 — wrong. Instead use the standard
     airbench trick of NOT shrinking: set whitening conv `padding=0` but then the
     downstream pool chain must tolerate odd sizes. The robust fix is to replace the
     final `self.pool = nn.MaxPool2d(4)` with `self.pool = nn.AdaptiveMaxPool2d(1)`,
     which yields a 1×1 output for ANY spatial size (3×3, 4×4, etc.) and is a no-op
     change in behavior when the input is already being globally pooled. This is the
     minimal, dimension-agnostic fix and is preferred.
  2. **Alternatively, keep 32×32** by padding the input by 1 column/row before the
     whitening conv (`F.pad`) so 2×2/pad-0 stays 32×32 → pools 32→16→8→4 →
     `MaxPool2d(4)` works unchanged. Cleaner spatially but adds a pad op in `forward`.

  **Recommendation: option 1 (`AdaptiveMaxPool2d(1)`)** — it is one line, removes all
  fragility around the 32→31 shift and the pool chain, and changes nothing else.
  Confirm the chain explicitly: 31 →prep→ 31 →(conv 128)→ 31 →MaxPool2d(2)→ 15
  →Residual→ 15 →(conv 256)→ 15 →MaxPool2d(2)→ 7 →(conv 512)→ 7 →MaxPool2d(2)→ 3
  →Residual→ 3 →AdaptiveMaxPool2d(1)→ 1 → flatten(512) → fc. Valid throughout.

- **Module wiring**: add `self.whiten = nn.Conv2d(3, 24, 2, padding=0, bias=False)` in
  `ResNet9.__init__`, and call it first in `forward` (line 102):
  `x = self.whiten(x); x = self.prep(x); ...`. CRUCIAL ordering bug to avoid: the
  existing `self.apply(self._weights_init)` (line 94) kaiming-inits every Conv2d
  INCLUDING `self.whiten`. Run `init_whitening_conv(model.whiten, patches)` in `main()`
  AFTER `model = ResNet9(...)` and AFTER any `.to(device)` move, so the eigendecomp
  weight is the final state and is not overwritten by kaiming. (Equivalently, exclude
  whiten in `_weights_init` via an identity check — but initializing after construction
  is simplest and least error-prone.)

### 3. GELU swap (optional, separable second lever)

Change `conv_bn` (lines 66-71) `nn.ReLU(inplace=True)` → `nn.GELU()`. hlb-CIFAR10
reports GELU as a small win over ReLU on this net family. Caveat for THIS experiment:
the kaiming init uses `nonlinearity="relu"` (line 99); GELU is close enough to ReLU
in gain that this is fine, but it is a confound. **Recommendation: run whitening-only
first** (clean attribution to the whitening mechanism vs the 95.22% base), then add
GELU as a second variant only if whitening alone clears the +0.1pp bar and budget
remains. If combining in one run to save a loop, accept the reduced attribution.

### 4. Normalization consistency (the #1 silent-failure risk)

The frozen eval (`prepare.py` lines 12-20) feeds images normalized with
`mean=(0.4914,0.4822,0.4465), std=(1,1,1)` — mean-subtract only, values ≈[-0.49,0.51].
The frozen whitening filters are applied to eval images in that space, so the
covariance/eigendecomp MUST be computed on patches in EXACTLY that space. Concretely:
build the 5000-image patch tensor from CIFAR10 loaded with
`ToTensor()+Normalize(EVAL_MEAN, EVAL_STD)` (identical to eval and to the train
transform's Normalize at line 132). Do NOT introduce airbench's per-channel std
normalization — it would desync train/whiten from the frozen eval and silently tank
accuracy. The train transform already normalizes to this space before Cutout, so the
whitening conv sees consistent statistics at train and eval. Since the whitening conv
is part of `forward`, it is applied at eval automatically (eval calls `model(inputs)`
directly), which is exactly what we want — train and eval both whiten identically.

## Reasoning with cited pointers

- **Whitening conv math (verbatim)**: airbench `init_whiten` /
  `airbench94_muon.py` — `torch.linalg.eigh(cov)`, `eigvecs.T.reshape`, inverse-sqrt
  scaling `/sqrt(eigvals+eps)`, `cat(scaled,-scaled)`, `requires_grad=False`,
  kernel 2, 3→24 channels, padding 0. Quoted in EXP-001 `proposals/idea-03.md` §1 and
  `knowledge/references/fast-cifar10-recipes.md` line 8. Paper: arXiv:2404.00498
  ("94% on CIFAR-10 in 3.29 Seconds"), which states the whitening layer is the key
  convergence accelerator and follows Page (2019) / tysam-code (2023).
- **Mechanism = convergence acceleration, not capacity**: hlb-CIFAR10 and
  myrtle.ai "How to Train Your ResNet" attribute the bulk of their speedup to the
  whitening front-end; it lets loss drop in epoch 1. This matches EXP-001's measured
  limiter (budget-bound, capacity-spare: 1.6 GB VRAM, gains in the low-LR tail).
- **Incremental, not wholesale**: EXP-001 `04-analysis.md` lines 34 & 41 explicitly
  name "Whitening initial conv on the ResNet-9 base" as the high-confidence next step,
  documented path to 95–96%+. This proposal is that exact step — minimal surface area
  on a proven 95.22% net, vs idea-03's full airbench rewrite (deferred as high-risk).
- **GELU**: knowledge note line 7 / hlb-CIFAR10 (GELU activations) — small win on this
  net family. Treated here as a separable, lower-priority lever.
- **Dimension handling**: read directly from `train.py` lines 84-107 — the
  `MaxPool2d(4)` at line 92 is the only spatial fragility introduced by padding=0;
  `AdaptiveMaxPool2d(1)` removes it. `prep`'s `padding=1` 3×3 conv preserves whatever
  spatial size enters it.

## Estimated effort

**Low–medium** relative to one experiment loop. This is an additive change on an
unchanged, working `train.py`: one ~12-line setup helper, a second (init-only) CIFAR
load, `self.whiten` + one `forward` line, `prep(3→24)`, `MaxPool2d(4)→AdaptiveMaxPool2d(1)`,
optimizer param filter, and (optional) the ReLU→GELU one-liner. No schedule rework, no
optimizer rework, no precision/throughput rework — those were the high-risk parts of
idea-03 and are entirely avoided here. Main work is the dimension/freeze/init-order
bookkeeping above. Budget 1, maybe 2 iterations for a clean run.

## Risk assessment

**Worst case**: a subtly-broken integration trains below 95.22% (no improvement) or
crashes, burning the loop. Because the base is proven and only the front-end changes,
a *correct* port should be ≥ baseline; risk is implementation, not method.

Highest-likelihood failure modes, in order:
1. **Spatial/pool break (most likely crash)**: 31×31 → `MaxPool2d(4)` on a 3×3 map
   returns empty. Mitigated by `AdaptiveMaxPool2d(1)` (option 1, §2). Must be applied;
   a one-batch smoke test catches it immediately.
2. **Init overwritten by kaiming**: if `init_whitening_conv` runs before
   `self.apply(self._weights_init)`, the eigendecomp weight is clobbered and the layer
   becomes a random frozen conv (degrades, doesn't crash — silent). Mitigated by
   initializing AFTER model construction in `main()` (§2).
3. **Frozen weight handed to optimizer / weight decay**: with `requires_grad=False`
   SGD skips it, but be explicit with the param filter (§1) so wd 5e-4 never touches
   it. Low impact (it has no grad) but worth pinning.
4. **Normalization desync** (§4): patch stats computed in a different space than eval
   → whitening filters miscalibrated → silent accuracy cap. Mitigated by computing
   patches in the exact `EVAL_MEAN/EVAL_STD` space; assert input range in setup.
5. **Whitening doesn't help at 192 epochs**: the airbench speedup is largest in the
   *few-epoch* regime; at ~192 epochs the base net may already have learned good
   low-level filters, so the marginal gain could be small (tenths, or near-zero). This
   is the assumption that most needs to hold for an *improvement* (not just a
   non-crash). See expected-accuracy justification below — this is the honest soft
   spot of the idea.

**Honest assessment**: the convergence-acceleration mechanism is strongest when
training is epoch-starved. EXP-001 already fits ~192 epochs of a completing one-cycle,
which is far past DavidNet's canonical 24 — so the net is *not* severely
epoch-starved, and whitening's benefit may be partially saturated. The gain is most
plausibly realized as a slightly better-conditioned optimization that nudges the final
low-LR plateau up by a few tenths, rather than the dramatic speedup airbench sees at
10 epochs. This makes the idea solid-but-modest rather than a guaranteed large win.

## Expected accuracy estimate vs 95.22% base

- **Most likely: 95.3–95.6%** (+0.1 to +0.4pp). Whitening adds a small, well-conditioned
  improvement on top of a net that is already near this recipe's ceiling. This clears
  the +0.1pp bar but with modest margin.
- **Optimistic (whitening + GELU both help, conditioning gain compounds): ~95.6–95.9%.**
- **Floor / null result: ~95.1–95.3%** — if the net is already epoch-saturated, the
  whitening gain is washed out and the result lands within noise of baseline, i.e. NOT
  a clear improvement. Given the ±0.1–0.15pp run-to-run noise typical here, there is a
  real chance this finishes statistically tied with 95.22%.

Justification: airbench's documented headroom is to 95–96% *with whitening as the key
ingredient*, and EXP-001 reached 95.22% *without* it on the same net class — so the
two together point to the low-95s to high-95s. The discount vs airbench's larger gains
is the epoch-rich regime (§risk 5): the benefit is most likely a few tenths, not a
full point. Best-across-epochs scoring (`best_test_acc`) protects against a slightly
sub-optimal anneal. Net: a credible, low-risk shot at +0.1–0.4pp, with an honest
non-trivial probability of landing within noise of the baseline — the correct call is
to run whitening-only first for clean attribution, then add GELU if it clears the bar.

## Sources

- [KellerJordan/cifar10-airbench](https://github.com/KellerJordan/cifar10-airbench)
- ["94% on CIFAR-10 in 3.29 Seconds on a Single GPU", arXiv:2404.00498](https://arxiv.org/abs/2404.00498)
- [hlb-CIFAR10 (tysam-code) main.py](https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py)
- [airbench neural network architectures (DeepWiki)](https://deepwiki.com/KellerJordan/cifar10-airbench/2.3-neural-network-architectures)
- EXP-001 analysis: `experiments/001/04-analysis.md` (next-step §); idea-03:
  `experiments/001/proposals/idea-03.md` (verbatim init_whiten); knowledge note:
  `knowledge/references/fast-cifar10-recipes.md`
