# Proposal idea-01: Frozen patch-whitening first convolution

## One-line summary
Prepend a fixed, non-learnable ZCA/PCA-whitening 3x3 conv (eigendecomposition of CIFAR-10
training patches, computed once at startup off the 300s timer) in front of the existing
`prep` conv, optionally pairing it with an identity-initialized learnable 1x1 mixer, so the
learnable stem receives decorrelated input. This is the foundational airbench/hlb/Page trick,
adapted to keep the ResNet9 pooling chain intact.

---

## Limiter this targets

From the diagnosis (`03-experiment-learnings.md`): "Most accuracy gain arrives in the low-LR
tail of a completing one-cycle" and the recipe already fits ~183 epochs in 300s. The limiter
is **optimization conditioning / convergence speed of the learnable stem**, not raw capacity
(VRAM is 1.6GB of 98GB; the net is not epoch-starved in the extreme).

Whitening attacks conditioning directly. The first learnable conv (`prep`, 3->64) currently
sees raw mean-subtracted RGB, whose 3x3x3=27-dim patch covariance is highly anisotropic
(strong inter-channel R/G/B correlation + strong spatial DC component). A poorly conditioned
input covariance makes the loss surface for the first layer's weights elongated, so SGD makes
slow progress along low-curvature directions early in training. Decorrelating the input
(covariance -> identity) sphereizes those directions, letting the one-cycle schedule do useful
work earlier and reach a marginally better tail minimum within the same time budget. This is
the same mechanism airbench credits as its "biggest convergence accelerator."

**Honest caveat carried from the EXP-002 idea review:** whitening's benefit is largest in
epoch-STARVED regimes (airbench's headline is reaching 94% in ~10 epochs). At 183 epochs our
one-cycle already substantially anneals, so the marginal gain is expected to be a FEW TENTHS,
and could land near or below the +0.1pp bar. See "Expected magnitude" below for the honest
estimate.

---

## Why promising (evidence)

1. **airbench (Keller Jordan 2024, arXiv:2404.00498)**: the frozen patch-whitening initial
   conv is the first layer in *every* airbench variant and is named the single biggest
   convergence accelerator. Construction (from the reference + hlb source):
   `eigenvalues, eigenvectors = torch.linalg.eigh(est_covariance, UPLO='U')`, scale eigenvectors
   by `1/sqrt(eigenvalues+eps)`, then concat `(V, -V)` to double output channels (so a +/-
   eigenvector pair gives a signed whitened response; downstream ReLU/GELU then sees both
   polarities). `requires_grad=False`. Patches from ~5000 training images, kernel 2, 3->24 ch.
   (`knowledge/references/fast-cifar10-recipes.md` line 8; web: airbench README, hlb main.py
   `get_whitening_parameters`/`set_whitening_conv`.)

2. **hlb-CIFAR10 (tysam-code)**: same construction, comment "we don't want to train this, since
   it is implicitly whitening over the whole dataset" -> "maps the input to a nicely distributed
   sphere where the most significant features each have their own axis." Confirms the freezing
   and the conditioning rationale.

3. **Our history**: EXP-001/002 established that tricks compose ADDITIVELY on the DavidNet base
   (`03-experiment-learnings.md` Patterns: "compose new ideas on top rather than re-deriving").
   Whitening is explicitly flagged as the "top next-step candidate (UNTRIED)" in the reference
   (line 15). It is orthogonal to the EMA+TTA eval-side wins of EXP-002, which stay ON.

4. **Code fact**: the existing `prep = conv_bn(3, 64)` is exactly the layer airbench/hlb
   replace/precede. The whitening conv slots cleanly in front of it, and the BN inside `prep`
   already absorbs any residual scale, so we do NOT need to hand-tune the whitening output scale.

---

## Concrete change to `train.py`

All edits are inside `train.py` (only editable file). Three touch points: a whitening-matrix
builder, the `ResNet9` module, and the `main()` setup (compute matrix off-timer, exclude from
the optimizer).

### Spatial-dimension decision (CRITICAL — the main adaptation)

The ResNet9 chain pools by 2,2,2 (layer1/2/3) then 4 (final) = total /32, requiring a 32x32
feature map entering `prep`. airbench's kernel=2/pad=0 whitening conv shrinks 32->31 and would
break our chain (31 is not divisible cleanly by the pool stack; final MaxPool2d(4) on a
4x4-ish map would misalign). **Decision: use a 3x3, padding=1 whitening conv**, which preserves
32x32 exactly and feeds the existing chain unchanged. Patches are therefore 3x3x3 = 27-dim;
covariance is 27x27; eigendecomposition is trivially cheap. This is a deliberate divergence
from airbench's kernel=2 chosen to respect our frozen pooling chain — whitening over 3x3 RGB
patches is still a valid decorrelating transform (David Page's original cifar10-fast used a
patch-whitening front end on the full image at this scale).

### Channel decision

27-dim patches -> 27 whitened directions -> concat (V, -V) = **54 output channels**. Then a
learnable 1x1 conv maps 54 -> 64 to feed the existing `prep` BN/ReLU width. We KEEP the original
`prep` (3x3, 64->... no: prep is 3->64). Two clean composition options; I specify Option A as
primary because it is the minimal, lowest-risk change:

**Option A (primary — whitening then keep prep as the learnable mixer):**
Insert whitening BEFORE `prep`, and change `prep`'s input channels from 3 to 54. i.e.
`self.whiten = nn.Conv2d(3, 54, 3, padding=1, bias=False)` (frozen) and
`self.prep = conv_bn(54, 64)` (learnable, kaiming-init as today). The existing 3x3 `prep` conv
becomes the learnable layer that consumes the whitened 54-channel signal. This adds only the
frozen 3x3x3x54 = 1458 fixed weights; param count change is tiny and `prep` grows from
3*64*9=1728 to 54*64*9=31104 learnable weights (negligible vs 6.5M total).

**Option B (airbench-faithful — whitening + identity-init 1x1 mixer + original prep):**
`self.whiten = nn.Conv2d(3,54,3,padding=1,bias=False)` (frozen), then
`self.mix = nn.Conv2d(54, 3, 1, bias=False)` initialized so the composition starts near
identity-on-RGB (set `mix.weight` to the pseudo-inverse-ish projection back to 3ch, or simply a
small random init and let it learn), then original `self.prep = conv_bn(3,64)`. This matches
airbench's "identity initialization pairs with whitening" but adds a hyperparameter (how to init
`mix`) and a risk of the identity init being wrong. **Recommend testing Option A first**; only
move to B if A underperforms, because A removes the identity-init failure mode entirely (the BN
in `prep` self-calibrates the whitened input scale).

### Whitening-matrix builder (new function)

```python
def compute_whitening_weight(train_set, mean, kernel=3, n_patches=50000, eps=1e-4):
    """Return a frozen conv weight [2*K, 3, kernel, kernel] whitening 3x3 RGB patches.
    Computed in the SAME normalized space as eval (mean-subtract, std=1)."""
    # Sample raw images, apply ONLY ToTensor + mean-subtract (NO crop/flip/cutout) so the
    # patch statistics match the eval input distribution exactly. Cap the number of images
    # to keep this cheap and off-budget.
    import numpy as np
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    # Pull a capped subset of raw uint8 images directly from train_set.data (HWC uint8).
    n_img = min(5000, len(train_set.data))
    imgs = torch.from_numpy(train_set.data[:n_img]).float().div_(255.0)  # [N,32,32,3]
    imgs = imgs.permute(0, 3, 1, 2).contiguous()                        # [N,3,32,32]
    imgs = imgs - mean_t                                                # match eval space
    # Unfold into kernel x kernel patches (hlb/airbench style).
    p = imgs.unfold(2, kernel, 1).unfold(3, kernel, 1)                  # [N,3,H',W',k,k]
    p = p.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3 * kernel * kernel)    # [M, 27]
    if p.shape[0] > n_patches:                                          # cap patch count
        idx = torch.randperm(p.shape[0])[:n_patches]
        p = p[idx]
    p = p - p.mean(0, keepdim=True)                                     # center patches
    cov = (p.T @ p) / (p.shape[0] - 1)                                  # [27,27]
    eigvals, eigvecs = torch.linalg.eigh(cov, UPLO='U')                 # ascending
    # Whitening filters: eigvecs scaled by 1/sqrt(eigval+eps); reshape to conv weight.
    W = (eigvecs / torch.sqrt(eigvals + eps).unsqueeze(0)).T            # [27,27] rows=filters
    W = W.reshape(3 * kernel * kernel, 3, kernel, kernel)               # [27,3,k,k]
    weight = torch.cat([W, -W], dim=0)                                  # [54,3,k,k]  +/- pairs
    return weight  # float32, to be loaded into a requires_grad=False conv
```

Notes: reading `train_set.data` (the raw HWC uint8 numpy array torchvision exposes) avoids
running the augmenting `train_tf` and keeps the patch space exactly eval-aligned (mean-subtract,
std=1). numpy is an allowed dep. This is a pure-CPU 27x27 eigendecomposition over <=50k patches:
milliseconds. Print `whitening_seconds` for transparency.

### `ResNet9.__init__` changes (Option A)

```python
self.whiten = nn.Conv2d(3, 54, 3, padding=1, bias=False)  # FROZEN, set after .to(device)
self.whiten.weight.requires_grad_(False)
self.prep = conv_bn(54, 64)   # was conv_bn(3, 64)
# ... layer1/2/3, pool, fc unchanged ...
self.apply(self._weights_init)   # kaiming-inits prep/layers/fc; whiten gets overwritten here
```

`_forward_once` gains one line at the top:
```python
def _forward_once(self, x):
    x = self.whiten(x)
    x = self.prep(x)
    ...
```

### `_weights_init` ordering hazard (CRITICAL)

`self.apply(self._weights_init)` kaiming-inits EVERY `nn.Conv2d`, INCLUDING `self.whiten`. So we
must (a) let `apply` run, then (b) OVERWRITE `whiten.weight` with the computed whitening matrix
AFTER construction AND after `.to(device)`, and set `requires_grad=False`. Concretely, add a
method and call it in `main()`:

```python
# in ResNet9:
def load_whitening(self, weight):
    with torch.no_grad():
        self.whiten.weight.copy_(weight.to(self.whiten.weight.device,
                                            self.whiten.weight.dtype))
    self.whiten.weight.requires_grad_(False)
```

### `main()` changes

1. After `train_set` is built and after `model = ResNet9(...).to(device, channels_last)`:
```python
t_w = time.time()
w_weight = compute_whitening_weight(train_set, EVAL_MEAN, kernel=3)
model.load_whitening(w_weight)
whitening_seconds = time.time() - t_w
print(f"whitening_seconds: {whitening_seconds:.2f}")
```
Place this BEFORE `t_start_training = time.time()` so it is OFF the 300s timer (mirrors the
off-budget `evaluator = Eval()` construction at module load). Do NOT also start the budget clock
yet — confirm `t_start_training` is taken after this block.

2. **Exclude the frozen whitening conv from the optimizer** (otherwise SGD with weight_decay
would decay the fixed weights even with zero grad, corrupting them via the wd term):
```python
optimizer = optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=PEAK_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True,
)
```
This is the single most important correctness line. With weight_decay=5e-4, leaving the frozen
weight in the param list would shrink it every step (decoupled or coupled wd applies to the
param regardless of grad), gradually destroying the whitening transform.

3. **EMA interaction**: `AveragedModel(model, ..., use_buffers=True)` copies ALL parameters and
buffers, including `whiten.weight`. Because `whiten.weight` never changes in `model`, its EMA
average converges to the same fixed value (EMA of a constant = that constant), so the EMA model's
whitening is correct. No change needed, but verify the EMA path still evaluates correctly.

4. `num_params` print: the existing `sum(p.numel() for p in model.parameters())` now includes the
1458 frozen whitening weights. Optionally report learnable-only count too; cosmetic.

---

## Risk assessment & failure modes

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Spatial-dim break** — kernel/padding shrinks the map and misaligns the pool chain | High if kernel=2 used | Use 3x3 **padding=1** which preserves 32x32 exactly. Verify first-batch shape entering `pool` is unchanged (still 512-wide -> 4x4 before MaxPool2d(4)). This is the headline adaptation. |
| **Frozen weights silently trained/decayed** — whiten in optimizer param list gets wd-shrunk | High if not excluded | Filter `requires_grad` in the SGD param list (change #2). requires_grad=False alone is NOT enough because wd acts on the param value. |
| **kaiming overwrite** — `self.apply(_weights_init)` clobbers the whitening matrix | Certain if not handled | `load_whitening()` runs AFTER `apply` and AFTER `.to(device)`. |
| **Normalization desync** — patch stats computed in a different space than eval inputs | Medium | Compute patches in the EXACT eval space: ToTensor-equivalent (`/255`) then subtract EVAL_MEAN, std=1. Do NOT apply crop/flip/cutout to the stat images. Eval still feeds raw mean-subtracted images; whitening is the first layer of the model so eval inputs pass through it identically to train. No desync as long as the conv lives inside the model (it does). |
| **Marginal gain below bar** — already-converged regime | Medium-High | See expected magnitude. This is the real scientific risk, not a correctness bug. |
| **bf16 autocast on a fixed fp32 matrix** — precision of whitening response | Low | The conv runs under autocast like every other conv; whitening tolerates bf16 (airbench uses fp16/bf16). Store weight in fp32, let autocast cast at runtime. The downstream BN re-normalizes scale. |
| **eps too small -> blow-up of low-variance directions** | Low | eps=1e-4 (hlb/airbench scale). Low-eigenvalue directions (high-freq noise) get bounded gain. If unstable, raise eps to 1e-3; BN downstream also caps scale. |

**Strongest assumption that must hold:** that decorrelating a 3x3 RGB patch covariance improves
the FINAL tail accuracy (not just early convergence) within a budget that already fully anneals.
If the one-cycle tail already finds an equally good minimum from raw input, the gain washes out
to ~0. This is the assumption most likely to fail.

---

## Expected magnitude vs the 95.82% bar (honest)

- airbench's whitening gain is measured in **epochs-to-94%**, a starved regime. Our 183-epoch
  fully-annealed run is much closer to the recipe's accuracy ceiling, so the marginal return is
  compressed.
- Realistic point estimate: **+0.1 to +0.4pp** (95.72 -> ~95.8-96.1). This straddles the +0.1pp
  bar. A plausible null outcome (+0.0 to +0.1pp, below bar) is genuinely on the table.
- Secondary upside: whitening could let us push PEAK_LR slightly higher (better-conditioned
  input tolerates larger steps) for a compounding gain — but that adds a tuning knob and risk;
  keep PEAK_LR=0.4 fixed for a clean A/B in the first run, note LR-raise as a follow-up.
- This is worth running because (a) it is the canonical untried front-end trick flagged as top
  candidate, (b) it composes additively with the kept EMA+TTA wins, and (c) the downside is a
  clean, cheap negative result. But I will not overstate it: this is a "few tenths, possibly at
  the bar" bet, not a sure improvement.

---

## Effort: LOW

~30-40 lines net in one file: one builder function, ~5 lines in `ResNet9`, ~6 lines in `main()`.
No new deps (numpy already allowed; `torch.linalg.eigh` is core torch). No schedule/budget logic
touched. One training run to A/B against 95.72. The only subtle parts are the three CRITICAL
correctness items (padding=1, optimizer param filter, post-`apply` load), all enumerated above.

## Verification checklist for the planner
- Assert `prepare.py` byte-unchanged (`git diff --quiet -- prepare.py`).
- Print and confirm `whitening_seconds` is small (<2s) and OUTSIDE `training_seconds`.
- Confirm feature-map shape entering `self.pool` is unchanged vs baseline (sanity print on step 1).
- Confirm `whiten.weight.requires_grad is False` and `whiten.weight` is NOT in `optimizer`'s
  param groups (print param-group numel sum vs learnable numel).
- Keep `torch.manual_seed(42)` / `torch.cuda.manual_seed(42)`; the patch subsample uses
  `torch.randperm` AFTER the seed is set, so it is deterministic and not seed-hacking.
- Single A/B: one run with whitening vs the recorded 95.72 baseline; +0.1pp bar.

## Sources
- [cifar10-airbench (KellerJordan)](https://github.com/KellerJordan/cifar10-airbench)
- [94% on CIFAR-10 in 3.29 Seconds (arXiv:2404.00498)](https://arxiv.org/abs/2404.00498)
- [hlb-CIFAR10 main.py (tysam-code)](https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py)
- knowledge/references/fast-cifar10-recipes.md (lines 8, 15, 19)
