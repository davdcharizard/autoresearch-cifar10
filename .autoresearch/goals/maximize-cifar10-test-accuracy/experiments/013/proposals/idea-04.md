# MOONSHOT — Sharpness-Aware Minimization (SAM) on the regularization-bound DavidNet

## 1. Summary

Hand-implement **Sharpness-Aware Minimization** (Foret et al., ICLR 2021) directly in
`train.py`'s training step — no new packages, no `sam-pytorch`. Each SAM step is two
forward-backward passes: an **ascent** to the local worst-case weight
`w + e_w`, `e_w = ρ · g / ‖g‖`, followed by a **descent** that applies the gradient
*computed at the perturbed point* to the *original* weights. This biases optimization
toward **flat minima**, which generalize better — a loss-geometry mechanism orthogonal
to every regularizer this net has already saturated.

The central cost problem (2× forward-backward → ~half the epochs → under-anneal) is the
dominant risk. The proposal therefore is **not plain SAM**. The recommended primary
configuration is **tail-only SAM**: plain SGD-Nesterov for the high-LR phase, switch to
SAM only in the low-LR tail (the last ~35–40% of the time budget, where this recipe's
accuracy concentrates — EXP-001 pattern). This keeps the global epoch count high (target
**≥120**, above the ≤110 under-anneal line) while spending the 2× cost exactly where
flat-minima selection matters most. ρ = 0.05 (canonical CIFAR-10 default); a 2-cell
ρ ∈ {0.05, 0.10} micro-sweep is the fallback if 0.05 is flat.

Concretely the recipe gains one hyperparameter block (`SAM_RHO`, `SAM_TAIL_FRAC`) and the
training step grows an ascent/descent gate. Everything else (EMA, TTA, whitening, ReZero,
aug, time-based one-cycle) is byte-unchanged.

## 2. What it targets (the named limiter)

The diagnosis is explicit and convergent: the net is **regularization-bound near its
generalization ceiling**, not optimizer-bound or capacity-bound. The evidence chain:

- **Optimizer axis exhausted** (EXP-009/010): tuned Muon only *ties* tuned SGD
  (96.33 vs 96.38) — "both SGD and Muon land at ~96.35±noise → the net is
  regularization-bound, not optimizer-bound" (03-experiment-learnings.md, Muon entry).
- **Regularization-scalar axis exhausted** (EXP-012): WD-shaping ties baseline, LS<0.2
  degrades — "the regularization-ALLOCATION axis ... is exhausted."
- **Input-aug axis saturating** (EXP-011): a second mechanistically-distinct aug (CutMix)
  only ties. The learnings entry itself names the escape hatch: *"Future regularization
  should target a DIFFERENT mechanism ... or **loss-geometry (SAM)**."*

SAM attacks generalization through **none** of these levers. It does not change the loss
function (label smoothing), the data distribution (augmentation), the weight-space penalty
(weight decay), or the update rule's preconditioner (Muon). It changes **which minimum**
the same SGD trajectory selects among the many that fit the training set — preferring
flat ones with lower worst-case neighborhood loss, which the SAM paper and a large
follow-up literature tie to a smaller generalization gap. This is the one untried axis the
diagnosis itself flags as live. That is the entire reason it qualifies as the moonshot.

## 3. Reasoning — can flat-minima beat the under-anneal cost?

This is the crux and must be argued quantitatively, not asserted.

### 3a. The cost arithmetic (be honest)

Current recipe: ~150 epochs / 300 s @ ~26.5k img/s. **Plain SAM** doubles forward-backward
per step. The forward+backward is the overwhelming majority of step wall-time (the per-step
`torch.cuda.synchronize()` and Python overhead are small at batch 512), so plain SAM lands
at **~75–80 epochs**. That is squarely in under-anneal territory: under-anneal already beat
us at 94 epochs (EXP-007) and 131 epochs (EXP-005). **Plain SAM is expected to lose** —
the flat-minima gain (+0.3–1.0pp on CIFAR ResNets at *matched* epochs) almost certainly
cannot offset halving the anneal budget on a net where "most accuracy lands in the low-LR
tail" and the tail is what gets truncated. I do not recommend plain SAM as the primary cell.

### 3b. Why tail-only SAM controls the cost

The recipe's structure makes a *temporal* split attractive over the *periodic* split
(LookSAM/SAM-k), because the literature is clear that naive periodic SAM-k **degrades
accuracy** ("naive periodic SAM ... exhibit[s] a significant decrease in accuracy";
"LookSAM significantly loses accuracy when the [interval] is larger than a threshold" —
search synthesis below). The recipe instead gives a *principled* place to spend SAM: the
**low-LR tail**, where (i) this exact recipe's accuracy demonstrably concentrates
(EXP-001: 89.9% @55% progress → 95.2% at LR→0), and (ii) flat-minima selection is most
meaningful because the iterate is settling into a basin rather than bouncing across the
landscape at high LR. SAM during the chaotic high-LR phase buys little basin-selection and
costs the most epochs.

Cost model for `SAM_TAIL_FRAC = f` (fraction of the time budget run under SAM):
the high-LR `(1−f)` fraction runs at full ~26.5k img/s; the `f` tail runs at ~half. Epochs
≈ `150 · (1 − f/2)`. For **f = 0.35**: ≈ `150 · 0.825 ≈ 124 epochs` — above the under-anneal
line. For f = 0.40: ≈120. The schedule is **time-based and keyed on `total_training_time`**,
so the anneal still completes to LR≈0 regardless — the cost shows up purely as fewer
*updates inside the tail*, i.e. each tail LR value is seen for fewer steps. The bet is that
SAM's better basin selection per step more than compensates for fewer tail steps, because
the *whole point* of the tail is settling, and SAM settles into flatter basins.

### 3c. Why this could clear the ~0.1pp noise floor

The published SAM gain on CIFAR ResNets at matched epochs is +0.3–1.0pp — 3–10× the
~0.1pp noise floor. Even if tail-only SAM captures only a fraction of the full-SAM gain
(it applies SAM to perhaps ~half the steps, the ones that matter most), capturing ≥0.2pp
of a 0.5pp full-SAM effect while losing ~26 epochs (150→124) — fewer than the EXP-004
142-epoch run that still won — is a plausible net positive. This is a genuine coin-flip,
not a near-certainty: the honest framing is *"a real chance to clear noise, gated on the
flat-minima gain surviving the modest tail-step reduction."*

## 4. Concrete change in THIS codebase

All edits are in `train.py`. No change to `prepare.py`, the model, the schedule math, EMA,
TTA, or aug.

**(a) Hyperparameters** (after line 30, near the other knobs):

```python
SAM_RHO = 0.05          # SAM neighborhood radius (canonical CIFAR-10 default)
SAM_TAIL_FRAC = 0.35    # apply SAM only when progress >= this (low-LR tail); SGD before
```

**(b) The training step** — currently lines 299–304:

```python
optimizer.zero_grad(set_to_none=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
loss.backward()
optimizer.step()
```

Replace with a gated SAM step. The gate reuses the existing `progress` value already
computed at line 286 for the LR schedule (`progress = min(1.0, total_training_time /
TIME_BUDGET_S)`), so no new timing call is needed:

```python
use_sam = progress >= SAM_TAIL_FRAC

if not use_sam:
    # ---- plain SGD-Nesterov (byte-identical to current step) ----
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.step()
else:
    # ---- SAM: ascent to w + e_w, descent with grad at the perturbed point ----
    # 1st forward-backward at w (running BN stats update HERE only)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = criterion(model(inputs), targets)
    loss.backward()

    # ascent step: e_w = rho * g / (||g|| + eps); climb to worst-case neighbor
    with torch.no_grad():
        grad_norm = torch.norm(
            torch.stack([
                p.grad.detach().norm(2)
                for p in sam_params if p.grad is not None
            ]), 2
        )
        scale = SAM_RHO / (grad_norm + 1e-12)
        for p in sam_params:
            if p.grad is None:
                continue
            e_w = p.grad * scale          # ascent direction (full precision params)
            p.add_(e_w)                   # w -> w + e_w
            sam_state[p] = e_w            # remember to undo exactly

    # 2nd forward-backward at the perturbed point; DO NOT update BN running stats
    optimizer.zero_grad(set_to_none=True)
    model.apply(_bn_freeze_stats)         # momentum 0 on the perturbed pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = criterion(model(inputs), targets)   # loss reported = perturbed loss
    loss.backward()
    model.apply(_bn_restore_stats)

    # restore original weights, then take the real optimizer step with perturbed grad
    with torch.no_grad():
        for p in sam_params:
            e_w = sam_state.get(p)
            if e_w is not None:
                p.sub_(e_w)               # w + e_w -> w
    optimizer.step()                      # Nesterov/momentum/wd applied to perturbed grad
```

**(c) Supporting setup** (once, before the loop, after the optimizer is built ~line 250):

```python
sam_params = [p for p in model.parameters() if p.requires_grad]
sam_state = {}
```

**(d) BN running-stats freeze on the perturbed pass.** Standard SAM updates BN running
stats only on the first pass (davda54/sam tip: stats "get computed in both passes but
should only count for the first"; the simplest correct fix is to zero BN momentum on the
second pass). Add module helpers near the top of `train.py`:

```python
def _bn_freeze_stats(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m._sam_saved_momentum = m.momentum
        m.momentum = 0.0          # running_mean/var unchanged on perturbed forward

def _bn_restore_stats(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm) and hasattr(m, "_sam_saved_momentum"):
        m.momentum = m._sam_saved_momentum
```

(Setting momentum=0 keeps `track_running_stats=True` so the layer still *uses* batch stats
for normalization — correct — but does not *update* the running buffers. This matters
because the EMA model averages BN buffers with `use_buffers=True`; double-counting the
perturbed batch would bias them.)

Notes on correctness that a planner must preserve:
- `e_w` is computed and applied to the **master fp32 parameters** (autocast only casts
  inside the forward; `p` and `p.grad` are fp32), so the perturbation is not bf16-rounded.
- The grad norm is the **global** 2-norm over all trainable params (standard SAM, not
  per-layer ASAM). The frozen whitening conv is excluded automatically (`requires_grad`
  filter), matching the optimizer's param list at line 245.
- `loss.item()` for logging should come from the **first** (unperturbed) pass to keep the
  printed train-loss comparable to prior runs; reassign `loss` capture accordingly.
- EMA update (lines 308–310) and everything below are unchanged — `optimizer.step()` has
  already restored `w`, so EMA sees the correct post-step weights.

## 5. Sources

- **Foret, Kleiner, Mobahi, Neyshabur — "Sharpness-Aware Minimization for Efficiently
  Improving Generalization," ICLR 2021** (arXiv:2010.01412). Core method: `e_w = ρ·g/‖g‖`
  ascent then descent with the perturbed-point gradient. **ρ = 0.05 is the reported default
  used across CIFAR-10 and ImageNet without further tuning.** Reports SOTA 0.30% *error*
  on CIFAR-10 (with heavy backbones / Shake-Shake, not directly our ResNet-9) and
  +0.4–1.9pp top-1 across the ResNet family on ImageNet. Notably: SAM *enables more epochs
  without overfitting*, whereas plain training overfits — relevant because our risk is the
  opposite (too few epochs).
- **Follow-up ρ guidance:** for CIFAR-10 ρ = 0.05–0.1 is standard; theory work finds
  ρ = 0.1 can be best on CIFAR-10 and that *overestimating ρ is less harmful than
  underestimating* — motivates the {0.05, 0.10} fallback sweep.
- **Mixed-precision caveat (load-bearing risk):** multiple sources note SAM "is not
  numerically stable under mixed precision training" and can cost up to ~4× wall in some
  setups. We mitigate by computing `e_w` and the grad-norm on the fp32 master params with
  autocast confined to the forward — but this is the assumption that most needs validation
  (see §7). bf16 (our dtype) has wider dynamic range than fp16, which helps.
- **davda54/sam (PyTorch reference, MIT):** confirms the two-pass usage and the BN-stats
  tip (compute running stats only on the first pass / set BN momentum to zero on the
  second). We hand-implement equivalently — no dependency added.
- **Efficient variants surveyed (and why we do NOT use them):** LookSAM / ESAM / SAM-k
  periodic methods trade accuracy for throughput; the recurring finding is that *naive
  periodic SAM-k degrades accuracy* and LookSAM loses accuracy past a threshold interval.
  This is the explicit reason the proposal prefers a **temporal tail-only** split (spend
  SAM where it matters, full SGD elsewhere) over periodic SAM. ESAM's sparse-weight
  perturbation is an additional dependency-free lever held in reserve if tail-only
  under-anneals.

Source links:
- https://arxiv.org/abs/2010.01412 (Foret et al., ICLR 2021)
- https://openreview.net/pdf?id=6Tm1mposlrM
- https://github.com/davda54/sam
- https://docs.mosaicml.com/projects/composer/en/latest/method_cards/sam.html
- https://sh-tsang.medium.com/brief-review-sharpness-aware-minimization-for-efficiently-improving-generalization-8a484db8c7e9
- https://arxiv.org/html/2406.08001v1 (efficient-SAM sampling survey, throughput context)

## 6. Estimated effort: **Medium**

One experiment loop. The change is contained to the training step plus ~15 lines of
setup/helpers, all in `train.py`. Medium (not low) because of three correctness traps that
must be gotten right or the result is silently wrong: (a) BN double-forward stat handling,
(b) exact perturbation restore before `optimizer.step()`, (c) ensuring momentum/weight-decay
are applied to the *perturbed-point* gradient (the `optimizer.step()` ordering above does
this for free since SGD reads `p.grad` after restore). Not high — no new architecture, no
new dependency, schedule/EMA/TTA untouched.

## 7. Risk assessment (honest)

**Dominant risk — under-anneal (the recurring failure, count 2).** Even tail-only SAM cuts
epochs; if the host is loaded (the noise-floor entry shows 131–150 epoch jitter on
identical code) a tail-only run could drop below the ~110 under-anneal line and lose on
epoch count alone, independent of any SAM merit. **Mitigation / decision rule:**
`num_epochs` is the first-read diagnostic. If a SAM cell finishes <115 epochs, treat the
accuracy as confounded by under-anneal (per the EXP-007 precedent) and *do not* conclude
SAM failed — re-run at a smaller `SAM_TAIL_FRAC` (e.g. 0.25). Quantified expectation:
plain SAM ~75–80 ep (expected loss), tail-only f=0.35 ~124 ep, f=0.40 ~120 ep.

**Will the flat-minima gain clear noise at reduced tail-steps?** This is a genuine
coin-flip, not a likely win. The +0.3–1.0pp literature numbers are at *matched* epochs and
mostly on heavier backbones (WRN-28-10, PyramidNet, Shake-Shake) than DavidNet/ResNet-9;
the gain on a small wide-shallow net under a 300s one-cycle is unproven. Tail-only further
dilutes the gain by applying SAM to only ~half the steps. Realistic outcome distribution:
~40% clears +0.1pp, ~35% ties within noise, ~25% loses to under-anneal. Worth it precisely
because it is the *one untried high-upside axis the diagnosis itself names*.

**Mixed-precision stability (bf16).** SAM is documented as numerically unstable under mixed
precision. The ascent direction `g/‖g‖` divides by a norm that, if computed in low
precision over an autocast region, can be noisy. Mitigation (above): compute `e_w` and the
global grad-norm on fp32 master params, autocast only the forward. bf16's wide exponent
range makes this safer than fp16. **This is the assumption most needing validation** — if
the first SAM epochs show loss spikes or NaNs, the fix is to (i) confirm the norm is fp32,
(ii) add the `1e-12` eps (present above), (iii) as a last resort run the SAM forwards
without autocast (slower, fewer epochs — likely fatal to the budget).

**Interaction with EMA + ReZero α.** The EMA averages BN buffers; the BN-momentum-zero
fix on the perturbed pass is *required* to avoid double-counting batches into the running
stats the EMA then averages. The ReZero `alpha` is in `sam_params` and gets perturbed like
any weight — fine, it is just another trainable scalar; no special-casing needed.

**Throughput estimate could be optimistic.** If the second forward-backward is *more* than
2× (autocast re-entry, BN apply/restore overhead, lost cudnn.benchmark autotune reuse), the
tail epochs drop further. The `model.apply(...)` BN walks are cheap (a handful of modules)
but the honest position is: read `num_epochs` first, accuracy second.

**Fallback ladder if the primary cell under-anneals or ties:** (1) shrink `SAM_TAIL_FRAC`
0.35→0.25; (2) ρ sweep {0.05, 0.10}; (3) ESAM-style sparse weight perturbation (perturb a
random ~50% subset of params each step) to cut the ascent cost — all dependency-free, all
within this same train.py scaffold.
