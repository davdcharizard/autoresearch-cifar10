# EXP-011 Proposal (idea-04, MOONSHOT): Tail-gated Sharpness-Aware Minimization (SAM)

## Summary

Implement Sharpness-Aware Minimization (Foret et al. 2021, arXiv:2010.01412) from scratch
in `train.py` and apply it **only in the low-LR tail** of the time-based one-cycle schedule,
keeping plain SGD-Nesterov (the proven recipe) for the first ~75% of the budget. SAM seeks
**flat minima** by perturbing weights to a local worst case w+ε before each gradient step;
flat minima are the diagnosed limiter for this net (it is regularization-/generalization-bound,
confirmed by EXP-010 where tuned Muon only *matched* SGD). The central problem with SAM here is
that it **doubles per-step cost** (two forward/backward passes), which under the fixed 300s budget
would roughly halve the epoch count from ~150 to ~75 — squarely inside the **under-anneal trap**
(EXP-005/007 both lost when epochs fell to ≤110 / 94). The tail-gated variant is engineered
precisely to dodge that trap: it pays the 2× cost only over the final ~25% of training (where
both accuracy and flat-minima-seeking concentrate), so the realized epoch count stays ~120–130,
above the ~110 under-anneal threshold.

**Honest framing:** this is the highest-risk idea in the portfolio. SAM's generalization gain is
real and well-documented (+0.3–1.0pp on CIFAR), but the EXP-010 result is a genuine yellow flag —
when SGD already trains stably to a fully-annealed minimum, "better optimizer" interventions have
repeatedly landed at the *same* minimum here. SAM is mechanistically different from Muon (it changes
the *loss geometry sought*, not the *update direction/scale*), so the EXP-010 falsification does not
directly transfer, but it lowers the prior. I recommend the tail-gated variant with a clear
kill-criterion, and I am explicit below about the probability it clears the bar.

## Mechanism (causal chain to the metric)

1. **Limiter (from diagnosis):** This whitened ResNet-9 at 300s is regularization-bound, not
   optimizer-convergence-bound nor capacity-bound. Evidence: tuned Muon = 96.33 vs SGD 96.38 (tie
   within noise, EXP-010); the only >noise win since EXP-001 was *augmentation* (EXP-008, +0.38pp);
   there is a ~4× epoch surplus vs airbench96. The model fits its training distribution well and the
   gap is generalization.
2. **What SAM changes:** SAM replaces the SGD objective `min L(w)` with the min-max
   `min_w max_{‖ε‖≤ρ} L(w+ε)` (arXiv:2010.01412 Eq. 1). The practical first-order solution: compute
   g = ∇L(w), set ε = ρ·g/‖g‖₂ (Eq. 2-3), then take the *actual* optimizer step using the gradient
   evaluated at the perturbed point, ∇L(w+ε). Minimizing the loss at the worst nearby point biases
   optimization toward **wide, flat basins** where the loss is robust to weight perturbation.
3. **Why flat minima → higher test accuracy here:** flat minima generalize better (the paper's core
   empirical claim and its PAC-Bayes bound, arXiv:2010.01412 §2-3). Because this net is exactly
   generalization-bound, an intervention that *directly* targets flatness is the right class of lever
   (same class as augmentation, which won; unlike the optimizer-direction levers, which tied).
4. **Why the TAIL specifically:** EXP-001 established that most accuracy lands in the low-LR tail of a
   completing one-cycle; that is also where the iterate settles into a basin, so it is where the
   *shape* of the basin (flat vs sharp) is determined. Applying SAM only there concentrates the 2×
   cost on the phase that matters most for flatness, and aligns it with the EMA-tail and flip-TTA-tail
   phases that already do their work there (synergy, §4 below).
5. **Net effect on the metric:** if SAM lifts the per-epoch generalization ceiling by more than the
   tail's halved epoch rate costs in anneal completeness, `best_test_acc` rises. The whole proposal is
   an argument that the tail-gating keeps the epoch loss small enough for the flatness gain to win.

## Concrete from-scratch code plan

All edits are in `train.py`. No new deps (SAM hand-rolled). The current training step is
`train.py:299-304` (`zero_grad; autocast forward; loss.backward(); optimizer.step()`), and the
time-based progress variable is computed at `train.py:286`.

### 1. Helper functions (module scope, near the other helpers)

```python
RHO = 0.05                 # SAM perturbation radius (paper's CIFAR value)
SAM_START_FRAC = 0.75      # turn SAM on only for the final 25% of the budget (tail)

@torch.no_grad()
def _grad_norm(params):
    # global L2 norm across all params with a grad (the SAM normalizer)
    return torch.norm(
        torch.stack([p.grad.norm(2) for p in params if p.grad is not None]), 2
    )

@torch.no_grad()
def _sam_ascend(params, rho, eps_store):
    # ε = ρ · g/‖g‖ ; move to w+ε, remember ε to undo later
    scale = rho / (_grad_norm(params) + 1e-12)
    for p in params:
        if p.grad is None:
            continue
        e_w = p.grad * scale            # bf16 grads are fine; this is a small additive step
        p.add_(e_w)
        eps_store[p] = e_w

@torch.no_grad()
def _sam_restore(params, eps_store):
    for p in params:
        if p in eps_store:
            p.sub_(eps_store[p])
    eps_store.clear()
```

`params` is the SAM-managed list = the same trainable list passed to the optimizer
(`[p for p in model.parameters() if p.requires_grad]`, `train.py:245`). The frozen whitening conv
has `requires_grad=False` so it is naturally excluded (no grad → skipped in both helpers).

### 2. Modified training step (replace `train.py:299-304`)

```python
sam_on = progress >= SAM_START_FRAC

optimizer.zero_grad(set_to_none=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
loss.backward()                          # g = ∇L(w)  (first pass)

if sam_on:
    # ---- SAM second pass ----
    _sam_ascend(sam_params, RHO, eps_store)        # w -> w+ε
    optimizer.zero_grad(set_to_none=True)
    # Freeze BN running-stat updates on the ascent pass (see §BN note)
    _set_bn_momentum(model, 0.0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs2 = model(inputs)
        loss2 = criterion(outputs2, targets)
    loss2.backward()                               # g_SAM = ∇L(w+ε)
    _set_bn_momentum(model, bn_mom_default)
    _sam_restore(sam_params, eps_store)            # w+ε -> w (grads now hold g_SAM)

optimizer.step()                          # Nesterov-SGD step using whichever grad is present
```

Key correctness points (all faithful to arXiv:2010.01412):
- **Order:** restore `w` *before* `optimizer.step()`, so the update is applied at the original
  weights `w` using the perturbed-point gradient `g_SAM` — this is exactly SAM (the optimizer's
  momentum buffer accumulates `g_SAM`, which is correct).
- **Grad norm across params, bf16:** `_grad_norm` stacks per-tensor norms then takes the global
  norm. Under bf16 autocast, `loss.backward()` accumulates grads in the parameters' own dtype
  (fp32 master params here — the model params are fp32; autocast only affects the forward op dtype),
  so `p.grad` is fp32 and the norm is numerically safe. No GradScaler is used (consistent with the
  recipe, which relies on bf16's fp32-equivalent range).
- **ε is small and additive:** ρ=0.05 with ‖g‖ normalization gives ‖ε‖=ρ=0.05, a tiny weight
  displacement; storing one `e_w` tensor per param doubles the transient param-memory for the
  managed set (~7.8M params × 4B ≈ 31 MB) — trivial against the 98 GB envelope (VRAM is a free lever,
  EXP-001).

### 3. BN running-stat subtlety (the one real correctness gotcha)

Both forward passes update BN running mean/var by default. The ascent pass runs at `w+ε` on the same
batch the descent pass already saw — letting it pollute the running stats with worst-case-perturbed
activations is off-distribution for inference. Standard SAM practice (and several reference
implementations) **disables BN running-stat updates on the ascent pass**. Implement with a tiny
helper that zeroes BN momentum during the second forward:

```python
def _set_bn_momentum(model, m):
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.momentum = m   # 0.0 = no running-stat update this pass
```

`bn_mom_default` is captured once (PyTorch default 0.1). This keeps the EMA-of-BN-buffers
(`use_buffers=True` in `AveragedModel`, `train.py:256`) clean: the EMA only ever averages BN stats
produced at real weights `w`. **Cost-saving alternative considered and rejected:** one could instead
accept the double BN update (simpler, what the original SAM repo did before refinements) — but given
the recipe leans on EMA'd BN buffers at eval, keeping them clean is worth the four-line helper. The
gradient w.r.t. BN's affine γ/β is unaffected by `momentum=0` (momentum only controls the running-stat
EMA, not the forward normalization or its backward), so the SAM gradient is still correct.

### 4. Setup additions (near `train.py:244-251`)

```python
sam_params = [p for p in model.parameters() if p.requires_grad]
eps_store = {}
bn_mom_default = 0.1
```

No change to the optimizer construction, LR schedule (`train.py:286-292`), EMA gate
(`train.py:308-310`), or TTA gate (`train.py:342`).

## Recommended variant + config

**Recommended: TAIL-GATED SAM** (`SAM_START_FRAC = 0.75`, `RHO = 0.05`, BN frozen on ascent).

Rationale for choosing tail-gated over the two alternatives:
- **Full-time SAM:** rejected. Realized epochs ≈ 150 / ~1.9 ≈ **78** (the 2× passes are not exactly
  2× wall because the optimizer step, dataloading, and EMA update are not duplicated; empirically SAM
  is ~1.8–1.9× per-step). 78 epochs is deep in the under-anneal zone (EXP-007 lost at 94, EXP-005 at
  131-vs-142 marginal). This almost certainly under-anneals and goes net-negative — the exact failure
  the diagnosis warns against.
- **Periodic-k SAM** (SAM second pass every k steps): cheaper but dilutes the flatness signal across
  the *whole* schedule including the high-LR phase where basin shape is not yet set. It also keeps an
  awkward partial throughput tax everywhere. Lower expected gain per unit risk than concentrating the
  full SAM treatment on the tail. Reasonable fallback if tail-gated under-anneals.
- **Lower ρ:** orthogonal knob, not a cost-reducer (ρ does not change pass count). Keep ρ=0.05 (the
  paper's tuned CIFAR value); do not scale it over the schedule for the first shot (one variable).

**Config summary:** `RHO=0.05`, `SAM_START_FRAC=0.75`, BN momentum 0→0 on ascent pass, everything
else byte-identical to the EXP-008 base recipe (SGD-Nesterov lr 0.4 wd 5e-4 nesterov, LS 0.2,
batch 512, bf16/channels_last, time-based one-cycle PCT_START 0.15, EMA 0.998 warmup 0.15,
flip-TTA from 0.8).

## Throughput / epoch analysis (the crux)

Base recipe fits ~150 epochs in 300s at full throughput (~26k img/s; 142–150 observed, EXP-004/008).
Let the plain-SGD per-step time be `t`. SAM steps cost ~`1.85t` (two forward/backward, but
single dataload/EMA/step; the BN-momentum toggles are negligible).

- **Fraction of budget at plain rate:** 0..0.75 of *training time* → 0.75 × 150 = **112.5 epochs**
  worth of work happens in the first 75% of the budget (time-keyed schedule, so this is exact in
  time, approximate in epochs).
- **Tail (final 25% of budget) at SAM rate:** the same 25% of *time* now fits 1/1.85 as many steps →
  0.25 × 150 / 1.85 ≈ **20.3 epochs** of work.
- **Realized total ≈ 112.5 + 20.3 ≈ 133 epochs** at full host throughput, dropping to ~115 under the
  lighter end of observed throughput (132-epoch runs → ~117). Both are **above the ~110 under-anneal
  threshold** — the design intent.

Critically, **the schedule is time-keyed** (`train.py:286`, `progress = total_training_time /
TIME_BUDGET_S`), so the LR still anneals fully to ~0 by the budget end regardless of how many steps
fit. The tail is not *truncated* (as in EXP-007); it just has *fewer, more expensive* steps, each of
which seeks a flatter point. The risk is therefore not "schedule cut off mid-anneal" but "the tail's
low-LR refinement gets ~20 epochs instead of ~37" — a softer degradation. This is the key structural
difference from the capacity-add failures: those lost the *anneal*, this keeps the anneal and trades
step *count* for step *quality* only in the tail.

**Sanity contrast with full-time SAM:** 150/1.85 ≈ 81 epochs end-to-end → would lose the whole
early-schedule throughput and land at ~78-81, under-annealed everywhere. Tail-gating recovers ~52
epochs of that loss by paying the tax only where it buys flatness.

## ρ (perturbation radius)

Use **ρ=0.05**, the value Foret et al. tune for CIFAR-10/100 (arXiv:2010.01412 §4, Table 1/Fig 3;
ρ∈{0.05,0.1} optimal for CIFAR). Our weight scale is conventional (kaiming-init convs, BN-normalized
activations, logits ×0.125), so the paper's CIFAR ρ should transfer without rescaling — SAM's
g/‖g‖ normalization makes ε scale-invariant to gradient magnitude, and ρ is in *weight* units which
are standard here. I do **not** recommend scheduling ρ for the first shot (keep it a single clean
variable). A reasonable second-shot knob if the first lands close: try ρ=0.1 (some CIFAR setups
prefer it), or ramp ρ up across the tail (flatter-seeking pressure increasing as LR→0).

## Interaction with EMA + one-cycle tail (synergy argument)

The tail (final ~25%) is already the EMA-and-TTA-active phase:
- **EMA** (decay 0.998, started at 15% warmup, `train.py:308-310`) averages the iterates throughout,
  including the SAM-on tail. EMA of SAM iterates is *complementary*: EMA denoises the iterate's
  short-horizon jitter, SAM biases each iterate toward a flat basin. Averaging iterates that already
  live in a flat basin should produce a centered, flat-basin point — the two stack rather than fight.
  And because BN stats are frozen on the ascent pass (§3), the EMA'd BN buffers stay clean.
- **flip-TTA** gates on at progress≥0.8 (`train.py:342`), inside the SAM-on window. No interaction
  (TTA is eval-side), but it means the same tail phase is where all three eval-side/tail levers
  concentrate — consistent operating point.
- **No LR retune needed:** SAM perturbs weights, not the optimizer's LR semantics; the existing
  one-cycle and Nesterov momentum are unchanged. (Note: SAM's ε is computed from the *raw* gradient,
  not the Nesterov-lookahead gradient — this matches the standard SAM-over-SGD composition; the
  Nesterov lookahead happens inside `optimizer.step()` using the SAM gradient, which is fine.)

## Risks & de-risking + kill-criterion

**Highest-risk idea in the portfolio.** Candid failure modes, most→least likely:

1. **Optimizer-axis null result (most likely failure).** EXP-010 showed tuned Muon ties SGD because
   SGD already reaches the net's minimum. SAM is a *different class* of intervention (it changes the
   geometry sought, not the update rule), so the falsification doesn't directly transfer — but it
   warns that "better optimization" of an already-well-trained net tends to land at the same place.
   If SGD's tail minimum is *already fairly flat* (one-cycle's high-LR phase is itself a flatness
   regularizer — large-batch/large-LR seek flat regions), SAM's marginal flatness gain may be small
   and below the ~0.1pp bar. **De-risk:** none structural; this is the moonshot bet. The pre-registered
   read (below) will reveal it via the tail trajectory.
2. **Under-anneal anyway (host contention).** If GPU 1 is contended (as in EXP-010's trial phase,
   throughput halved), realized epochs could fall well below 110 and the tail's expensive steps make
   it worse than plain SGD would be. **De-risk:** read `num_epochs` first; if <110, the comparison is
   confounded and the result is inconclusive, not a verdict on SAM.
3. **BN/EMA interaction bug.** If the BN-freeze helper is mis-wired (e.g. momentum not restored),
   running stats drift. **De-risk:** a one-step smoke test asserting BN running_mean is byte-identical
   before/after an ascent-only pass, and that `optimizer.step()` consumes `g_SAM` (check a param moved
   in the g_SAM direction, not g).

**Kill-criterion (pre-registered):** abandon the SAM axis (do not iterate on ρ/k variants) if EITHER
(a) the run completes with `num_epochs ≥ 115` (clean, not under-annealed) AND `best_test_acc < 96.38`
(no gain over baseline) — this is the EXP-010-style "optimizer helped nothing" verdict for SAM; OR
(b) the run under-anneals (`num_epochs < 110`) AND best is below baseline — inconclusive on SAM, but
tail-gated SAM is too throughput-fragile for this host, so stop. Only iterate (try periodic-k or
ρ=0.1) if best lands in [96.33, 96.48) with epochs ≥115 — a near-miss worth one refinement.

## Expected effect (pp + probability)

- **If SAM's flatness gain materializes at full strength** (paper-scale +0.3–0.5pp on CIFAR with a
  strong base), tail-gated SAM captures maybe 40–60% of that (only the tail gets it) minus the
  ~20-vs-37 tail-epoch cost → net **+0.10 to +0.25pp** (96.48–96.63).
- **If SAM ties (EXP-010 scenario)** the flatness of SGD's tail minimum is already near-optimal →
  **−0.05 to +0.05pp** (within noise, no improvement), possibly slightly negative from the tail
  epoch loss.
- **If it under-anneals** (contention) → **−0.1 to −0.3pp**.

**Probability of clearing the +0.10pp bar AND >noise: ~25–30%.** This is genuinely lower than the
augmentation-class ideas in the portfolio, for the honest reason that EXP-010 demonstrated this net's
minimum is robust to optimizer changes. SAM's *mechanistic* distinctness from Muon (geometry vs
direction) is what keeps the probability from being lower — flatness-seeking is in the *same class* as
augmentation (the one lever that has worked), not the same class as optimizer-direction swaps (the
levers that tied). The realistic central outcome is a tie within noise; the upside (a clean +0.2pp) is
real but a minority of the probability mass. Recommend running it as the deliberate high-variance bet
of the experiment with the kill-criterion firmly applied.

## Pre-registered read

Record and compare against the EXP-008 trajectory (ep25 92.31 → 96.38 tail):
1. **`num_epochs` first** — the gating diagnostic. ≥115 = clean (verdict valid); <110 = under-anneal
   (inconclusive, see kill-criterion b).
2. **Early trajectory (ep25, ep50, ep75) must match plain SGD** (~92.3 / ~93.75 / ~94.8) since SAM is
   OFF until 75% progress. If it diverges early, the gate is mis-wired.
3. **Tail trajectory (from the SAM-on gate ~ep100 onward):** the make-or-break read. SAM working →
   the tail should climb *above* the SGD baseline tail and hold a higher flat plateau (best > 96.38).
   SAM null → tail tracks or slightly trails SGD (fewer tail epochs, same minimum). Look specifically
   for a step-change in slope at the SAM gate.
4. **best vs final:** if best==final and still rising at budget end → under-annealed (tail too short),
   maps to kill-criterion (b). If best is a clear plateau peak → fully annealed, verdict valid.
5. **peak_vram_mb** sanity (should be ~base + tens of MB; no surprise blow-up from eps_store).

## Effort

**Medium.** ~40 lines of new code in one file (helpers + a ~10-line edit to the training step +
3 setup lines), no new deps, no LR retune, no architecture change. The only fiddly parts are the
BN-momentum toggle and the ascend/restore-around-step ordering, both covered above. One smoke test +
one full run. Comparable to one experiment loop.

## Citation

Foret, Kleiner, Mobahi, Neyshabur. "Sharpness-Aware Minimization for Efficiently Improving
Generalization." arXiv:2010.01412 (ICLR 2021). ρ=0.05 CIFAR value, Eq. 1-3 (min-max objective,
first-order ε solution), §2-3 (flat-minima / PAC-Bayes motivation), §4 (CIFAR results, +0.3–1.0pp).
