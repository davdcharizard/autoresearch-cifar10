# Idea-02: Cheap robust eval/optimization boosters on DavidNet — weight EMA + flip-TTA (+ light recipe tune)

## Summary

Keep EXP-001's proven ResNet-9/DavidNet architecture and time-based one-cycle recipe
(95.22% baseline) completely intact, and bolt on two low-risk, well-evidenced
generalization boosters that do not touch the core training dynamics:

1. **Weight EMA** — maintain an exponential moving average of the model weights during
   training and evaluate the EMA weights each epoch (instead of the raw SGD iterate).
   Polyak/EMA averaging of the trajectory is a near-free smoothing of the final iterate;
   under a one-cycle schedule the raw weights are still "ringing" along the loss surface,
   and the average sits in a flatter, better-generalizing basin.
2. **Flip test-time augmentation (TTA)** — inside `ResNet9.forward`, gated on
   `not self.training`, average the logits of `x` and `x.flip(-1)` (horizontal mirror).
   The frozen eval calls `model(inputs)` directly, so this is a legitimate eval-time
   change that lives entirely inside `train.py`'s model code.

Both are standard "bag of tricks" boosters that compose additively (EMA improves the
weights being evaluated; TTA improves how those weights are read). A small, optional,
evidence-gated recipe tweak (label_smoothing 0.2 -> 0.1) is included as a secondary knob.

Implementation is via `torch.optim.swa_utils.AveragedModel` with `get_ema_multi_avg_fn`
(already in `torch==2.9.1`, **no new dependency**) plus `use_buffers=True` so BatchNorm
running stats are EMA-averaged inline — avoiding a separate `update_bn` pass that would
burn training-budget time.

Expected: **95.5–95.9%** best_test_acc (vs 95.22%), i.e. +0.3–0.7pp, comfortably above the
+0.1pp bar. Effort: **low**.

---

## What it targets (mechanism tied to the diagnosis)

EXP-001's analysis (`experiments/001/04-analysis.md`) found the bulk of accuracy gain
arrives in the **low-LR tail** of the one-cycle anneal (ep102 89.9% -> ep192 95.22%), and
named the remaining gap to the ~96% fast-CIFAR ceiling as a *generalization/robustness*
problem rather than an under-training one. The "Unexplored Avenues" / "Next Steps"
sections explicitly list **EMA of weights** and **flip TTA inside `forward`** as cheap,
untried tenths-of-a-pp levers (lines 35, 37, 42). This proposal develops exactly those.

Causal chain to the metric:

- **EMA -> metric.** The reported `best_test_acc` is `max` over per-epoch
  `evaluator.evaluate(...)`. Today that evaluates the *raw* SGD iterate. Under a one-cycle
  schedule with a high peak LR (0.4) and ~96 steps/epoch, the iterate at any single epoch
  boundary is a noisy sample of a weight trajectory that is still oscillating. Averaging
  the trajectory (EMA) is a variance-reduction operator on the evaluated weights: it
  removes the per-step gradient noise and lands in a flatter minimum that generalizes
  better. This directly raises the number `evaluate` returns at each epoch, hence `best_acc`.
- **Flip-TTA -> metric.** CIFAR-10 classes are (near-)horizontal-flip invariant; the train
  pipeline already uses `RandomHorizontalFlip`, so the model is trained to be flip-robust
  but not perfectly flip-invariant. Averaging logits over `x` and `flip(x)` is a Bayes-style
  ensemble of two correlated views that cancels view-specific errors, directly improving
  per-batch `argmax` correctness inside the frozen `evaluate` loop.

Neither change alters the training optimization, so EXP-001's hard-won 95.22% trajectory is
preserved as a floor (modulo the tiny per-step EMA-update overhead, quantified below).

---

## Exact `train.py` changes

All changes are inside `train.py` (the only editable file). `prepare.py` is untouched.

### 1. Flip-TTA gate inside `ResNet9.forward`

Current `forward` (lines 101–107):

```python
def forward(self, x):
    x = self.prep(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.pool(x).flatten(1)
    return self.fc(x) * self.scale_out
```

Refactor the feature path into a private `_forward_once` and add an eval-gated flip branch:

```python
def _forward_once(self, x):
    x = self.prep(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.pool(x).flatten(1)
    return self.fc(x) * self.scale_out

def forward(self, x):
    if self.training:
        return self._forward_once(x)
    # Eval-time flip TTA: average logits of the image and its horizontal mirror.
    return 0.5 * (self._forward_once(x) + self._forward_once(x.flip(-1)))
```

Why this works with the frozen interface: `prepare.py`'s `Eval.evaluate` calls
`model.eval()` then `outputs = model(inputs)` (lines 34, 42). `model.eval()` sets
`self.training = False` on the module (and, with EMA, propagates through the
`AveragedModel` wrapper to `self.module` — verified, see Reasoning), so the `not training`
branch fires only at eval. Averaging logits (not softmax) matches airbench's `infer_mirror`
TTA and keeps the loss computed in `evaluate` well-behaved (`scale_out`=0.125 already keeps
logits small). `x.flip(-1)` mirrors the width dimension (CHW -> last axis = W), the correct
horizontal flip.

### 2. Weight EMA via `torch.optim.swa_utils.AveragedModel`

Add the import (top of file, alongside existing torch imports):

```python
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
```

After the raw `model` is built and moved to device (after line 151), create the EMA model:

```python
EMA_DECAY = 0.998  # per-step decay; see cadence analysis for the choice
ema_model = AveragedModel(
    model,
    multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY),
    use_buffers=True,  # EMA-average BN running_mean/var inline (no update_bn pass)
).to(device, memory_format=torch.channels_last)
```

`AveragedModel` does `self.module = deepcopy(model)` and its `forward` delegates to
`self.module(*args, **kwargs)` (verified from `torch/optim/swa_utils.py` v2.9.0), so it is a
drop-in `nn.Module` that runs the same flip-TTA-augmented `forward` as the raw net.

In the training loop, after `optimizer.step()` (after line 206), update the EMA:

```python
ema_started = False
EMA_WARMUP_FRAC = 0.15  # don't average the high-noise ramp phase
...
# inside the loop, after optimizer.step():
if progress >= EMA_WARMUP_FRAC:
    ema_model.update_parameters(model)
    ema_started = True
```

On the **first** `update_parameters` call (`n_averaged == 0`) `AveragedModel` *copies* the
raw params/buffers into the EMA model, so starting EMA at 15% progress seeds it with a
sane net rather than the kaiming-init weights — important because we average across the
remaining 85% of the budget where the real accuracy gains happen.

### 3. Eval the EMA model (the key interface point)

Change the per-epoch eval (line 234) to evaluate the EMA model once it has started,
and fall back to the raw model in the warmup epochs:

```python
eval_model = ema_model if ema_started else model
test_loss, test_acc = evaluator.evaluate(eval_model, device)
```

`evaluator.evaluate(eval_model, device)` is exactly the frozen call signature
(`prepare.py` line 33: `def evaluate(self, model, device)`). It calls `model.eval()` on
whatever object we hand it; passing `ema_model` (an `AveragedModel`) makes `model.eval()`
propagate to the wrapped `self.module`, and `model(inputs)` runs the EMA weights through the
TTA forward. **This is the load-bearing feasibility fact and it is verified** (see Reasoning).
This still satisfies "≤1 val/epoch": exactly one `evaluator.evaluate` call per epoch.

BN-stat handling: with `use_buffers=True`, `running_mean`/`running_var` are EMA-averaged
with the same decay as the weights during `update_parameters`, so the EMA weights are
evaluated against EMA-consistent BN stats — no train-time/eval-time BN mismatch, and no
extra forward passes spent on `update_bn` (which would cost training budget). Tradeoff vs.
the alternative (`use_buffers=False` + post-hoc `update_bn`): `update_bn` recomputes exact
BN stats for the averaged weights on a few train batches and is sometimes marginally more
accurate, but it requires extra forward passes (eating into the 300s budget or adding wall
time) and a train-loader pass at eval. Inline buffer-EMA is the lower-risk, lower-cost
choice here; if it underperforms, the fallback is a single `update_bn` over ~5 batches
*after* the time loop ends (off training-budget, on wall-clock — see cadence note).

### 4. Optional recipe tweak (evidence-gated, secondary)

EXP-001 used `LABEL_SMOOTHING = 0.2`, which is on the high side. "Bag of Tricks"
(arXiv:1812.01187) and most fast-CIFAR recipes use **0.1**. With EMA+TTA already smoothing
predictions, 0.2 may over-soften. This is a **one-line, reversible** change
(`LABEL_SMOOTHING = 0.1`). Recommendation: run the primary EMA+TTA config first with
`LABEL_SMOOTHING` unchanged at 0.2 (to isolate the boosters), and only if that lands a clean
improvement, optionally try 0.1 as a follow-up. Do **not** bundle PCT_START / peak-LR
changes — those perturb the proven schedule and add risk for little expected upside.

---

## Wall-clock and eval-cadence analysis

This is the main feasibility risk, so it gets explicit treatment.

**Per-step EMA overhead.** `update_parameters` with `multi_avg_fn` uses a foreach
`torch._foreach_lerp_` over ~6.5M params + BN buffers — a handful of fused CUDA kernels,
no host sync beyond what already exists. On an H20 at ~16ms/step this adds well under ~1ms,
i.e. <~5% step overhead, and only fires after 15% warmup. Net effect on epochs-in-budget:
likely ~190 epochs vs EXP-001's 192 — negligible, and the budget is *training-time* gated
(`total_training_time` accumulates measured `dt`), so this cannot overrun 300s; it just
means a few fewer steps, which the completing one-cycle anneal tolerates.

**Eval cost doubling (TTA).** Flip-TTA doubles eval forward passes. EXP-001's total wall was
**447.4s** against a **600s hard cap**, with ~192 single-forward evals over the 10k-image
test set (batch 256, ~40 batches). Eval is `@torch.inference_mode()` and fast; the eval
portion of the 447s is a fraction of the 147s of non-training wall. Doubling eval forward
cost adds at most that eval fraction again. Conservative estimate: total wall rises to
~480–520s, still under the 600s cap. **Risk mitigation if wall is tight:** the cleanest
lever is to *not* eval every epoch with TTA but it must stay ≤1 eval/epoch — so instead,
reduce eval frequency is disallowed-flavored (we keep 1/epoch). The robust mitigation is to
keep TTA on but accept ~190 epochs; if wall approaches the cap in practice, drop TTA and
keep EMA-only (EMA is the larger, cheaper win). Monitor the per-epoch `wall:` print
(EXP-001 already added it) during the run.

**EMA decay choice.** With ~96 steps/epoch and ~190 epochs (~18k steps), a per-step decay of
0.998 has an effective averaging horizon of ~1/(1-0.998)=500 steps (~5 epochs), which tracks
the fast one-cycle anneal without lagging the rapidly-improving low-LR tail. A decay too high
(0.9995, ~2000-step horizon) would lag the final anneal and *hurt*; too low (0.99, ~100-step)
under-smooths. 0.998 is the safe middle; it is the single most important hyperparameter and
the primary thing to sanity-check if results disappoint.

---

## Reasoning with cited pointers

- **EMA/Polyak averaging for SGD.** Averaging the SGD iterate trajectory yields a point that
  generalizes better than the final iterate, especially under cyclic/high LR — the
  foundation of SWA (Izmailov et al., "Averaging Weights Leads to Wider Optima and Better
  Generalization", arXiv:1803.05407) and its EMA variant. PyTorch packages both in
  `torch.optim.swa_utils`.
- **`AveragedModel` mechanics (verified from source, `torch/optim/swa_utils.py` v2.9.0):**
  `self.module = deepcopy(model)`; `forward` returns `self.module(*args, **kwargs)`;
  `.eval()` propagates to `self.module` via standard `nn.Module` recursion; `use_buffers=True`
  averages BN buffers with the same EMA fn; `get_ema_multi_avg_fn(decay)` does
  `foreach_lerp_(ema, current, 1-decay)` i.e. `ema = decay*ema + (1-decay)*current`; first
  `update_parameters` (n_averaged==0) copies params. These four facts are exactly what make
  `evaluator.evaluate(ema_model, device)` work with the frozen interface and BN-consistent.
  (https://docs.pytorch.org/docs/2.9/generated/torch.optim.swa_utils.AveragedModel.html ,
  https://github.com/pytorch/pytorch/blob/v2.9.0/torch/optim/swa_utils.py)
- **BN handling guidance:** PyTorch docs state running BN stats need either `update_bn()` or
  `use_buffers=True`; "updating the statistics in normalization layers increases accuracy."
  We choose inline `use_buffers=True` to avoid spending training-budget time on an
  `update_bn` pass. (https://docs.pytorch.org/docs/2.9/optim.html)
- **Flip TTA.** airbench's `infer_mirror` averages logits over the image and its horizontal
  mirror for a documented eval-time gain; standard practice on CIFAR. "Bag of Tricks for
  Image Classification" (He et al., arXiv:1812.01187) covers TTA and label-smoothing=0.1 as
  standard recipe components.
- **Codebase facts grounding feasibility:** `prepare.py` `Eval.evaluate(self, model, device)`
  does `model.eval()` then `model(inputs)` (lines 33–42) — accepts any module and runs its
  `forward`, so both boosters are reachable without touching the frozen file. EXP-001
  `train.py` already passes a model object to `evaluator.evaluate` (line 234) and tracks
  `best_acc` (lines 236–237); the change is purely *which* model object we pass.
- **EXP-001 self-identified these as next steps** (`04-analysis.md` lines 35, 37, 42),
  marking them medium-high confidence cheap gains on the now-stable base.

---

## Do EMA and TTA compose?

Yes, and orthogonally. EMA changes *which weights/BN-stats* are evaluated (a better point in
weight space); TTA changes *how those fixed weights read each image* (a 2-view ensemble at
inference). They operate on different axes and their gains are empirically near-additive in
fast-CIFAR recipes (airbench stacks EMA-like averaging with mirror TTA). The TTA forward
lives in the inner `ResNet9.forward`, which `AveragedModel.forward` calls via `self.module`,
so EMA-weighted inference automatically goes through the TTA path — no special wiring needed.

---

## Estimated effort

**Low.** ~20 lines changed in one file: refactor `forward` (4 lines), add EMA import + model
(4 lines), per-step `update_parameters` guard (3 lines), swap eval target (2 lines), optional
one-line label-smoothing flip. No architecture change, no schedule change, no new dependency.
One training run within the existing harness. The only "thinking" cost is the EMA decay
choice and watching the wall-clock cap.

---

## Risk assessment (worst case)

- **Most-likely failure / key assumption:** the EMA decay is mistuned for the one-cycle
  tail. If decay is too high, the EMA lags the sharply-improving low-LR endgame (where
  EXP-001 got most of its accuracy) and EMA acc lands *below* the raw 95.22% — a regression.
  Mitigation: (a) decay=0.998 sized to ~5-epoch horizon; (b) the eval `eval_model` swap means
  early epochs still report raw-model acc, and `best_acc` is a `max`, so as long as EMA
  *ever* beats raw at some epoch we capture it; but note **if EMA never beats raw, best_acc
  could in principle stay at the raw value only if raw is also evaluated** — here we evaluate
  EMA (not raw) after warmup, so a badly-lagging EMA *would* cap best_acc below 95.22%. This
  is the real worst case. Cheap insurance: evaluate **both** is disallowed (≤1 eval/epoch),
  so instead pick decay conservatively and, if the run shows EMA tracking below raw in the
  tail, the fallback config is TTA-only (no EMA), which cannot regress weights.
- **Wall-clock cap:** doubled TTA eval pushes total wall toward the 600s kill threshold.
  EXP-001 had 153s of headroom; estimate leaves ~80–120s margin, but it is the second risk.
  Mitigation: monitor `wall:` print; if tight, drop TTA, keep EMA.
- **BN-stat staleness with `use_buffers=True`:** EMA-averaged BN buffers could be slightly
  off vs. the averaged weights. Low risk by end of training (stats converge); fallback is a
  post-loop `update_bn` over ~5 batches (off training-budget).
- **Floor protection:** because neither booster changes optimization, the *raw* model still
  follows EXP-001's exact 95.22% trajectory; the only way to land below baseline is the EMA
  regression above, which the TTA-only fallback removes. Net downside is bounded and the
  expected value is clearly positive.

## Expected accuracy estimate

- EMA on a completing one-cycle: typically **+0.2–0.5pp** (SWA/EMA literature, fast-CIFAR).
- Flip TTA: typically **+0.2–0.4pp** (airbench `infer_mirror`).
- Composed (sub-additive to be conservative): **+0.3–0.7pp** over 95.22%, i.e.
  **~95.5–95.9%** best_test_acc. Even the low end (95.52%) clears the +0.1pp bar (95.32%)
  with margin. Most-likely single-number estimate: **~95.7%**.

## Sources

- [AveragedModel — PyTorch 2.9 docs](https://docs.pytorch.org/docs/2.9/generated/torch.optim.swa_utils.AveragedModel.html)
- [torch/optim/swa_utils.py @ v2.9.0](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/optim/swa_utils.py)
- [torch.optim docs (SWA/EMA + BN handling)](https://docs.pytorch.org/docs/2.9/optim.html)
- SWA: Izmailov et al., arXiv:1803.05407
- Bag of Tricks: He et al., arXiv:1812.01187
