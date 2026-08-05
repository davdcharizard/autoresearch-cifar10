# Proposal idea-01: Modernized bag-of-tricks recipe on ResNet-20

## Summary

Keep the ResNet-20 architecture essentially unchanged and replace the training **recipe**.
The single biggest lever, per the diagnosis, is the learning-rate schedule: the baseline uses
`MultiStepLR(milestones=[32000, 48000])` calibrated to a nominal 64,000 steps, but only ~37,400
steps complete in the 300 s budget. The model therefore **never reaches its second LR drop (48k)** and
is under-annealed at the moment `best_test_acc` is read. We fix this by sizing a **one-cycle (or cosine)
schedule to the *achievable* step budget**, so LR decays smoothly to ~0 right as the clock runs out —
guaranteeing a fully-annealed model. On top of that we stack low-risk generalization tricks (label
smoothing, zero-init residual BN, Nesterov, no-bias-decay) and cheap throughput tricks (bf16 autocast +
channels_last) that buy *more* steps inside the same 300 s, all consistent with the frozen eval harness.

This is the high-floor / low-ceiling candidate: the schedule fix alone should clear the +0.1 pp bar with
high confidence, and the remaining tricks are additive and individually well-evidenced.

## What this targets (tie to the diagnosis)

Two named limiters from the diagnosis, addressed directly:

1. **Under-annealing.** `train.py` lines 145-147 set `MultiStepLR(milestones=[32000,48000], gamma=0.1)`
   and `MAX_STEPS = 64000` (line 24). The loop (lines 162, 202) exits on the 300 s wall-clock budget
   long before step 48000. So the LR is still 0.01 (after only the first drop at 32k) when training stops;
   the final 0.001 annealing phase — which is where ResNet does most of its accuracy refinement — never
   happens. A schedule whose **endpoint is the real step count** removes this waste entirely.
2. **Throughput.** `total_training_time` accumulates per-step `dt` *after* `torch.cuda.synchronize()`
   (lines 178-181) and the loop budget is on that training-only clock; startup/compile/eval are excluded.
   Therefore any per-step speedup converts 1:1 into *more optimization steps within budget* → a longer
   effective schedule and more epochs. bf16 + channels_last are free accuracy via more steps.

## Exactly what changes in `train.py`

All edits are in `train.py` only. `prepare.py` (eval harness, 300 s `TIME_BUDGET_S`, eval normalization)
is untouched.

### 1. Learning-rate schedule sized to achievable steps (primary change)

- Compute a **step budget estimate** instead of trusting `MAX_STEPS=64000`. Two robust options; prefer (b):
  - (a) Hardcode `TOTAL_STEPS ≈ 37000` from the diagnosis (baseline reached ~37,431).
  - (b) **Calibrate at runtime**: run a short warmup of N≈100 timed steps, measure median `dt`,
    estimate `TOTAL_STEPS = int(0.97 * TIME_BUDGET_S / median_dt)` (0.97 leaves margin so we don't
    overshoot the schedule end), then build the scheduler with that length. This auto-adapts to the
    speedups from bf16/channels_last (which change `dt`) and to hardware variance. The N warmup steps
    are *real* training steps (not wasted) — just exclude them from the scheduler-length count or fold
    them in. Recommended: do a no-LR-stepping calibration pass folded into total.
- Replace `MultiStepLR` with **`torch.optim.lr_scheduler.OneCycleLR`** (already in torch 2.9):
  `OneCycleLR(optimizer, max_lr=MAX_LR, total_steps=TOTAL_STEPS, pct_start=0.15,
  anneal_strategy='cos', div_factor=10, final_div_factor=100)`.
  This gives a short warmup (0→max_lr over the first 15% of steps) then cosine decay to ~max_lr/1000.
  Keep `scheduler.step()` per batch (line 176) — OneCycleLR is designed for per-step stepping.
- Guard against `step >= TOTAL_STEPS` overrunning the schedule: stop calling `scheduler.step()` once the
  schedule is exhausted (OneCycleLR raises if stepped past `total_steps`), or clamp. The wall-clock budget
  is the real terminator; the schedule should finish slightly *before* the clock (hence the 0.97 margin).
- Alternative if OneCycle proves finicky: `CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)` with a manual
  linear warmup for the first ~400 steps. Cosine-to-zero is the trick from *Bag of Tricks* (He 2018, §4.1).

### 2. Optimizer tweaks

- `optim.SGD(..., momentum=0.9, nesterov=True, weight_decay=WEIGHT_DECAY)` — add `nesterov=True` (line 142-144).
- **No-bias / no-BN decay** (Bag of Tricks §4.2): split params into two groups — apply `weight_decay` only
  to conv/linear *weights* (ndim ≥ 2), and `weight_decay=0` to all biases and BN γ/β (ndim ≤ 1). Build via
  a loop over `model.named_parameters()` keyed on `param.ndim` / name. Cheap, standard, small positive effect.
- `max_lr`: with batch 128, a peak of **0.4–0.5** is typical for one-cycle CIFAR ResNets (cifar10-fast uses
  high peak LR with a triangular/one-cycle profile). Start at `max_lr=0.4`. If unstable, fall back to a
  cosine schedule with base LR 0.1 (the current value), which is the conservative, near-certain-to-work path.
- Keep `weight_decay=5e-4` candidate vs current `1e-4`: cifar10-fast and most fast-CIFAR recipes use 5e-4.
  Recommend `5e-4` paired with no-bias-decay; this is a known good pairing for short schedules.

### 3. Architecture micro-tweak: zero-init last BN gamma

- In `BasicBlock` (lines 33-57), the residual is `out = bn2(conv2(...))` then `out += shortcut`. Add, after
  model construction, a pass that sets `block.bn2.weight` (γ) to 0 for every `BasicBlock`. This makes each
  block initially an identity map (Bag of Tricks §4.1, "zero-γ"), easing early optimization. Implement as a
  small loop over modules in `ResNet.__init__` after `self.apply(self._weights_init)`. Pure-init change, zero
  runtime cost, low risk.

### 4. Regularization: label smoothing + RandomErasing

- Replace `F.cross_entropy(outputs, targets)` (line 173) with
  `F.cross_entropy(outputs, targets, label_smoothing=0.1)` (supported natively in torch 2.9). The eval harness
  in `prepare.py` uses plain `F.cross_entropy` for *reporting* loss only; accuracy (the metric) is unaffected
  by our training loss change. Label smoothing 0.1 is the Bag-of-Tricks default and improves generalization.
- Add **`transforms.RandomErasing(p=0.5)`** to the *train* transform (after `ToTensor`/`Normalize`, lines
  117-124) as a Cutout substitute (Cutout itself is not in torchvision, but RandomErasing is the canonical
  drop-in and is already available — no new dependency). cifar10-fast credits Cutout as a core ingredient.
  Use a modest patch (`scale=(0.02, 0.2)`) to avoid over-regularizing a short schedule. **Risk note:** heavy
  augmentation can *hurt* very short schedules by slowing convergence; if epochs are few, keep it light or
  gate it behind an ablation. This is the most droppable trick.

### 5. Throughput: bf16 autocast + channels_last (free extra steps)

- Wrap the forward+loss in `with torch.autocast('cuda', dtype=torch.bfloat16):` (around lines 172-173).
  bf16 needs **no GradScaler** (unlike fp16), so the loop stays simple and numerically safe. H20 has strong
  bf16 throughput. This shortens `dt`, which — because the schedule length is computed from measured `dt` —
  automatically lengthens the effective training in steps.
- Convert model + inputs to `channels_last`: `model = model.to(device, memory_format=torch.channels_last)`
  and `inputs = inputs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)`.
  Conv kernels are faster in NHWC. Negligible risk; standard.
- **torch.compile: do NOT use for the primary run.** Its compile cost lands inside the *first* timed step
  (the `dt` accumulation at lines 167-181 includes the first forward), which would corrupt the throughput
  calibration and could blow the 10-minute wall-clock guard during graph capture. The budget excludes startup
  but `model(inputs)` compilation happens *inside* the timed loop here. Treat compile as a separate follow-up
  experiment, not part of this low-risk recipe.

### Normalization consistency (constraint compliance)

The train transform keeps `mean=(0.4914, 0.4822, 0.4465), std=(1, 1, 1)` exactly as in `prepare.py`
`Eval.__init__` (lines 13-20). We do **not** change normalization. RandomErasing operates on already-normalized
tensors, which is fine since erased values default to 0 (≈ the per-channel mean after this normalization,
since std=1 means pixels are roughly mean-centered) — acceptable and standard. No eval-harness coupling is broken.

## Reasoning grounded in the literature

- **Schedule = the dominant lever.** *Bag of Tricks for Image Classification* (He et al., CVPR 2019,
  arXiv:1812.01187, §4.1) shows cosine-decay-to-zero plus warmup as a core refinement; the whole point is that
  the LR must actually finish annealing. Our baseline's milestone schedule provably does not (it stops at the
  0.01 plateau). Re-targeting the schedule endpoint to the achievable ~37k steps is the textbook fix.
- **One-cycle for fast CIFAR.** David Page's *How to Train Your ResNet* / cifar10-fast (DAWNBench winner,
  ~94% in tens of seconds) reaches high accuracy in *very few epochs* precisely via a one-cycle LR (linear
  warmup to a high peak, then decay) + Cutout. Our 300 s / ~96-epoch budget is far more generous than
  cifar10-fast's, so a one-cycle/cosine schedule sized to our budget is squarely in its regime.
- **Stacked tricks are additive.** Bag of Tricks reports ResNet-50 ImageNet 75.3% → 79.29% by stacking
  zero-γ BN, no-bias-decay, label smoothing (ε=0.1), and cosine schedule (the search result above and the
  paper's Table 5/6). On CIFAR these same tricks each contribute ~0.2–0.8 pp; even a fraction of that, summed,
  comfortably exceeds +0.1 pp.
- **bf16/channels_last = more steps, not just speed.** Because `train.py` ties schedule length and budget to
  measured per-step time, throughput gains translate into a longer effective schedule (more annealing
  resolution) and more epochs of augmented data — a second-order accuracy benefit beyond raw speed.

Sources:
- [Bag of Tricks for Image Classification (He et al., CVPR 2019, arXiv:1812.01187)](https://ar5iv.labs.arxiv.org/html/1812.01187)
- [CVPR 2019 open-access PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)
- David Page, *How to Train Your ResNet* (cifar10-fast / myrtle.ai series) — one-cycle LR + Cutout as the core recipe.
- OneCycle policy: Smith & Topin, *Super-Convergence* (arXiv:1708.07198), basis for `torch.optim.lr_scheduler.OneCycleLR`.

## Estimated effort

**Low–medium.** All changes are localized to `train.py` and use only existing torch/torchvision APIs
(`OneCycleLR`, `torch.autocast`, `channels_last`, `RandomErasing`, `label_smoothing`, param-group split).
No new files, no architecture surgery. The one piece needing care is the runtime step-budget calibration and
the OneCycle `total_steps` guard (overrun raises). Budget ~1 implementation pass + 1 run.

## Risk assessment

- **Most-needs-to-hold assumption:** the schedule endpoint must align with the *actual* stopping step. If
  `TOTAL_STEPS` is overestimated, the model stops *before* fully annealing (partially repeating the baseline's
  failure, though far less severe). If underestimated, training hits the schedule end early and runs the last
  steps at near-zero LR (harmless, mild waste). Mitigation: the 0.97 margin + runtime calibration, and the
  asymmetry favors slight *under*-estimation (anneal a touch early rather than not at all).
- **OneCycle instability at high peak LR (0.4–0.5).** A too-high peak can diverge early. Mitigation: start at
  0.4, and the documented fallback is `CosineAnnealingLR` with base LR 0.1 + short warmup — that conservative
  variant alone (correctly sized schedule, the diagnosis's core fix) is very likely to clear +0.1 pp.
- **Over-regularization on a short schedule.** Label smoothing + RandomErasing + 5e-4 wd together could slow
  convergence enough to hurt within ~37–96 epochs. Mitigation: RandomErasing is the designated droppable trick;
  keep its area small, and ablate if the run underperforms.
- **bf16 numerics.** Very low risk for ResNet-20 (small, BN-stabilized); bf16 has fp32 dynamic range so no
  GradScaler needed. channels_last is purely a memory-format change. Worst case these are no-ops on accuracy.
- **Worst case overall:** the throughput/augmentation tricks wash out and only the schedule fix matters — that
  still yields a real, attributable gain. The genuine downside risk is a too-aggressive `max_lr` causing a bad
  run; this is fully recoverable by reverting to the cosine+0.1 fallback, which is near-certain to beat baseline.

## Expected accuracy estimate

- **Conservative (schedule fix only, cosine+warmup, base LR 0.1, no other tricks):** ~92.3–92.8% — the
  baseline's lost final annealing phase typically buys ResNet ~0.7–1.2 pp on CIFAR-10.
- **Full recipe (one-cycle peak 0.4 + zero-γ + label smoothing + no-bias-decay + bf16 more steps + light
  RandomErasing):** ~93.0–93.8%. Each trick contributes ~0.1–0.5 pp on CIFAR; stacked on a ResNet-20 with a
  correctly annealed, slightly longer (more-steps) schedule, low-93s is a reasonable central estimate.
- Both ranges clear the +0.1 pp bar over 91.57% with margin; the conservative path is the safety net that makes
  this the high-floor candidate.
