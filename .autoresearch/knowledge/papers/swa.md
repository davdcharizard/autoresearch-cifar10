# SWA — Stochastic Weight Averaging (Izmailov et al. 2018)

**Paper**: "Averaging Weights Leads to Wider Optima and Better Generalization", UAI 2018, arXiv:1803.05407
**Topic**: Generalization / optimization — trajectory weight averaging
**Why relevant**: directly targets the diagnosed binding constraint on this goal (generalization at fixed k=4 capacity in 300s) at near-zero extra compute. First applied in EXP-019.

## Core idea
SGD with a constant or cyclic learning rate explores the boundary of a wide flat region of the loss
surface, bouncing around the optimum without settling into it. Averaging the weights of multiple points
sampled along this trajectory ("SWA solution") lands near the CENTER of that flat region — a wider, flatter
optimum that generalizes better than any single SGD iterate, despite often having slightly higher *train*
loss. Gains reported across CIFAR-10/100 (VGG16, PreResNet-110/164, WideResNet-28-10): typically ~0.5–1.3pp
test accuracy over a same-budget SGD baseline.

## The critical precondition (this is what EXP-006 lacked)
The averaging phase MUST keep the LR at a non-trivial floor — constant or cyclic — so the iterate keeps
MOVING through the flat region. If the LR has already annealed to ~0 (e.g. cosine-to-0), the tail points
are all ≈ the endpoint, so their average reproduces the endpoint and SWA adds nothing. EXP-006 evaluated an
EMA copy on the cosine-to-0 schedule and got a null (95.97) for exactly this reason. SWA is only meaningful
with a terminal-LR floor.

## Recipe (standard)
1. Train normally (here: cosine decay) for the first ~75% of budget, decaying to a moderate floor LR
   (NOT to 0). The SWA LR is typically a moderate constant (paper uses values like 0.01–0.05 for these nets).
2. For the final ~25%, hold the LR constant at the floor. Once per cycle/epoch, update a running average of
   the model parameters:  `w_swa ← (w_swa · n + w) / (n+1)`.
3. **Recompute BatchNorm statistics** for the averaged weights before evaluation: the averaged params were
   never the live BN-tracking model, so its running_mean/running_var are stale. Do a forward-only pass over
   training data (a partial pass over ~50–100 batches estimates BN stats well enough and is cheap).
4. Evaluate the BN-recomputed averaged model.

## PyTorch implementation (no new dependency)
- `torch.optim.swa_utils.AveragedModel(model)` — wraps the model, `.update_parameters(model)` folds the
  current weights into the running average.
- `torch.optim.swa_utils.update_bn(loader, swa_model, device)` — resets BN stats and recomputes them over
  `loader` (does a FULL pass; for budget control, recompute manually over a truncated batch count instead).
- `torch.optim.swa_utils.SWALR` — an LR scheduler that anneals to the SWA LR; not required if the schedule
  is hand-rolled (this project drives LR by elapsed-time fraction).

## Caveats for this project
- The constant-LR tail forgoes cosine-to-0's final "sharpening"; net effect depends on whether flat-region
  averaging more than compensates. If not, expect a mild regression — fails gracefully.
- Keep BN-recompute cheap (partial pass) so it does not eat epochs at the 300s budget.
- Eval the averaged model in the tail only; the raw model is still evaluated each epoch in the main phase
  (per-epoch eval count unchanged — respects the once-per-epoch eval constraint).
- Params unchanged → throughput-near-neutral (a fair same-budget test, unlike capacity scaling/SAM).

## Empirical result on this project (EXP-019)
Constant-0.05-LR tail (SWA_START_FRAC=0.75) on the EXP-012 recipe, 91 epochs, throughput-neutral. SWA engaged
exactly as predicted: the un-annealed raw iterate sat at ~91.8% (LR floor), and the BN-recomputed average climbed
93.95→**95.97%** over the tail (project-lowest test loss **0.1788** vs 0.195). BUT that was **−0.25pp under the
cosine-to-0 baseline (96.22)** — the flatter/lower-loss optimum did not convert to higher top-1. Confirms the
EXP-006 diagnosis (floor needed) AND that, for a top-1 metric at a short fixed budget, a well-tuned cosine-to-0
schedule can beat SWA. EXP-020 then lowered the floor 0.05→0.02 → **96.13** (+0.16pp over EXP-019, still −0.09 vs
baseline), confirming the floor was too high. **Decisive**: the floor sweep is monotone toward the baseline FROM
BELOW (as SWA_LR→0 the constant tail degenerates into cosine-to-0 itself), so SWA's supremum over the floor IS the
96.22 baseline — it cannot exceed a tuned cosine-to-0 on top-1 here. Weight-averaging axis CLOSED for this metric/budget.
