# Report EXP-006: EMA weight averaging for evaluation
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-006.md
- **Plan**: plans/plan-006.md
- **Log**: logs/exp-log-006.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%) under a fixed 300s training budget on one H20, editing only `train.py`.
Higher is better. Baseline **96.00%** (EXP-003); success bar ≥ **96.10%** (baseline + 0.1pp).

## Idea & Hypothesis
Chosen idea: evaluate an **EMA (exponential moving average) copy of the weights** each epoch instead of the raw
SGD iterate (Polyak averaging / SWA family). Selected as the cleanest *different axis* — orthogonal to the
saturated capacity (k=4) and regularization (Cutout + label smoothing + WD) axes — and throughput-neutral, so it
avoids the epoch-loss trap that sank k=6 (EXP-004) in this launch-bound regime. Hypothesis: trajectory averaging
(decay 0.999, BN buffers averaged) settles into a flatter, better-generalizing minimum → `best_test_acc` clears
96.10 (expected ~96.1–96.4%) at unchanged epoch count and VRAM.

## Approach
Four edits to `train.py` only (no new dependency — `torch.optim.swa_utils` is core torch, verified on torch
2.9.1+cu128): (1) import `AveragedModel`, `get_ema_multi_avg_fn`; (2) add `EMA_DECAY = 0.999`; (3) build
`ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY), use_buffers=True)` after the model
is placed on device (channels_last); (4) call `ema_model.update_parameters(model)` each step right after
`optimizer.step()` (inside the timed region, so its cost is charged to the 300s budget) and switch the per-epoch
eval call to `evaluator.evaluate(ema_model, device)`. `use_buffers=True` averages BN running stats with the
parameters so the EMA weights are evaluated with matching BN statistics (no extra budget-costing BN-recompute).
Everything else held fixed (k=4 WideResNet, Cutout(16), PEAK_LR 0.2, WD 1e-4, label smoothing 0.1, batch 128,
bf16, channels_last, Nesterov, cosine-to-0 schedule, seed 42, eval frozen). Single-variable: raw vs averaged
weights, nothing else.

## Execution
One run, no retries, no errors. Clean startup (`Device: cuda`, params 4,299,866 — unchanged, confirming EMA adds
no model capacity). Loss decreased normally with no NaN. Throughput stayed healthy at dt ~10–11ms/step (≈
EXP-003's ~10ms; NOT the ~17ms jitter that cost EXP-005 epochs), confirming the per-step EMA `lerp_` is
negligible. Run completed in 300.0s training / 377.7s total, 70 epochs / 27,020 steps, peak VRAM 507.9 MB (≈
EXP-003 + one model copy, as predicted). Epoch-1 EMA eval was low (20.85%) — expected cold-start as the average
is still init-heavy — and `best_acc` (per-epoch max) reflected the converged late-training EMA.

## Results
- **Primary metric**: **95.97%** (baseline 96.00%, delta **−0.03 pp**, −0.03%) — below the +0.1 bar AND
  marginally below baseline.
- **Observations**: Late-epoch EMA evals were extremely stable (95.77–95.80 across ep 65–70; best 95.97).
  `final_test_loss` **0.2055** ≈ EXP-003's 0.204 — the EMA did **not** lower eval loss, i.e. it did not find a
  flatter/lower-loss minimum than the raw iterate. 70 epochs is within the established 65–77 noise band, so
  throughput was effectively neutral (cost hypothesis confirmed; accuracy hypothesis not).
- **Analysis**: The mechanism the hypothesis relied on — EMA rescuing accuracy from a noisy SGD endpoint — has
  little to act on here because the **cosine schedule already anneals LR to ~0**, so the final raw iterate is
  itself a well-settled, low-noise point. Averaging the last ~1000 steps (decay 0.999) of already-converged,
  near-identical weights reproduces essentially the same point, landing within noise of (and here a hair below)
  the raw 96.00. EMA's classic gains appear with constant or cyclic/high terminal LR (where the iterate
  oscillates around the minimum); it is largely redundant with a cosine-to-0 schedule. This is a genuine
  no-improvement, consistent with the noise-floor finding (sub-0.2pp deltas are noise at this budget).
- **Key Learning**: EMA/weight-averaging is redundant with the existing cosine-to-0 schedule — the annealed
  endpoint is already effectively "averaged," so EMA at decay 0.999 reproduces it (95.97, no loss reduction)
  rather than improving it.

## Verification
- **Conditions**: Cond 1 (clean completion in budget) PASS; Cond 2 (best_test_acc ≥ 96.10) **FAIL** (95.97);
  Cond 3 (no constraint violations) skipped — aborted after Cond 2 per protocol.
- **Review Notes**: Results confirmed trustworthy — clean single run, frozen eval set, fixed seed 42, exactly one
  eval/epoch, `num_params` unchanged (4,299,866) proving the change was pure weight averaging (no capacity added,
  no eval/seed gaming). The −0.03pp is within the run-to-run noise band, not a real regression.
- **Verdict**: no-improvement
- **Verdict Basis**: valid, trustworthy run; primary metric below the +0.1 bar (verification condition 2 failed).

## Unexplored Avenues
- **EMA paired with a terminal-LR floor (constant tail instead of cosine-to-0)**: EMA helps most when the iterate
  keeps moving near the minimum. Holding a small constant LR for the final ~20% (instead of annealing to 0) would
  give EMA something to average over — but this changes the schedule (a second variable) and the cosine-to-0
  recipe is itself well-validated, so expected net gain is uncertain and likely still noise-scale.
- **SWA with an explicit cyclic/high-constant SWA phase** (Izmailov 2018's actual protocol): collect snapshots at
  a higher cyclic LR, then BN-recompute — a more faithful SWA than EMA-over-annealed-weights. Higher effort
  (schedule surgery + BN-recompute pass costing budget); medium-low priority given EMA's flat result here.
- **Lower decay (e.g. 0.99 / 0.997)**: a shorter window tracks even closer to the (already good) final iterate →
  would converge toward the raw 96.00, not clear the bar. Unlikely to help; not worth a loop.

## Next Steps
1. **Squeeze-and-Excitation blocks on k=4** — *medium confidence*; the other fresh axis from EXP-006 brainstorm
   (architectural efficiency at fixed width). Watch throughput: SE's many small kernels risk cutting epochs in
   this launch-bound regime — verify dt/epoch count holds.
2. **Throughput-up to buy epochs (larger batch 256 + LR scaling)** — *medium confidence*; H20 is memory/launch-
   bound at k=4, so a larger batch may raise utilization and add epochs of the regularized model (EXP-003 showed
   extra epochs help). A genuinely different lever (more training) than the saturated axes.
3. **Peak-LR sweep on k=4 (0.15 / 0.3)** — *low-medium confidence*; least-tuned recipe knob, cheap, but expected
   sub-0.2pp at this already-forgiving schedule.

Avoid: further weight-averaging variants and pure regularization tweaks — both now shown noise-scale.

## Exit Action Results
- None defined for this goal — skipped.
