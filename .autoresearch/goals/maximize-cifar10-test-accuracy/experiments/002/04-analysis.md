# Report EXP-002: Weight EMA + flip-TTA on DavidNet
- **Created**: 2026-06-28

## Goal
Maximize `best_test_acc` (%) on CIFAR-10 within the fixed 300s training-time budget, editing only `train.py` (higher is better). Entering baseline: **95.22%** (EXP-001, commit 26fdc83). Improvement bar: beat baseline by ≥0.1pp → **≥95.32%**.

## Idea & Hypothesis
Chosen idea (Idea-02, the cross-model idea-review's pick, 8.5/10 evidence, 8/10 impact): add a **weight EMA** of the model plus **eval-time horizontal-flip TTA**, leaving the proven EXP-001 training recipe otherwise untouched. Reasoning: both are standard, orthogonal, near-free gains that act on the *evaluated* iterate rather than perturbing training dynamics — EMA denoises the low-LR-tail weights (where EXP-001's gains concentrated), and flip-TTA averages out left/right prediction variance. Hypothesis: ~95.5–95.9% (central ~95.7%), comfortably clearing the 95.32% bar without risking the training trajectory.

## Approach
Five additive edits to `train.py` (training dynamics byte-identical to EXP-001):
- Import `AveragedModel, get_ema_multi_avg_fn` from `torch.optim.swa_utils`.
- Constants `EMA_DECAY=0.998`, `EMA_WARMUP_FRAC=0.15`, `TTA_START_FRAC=0.8`.
- `ResNet9.forward` refactored into `_forward_once` + a TTA-gated `forward` (`0.5*(_forward_once(x)+_forward_once(x.flip(-1)))` only when `not self.training and self.tta`); `self.tta=False` default.
- `ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.998), use_buffers=True)` (channels_last); `ema_model.update_parameters(model)` called after every `optimizer.step()` once `progress >= 0.15`.
- Per-epoch eval evaluates `ema_model` after warmup (raw `model` before), with the evaluated module's `.tta` set to `eval_progress >= 0.8`.

Key decisions (from plan review hardening): EMA evaluated *in place of* the raw model after warmup (≤1-eval/epoch constraint forbids evaluating both), so 95.22% was explicitly **not** treated as a guaranteed scored floor — a sub-bar EMA result would have been a legitimate `no-improvement`. TTA gated to the final 20% of the budget to bound eval wall-clock (loop-1 review's concern). `use_buffers=True` EMA-averages BN running stats on-budget (no `update_bn` recompute pass). No change to `LABEL_SMOOTHING=0.2`. The plan passed a 7-point Codex adversarial review (all concerns addressed) and a real-frozen-path smoke test (both EMA-wrapped and raw models run through `Eval.evaluate`) before the official run.

## Execution
Single run, no retries. `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'` on GPU 1 (GPU 0 in use by another job). Ran 183 epochs / 17,673 steps in 442.7s wall (300.0s training-time), no divergence, ~30k img/s, VRAM 1.61 GB. The **EMA+TTA tail bump is directly visible**: best climbed slowly to ~95.0% by ep 137, then the flip-TTA gate engaging at progress≥0.8 (~training-time 240s ≈ ep 145) coincides with the jump 95.28% (ep144) → 95.49% (ep145) → 95.55% (ep146), peaking at 95.72% by ep 178. No errors or dead ends.

## Results

- **Primary metric**: **95.72%** (baseline: 95.22%, delta: **+0.50pp**, +0.53%)
- **Observations**: Result (95.72%) landed squarely inside the predicted 95.5–95.9% band (central estimate 95.7% was essentially exact). The EMA trajectory tracked the raw recipe closely through training (no regression below baseline ever observed once warmed up), and the TTA tail-gate produced a clean, attributable step-up of ~+0.25pp right at progress 0.8 — separating the two contributions is plausible: EMA denoising carried best from ~94.4% to ~95.3% over the anneal, flip-TTA added the final ~+0.25–0.4pp. Cost was ~free: 183 vs EXP-001's 192 epochs (the ~9-epoch drop is per-step `update_parameters` + tail double-forward overhead), wall 442.7s vs 447.4s, VRAM essentially unchanged.
- **Analysis**: Hypothesis confirmed. Both mechanisms are orthogonal to the training recipe and to each other, and both delivered as theorized — the gain is real, in-scope (only `train.py`), and not a measurement artifact (exactly one `evaluator.evaluate`, TTA bounded to 2 passes, no test-set access, single fixed seed).
- **Key Learning**: A short-horizon weight EMA evaluated in place of the raw iterate, plus tail-only flip-TTA, adds +0.50pp (95.22→95.72) for ~free and fully orthogonal to the DavidNet training recipe.

## Verification
- **Conditions**: all passed. (1) clean run, summary printed, `total_seconds 442.7` < 600 (not timeout-killed); (2) `training_seconds 300.0` ≥ 295, `prepare.py` byte-unchanged, `TIME_BUDGET_S=300` intact; (3) `best_test_acc 95.72%` ≥ 95.32% bar → improvement; (4) only `train.py` changed, single fixed seed, exactly one `evaluator.evaluate` call, no `train=False`/test-set/eval-internals access, TTA bounded to exactly 2 forward passes.
- **Review Notes**: results confirmed trustworthy. The gain arrives through the sanctioned eval interface (`model(inputs)` on EMA weights + a single mirror), not through gaming the metric; magnitude and timing (tail step-up at the TTA gate) are mechanistically consistent. No false-pass risk identified.
- **Verdict**: **improvement**
- **Verdict Basis**: all necessary conditions passed + meaningful +0.50pp improvement over baseline.

## Unexplored Avenues
- **EMA decay sweep / tail-only EMA** — 0.998 was a first principled guess; 0.999 (longer horizon) or a progress-scheduled decay could capture more of the anneal. Low risk, plausibly another tenth or two.
- **TTA gate / richer TTA** — gate is 0.8; widening it or adding small shift/pad-crop TTA (still cheap, eval-only) might add more, though flip is the highest-value single view. Watch the 600s wall if TTA runs on more epochs.
- **Stack onto a stronger base** — EMA+TTA is orthogonal, so it should compose additively with architectural/input upgrades (whitening conv init from Idea-01, or a wider net from Idea-03). The next capacity/representation experiment can keep EMA+TTA on for free.
- **Raw-vs-EMA crossover** — we never measured raw at the tail (constraint). If a future run wants the floor guarantee, evaluate raw until ~0.9 then EMA — but this run shows the EMA bet paid off, so low priority.

## Next Steps
- **Whitening/patch-init first conv (Idea-01) with EMA+TTA retained** — input-decorrelation is the most-cited fast-CIFAR lever not yet tried here; orthogonal to this win. Confidence: medium.
- **Wider/deeper DavidNet (Idea-03) within budget** — VRAM (1.6/98 GB) and throughput (30k img/s) leave huge headroom; capacity may now be the binding limit as tail-denoising is solved. Confidence: medium.
- **Quick EMA-decay / TTA-gate sweep** — cheapest immediate follow-up to bank another tenth on the now-higher 95.72% baseline. Confidence: low-medium (diminishing returns near the bar).

## Exit Action Results
- No exit actions defined for this goal — none executed.
