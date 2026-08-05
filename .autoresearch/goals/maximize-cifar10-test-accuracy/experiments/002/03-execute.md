# EXP-002: Weight EMA + flip-TTA on DavidNet

## Execution

Overall Status & Info:
- **Created**: 2026-06-28
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-002
- **Commit**: c404104 (on autoresearch/maximize-cifar10-test-accuracy-002; merged to integration branch)
- **PR**: N/A — no git remote configured (local-only per TASK.md); no PR created
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented the plan's Milestone 1 additively on top of the EXP-001 DavidNet `train.py` (95.22% base). Five edits, training dynamics untouched: (1) imported `AveragedModel, get_ema_multi_avg_fn` from `torch.optim.swa_utils`; (2) added constants `EMA_DECAY=0.998`, `EMA_WARMUP_FRAC=0.15`, `TTA_START_FRAC=0.8`; (3) refactored `ResNet9.forward` into `_forward_once` + a `forward` that flip-averages logits only when `not self.training and self.tta`, and added `self.tta=False` in `__init__`; (4) built `ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.998), use_buffers=True)` on channels_last + an `ema_started` flag, and call `ema_model.update_parameters(model)` after each `optimizer.step()` once `progress >= 0.15`; (5) at per-epoch eval, evaluate `ema_model` after warmup (else raw `model`), setting the evaluated module's `.tta` flag to `eval_progress >= 0.8`. Compiles; smoke test against the real frozen `Eval.evaluate` passed for both the wrapped EMA and raw models.

### Surprises & Discoveries
- None of note. `AveragedModel.forward` cleanly delegates to `self.module(*args)`, so the TTA-gated `ResNet9.forward` is reachable through the frozen `model(inputs)` eval interface. `ema_model.eval()` (called inside `Eval.evaluate`) propagates `training=False` to `ema_model.module`, so the TTA branch activates correctly during eval.

### Decisions
- **Single eval target per epoch (EMA after warmup, raw before).** The goal's ≤1-validation/epoch constraint forbids evaluating both raw and EMA each epoch, so after warmup only the EMA is scored. Per the plan (§ Code Changes 6), EXP-001's 95.22% is therefore not a guaranteed scored floor — a sub-bar EMA result is a legitimate `no-improvement`, not `invalid`. Chosen to give the EMA the maximum number of epochs to post its best (the cleanest single-eval test of the hypothesis).
- **TTA gated to the final 20% (`progress >= 0.8`).** Per loop-1 idea-review wall-clock concern: doubling every eval risks the 600s cap; the flip gain concentrates in the low-LR tail, so only ~last 20% of epochs pay the 2× eval cost.
- **`use_buffers=True` (EMA-average BN running stats).** Standard EMA-of-BN-stats (timm `ModelEmaV2`), on-budget; no off-budget `update_bn` recompute pass.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (local, background) — PID recorded at launch
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-28
- **Ended**: 2026-06-28
- **PID**: 1604305 (main)

Description:
- Official EXP-002 run: `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'` on GPU 1. Trains the unchanged DavidNet recipe for the 300s training-time budget while maintaining a 0.998 weight EMA (started at 15% progress); evaluates the EMA each epoch with flip-TTA enabled in the final 20%. Expected: ~95.5–95.9% best_test_acc (central ~95.7%) vs the 95.22% baseline, bar ≥95.32%. Watch epoch-1 loss for divergence (would indicate an EMA/forward wiring bug) and the per-epoch `wall:` print vs the 600s cap.

Observations:
- No divergence — epoch-1 test_acc 57.08% / loss 1.29, monotone-ish climb thereafter; ran 183 epochs / 17,673 steps (source: run.log, eval ep 1 & summary).
- **EMA + TTA tail bump clearly visible**: best plateaued ~94.4–95.0% through ep ~90–137, then accelerated as the LR annealed; the flip-TTA gate (progress≥0.8, ~training-time 240s ≈ ep 145) coincides with the jump 95.28% (ep144) → 95.49% (ep145) → 95.55% (ep146) (source: run.log eval ep 144–146).
- Peak best_test_acc 95.72% reached at ep 178 (test_acc 95.72%); final-epoch (183) test_acc 95.70% (source: run.log eval ep 178 & summary).
- Wall 442.7s, well under the 600s `timeout` cap; throughput ~30k img/s, dt ~17ms/step; VRAM 1.61 GB (EMA copy negligible) (source: run.log summary + step lines).

Key Metrics:
- best_test_acc: 95.72% @ ep 178 (source: run.log L "best_test_acc:    95.72%")
- final_test_acc: 95.70% @ ep 183 (source: run.log summary)
- final_test_loss: 0.3367 (source: run.log summary)
- training_seconds: 300.0 (source: run.log summary)
- total_seconds: 442.7 (source: run.log summary)
- peak_vram_mb: 1614.9 (source: run.log summary)
- num_epochs: 183 | num_steps: 17673 (source: run.log summary)

## Verification Results

### Conditions Checked

1. **Runs clean within the wall guard** — PASS. Process completed normally (full summary printed), `total_seconds: 442.7` < 600 (not a `timeout` kill), `best_test_acc:` present. (source: run.log summary)
2. **Full training budget + prepare.py frozen** — PASS. `training_seconds: 300.0` ≥ 295; `git diff --quiet -- prepare.py` exit 0 (unchanged); `grep -q "TIME_BUDGET_S = 300" prepare.py` matched. (source: run.log summary; git)
3. **Improves over baseline by ≥ +0.1pp** — PASS → **improvement**. `best_test_acc = 95.72%` ≥ bar 95.32% (baseline 95.22%, +0.50pp). (source: run.log "best_test_acc:    95.72%")
4. **Genuine, in-scope, no reward-hack** — PASS. `git diff --name-only autoresearch/...-dev` lists only `train.py`; single fixed `torch.manual_seed(42)`/`cuda.manual_seed(42)`, no seed search; exactly one `evaluator.evaluate(` call site (L272), invoked once per epoch outside the step loop; no `train=False` (spaced grep empty); the only `CIFAR10(` is the **train** split (L149 `train_set`, `train=True`); no `.loader`/`test_set`/`testset` reaches into eval internals; eval-time `forward` does exactly two `_forward_once` passes (image + single `x.flip(-1)`), no hidden multi-view loop. (source: train.py; git diff)

**All necessary conditions passed → verdict: improvement.**

### Informational Metrics
- peak_vram_mb: 1614.9 (source: run.log summary) — EMA copy overhead negligible vs EXP-001's ~1.6 GB.
- num_epochs: 183 | training_seconds: 300.0 (source: run.log summary) — vs EXP-001's 192 epochs; the ~9-epoch reduction is the per-step EMA `update_parameters` + tail TTA overhead, well within budget.
- total_seconds: 442.7 (source: run.log summary) — wall vs 600s cap; comparable to EXP-001's 447.4s (tail-gated TTA kept eval overhead small).
- final_test_acc: 95.70% @ ep 183 | final_test_loss: 0.3367 (source: run.log summary).

## Errors & Dead Ends

<!-- none yet -->

## Human Notes

> (none — autopilot)
