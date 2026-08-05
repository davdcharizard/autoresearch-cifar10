# EXP-003: Frozen patch-whitening first convolution

## Execution

Overall Status & Info:
- **Created**: 2026-06-28
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-003
- **Commit**: 6e25456 (on autoresearch/maximize-cifar10-test-accuracy-003; merged to integration branch)
- **PR**: N/A — no git remote configured (local-only per TASK.md)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented Milestone 1 additively on the EXP-002 base (DavidNet + EMA + flip-TTA, 95.72%). Four edits to `train.py`: (1) added `compute_whitening_weight(train_set, mean, kernel=3, n_img=2000, n_patches=50000, eps=1e-4)` — reads raw `train_set.data` (HWC uint8), `/255` − `EVAL_MEAN` (eval space), unfolds into 3×3×3=27-dim interior patches (capped at n_img=2000 → ~1.8M patches, subsampled to 50k via a **local** Generator), 27×27 covariance, `torch.linalg.eigh`, eigvecs scaled by `1/√(eig+eps)`, reshaped and concatenated `(W,−W)` → frozen `[54,3,3,3]`; (2) `ResNet9` gains `self.whiten = Conv2d(3,54,3,pad=1,bias=False)` (frozen), `prep` widened to `conv_bn(54,64)`, a `load_whitening()` method, and `_forward_once` prepends `self.whiten(x)`; (3) in `main()`, the whitening is computed + loaded after `model.to(device)` but **before** `t_start_training` (off the 300s budget; printed as `whitening_seconds`), and the SGD optimizer is built over `requires_grad`-filtered params (excludes the frozen conv); EMA `AveragedModel` construction stays after, so its initial copy carries the loaded whitening. Compiles; the smoke test passed (whitening_seconds 0.36s, frozen + optimizer-excluded, pool-input 512×4×4 chain intact, real frozen `Eval.evaluate` finite for EMA and raw, learnable 6,602,496 params).

### Surprises & Discoveries
- `whitening_seconds` is ~0.36s — eigendecomposition of a 27×27 covariance over ~1.8M patches is trivially cheap, confirming the off-budget cost is negligible.
- The 3×3/pad-1 whitening conv keeps the feature map entering `pool` at exactly 512×4×4 (verified via a forward hook in the smoke test), so the MaxPool chain needed no adaptation — the headline spatial-dim risk is retired.

### Decisions
- **Option A (whitening → widened `prep` as the learnable mixer), kernel=3/pad=1, eps=1e-4** as pinned in the plan. Option B (identity-init 1×1 mixer) and kernel=2 deferred — they are separate experiments, not in-run variants.
- **Local `torch.Generator().manual_seed(0)`** for the patch subsample (review #5) so the global training RNG (`manual_seed(42)`) is untouched — clean attribution.
- **n_img capped at 2000** (review #2) to bound off-budget patch materialization (~190 MB) and keep `whitening_seconds` small.

## Experimental Adjustments

<!-- none -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (local, background) — PID recorded at launch
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-28
- **Ended**: 2026-06-28
- **PID**: 1637616 (launcher)

Description:
- Official EXP-003 run: `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'` on GPU 1. Adds a frozen ZCA-whitening front-end to the EXP-002 recipe (EMA+TTA, otherwise byte-identical), to test whether input decorrelation lifts `best_test_acc` from 95.72% within the same 300s training budget. Expected ~95.9–96.1% (central ~95.95%), bar ≥95.82%. Watch epoch-1 loss and per-epoch `wall:` vs 600s; record early-epoch vs tail trajectory for interpretability.

Observations:
- No divergence; ran 174 epochs / 16,802 steps. `whitening_seconds: 0.08` (off-budget startup). (source: run.log head + summary)
- **Whitening sped early convergence (mechanism confirmed)**: ep1 60.19% / ep10 85.45% / ep25 88.84% — markedly ahead of EXP-002's ep1 57.08% / ep10 81.57% / ep25 79.35% (source: run.log eval ep 1/10/25 vs experiments/002 trace in 04-analysis.md).
- Tail crossed the bar late: best reached 95.82% at ep159, peaked **95.87% at ep162** and held; final-epoch (174) test_acc 95.83%. The whitening run hit a higher tail than EXP-002 (95.72%) despite **9 fewer epochs** (174 vs 183) — net conditioning win over the lost update count. (source: run.log eval ep159–174 & summary)
- Wall 452.8s < 600s cap; throughput ~29.3k img/s (slightly below EXP-002's ~30k — the extra 54-ch whitening conv per step), VRAM 1.61 GB. (source: run.log summary + step lines)

Key Metrics:
- best_test_acc: 95.87% @ ep162 (source: run.log "best_test_acc:    95.87%")
- final_test_acc: 95.83% @ ep174 | final_test_loss: 0.3305 (source: run.log summary)
- whitening_seconds: 0.08 | training_seconds: 300.0 | total_seconds: 452.8 (source: run.log summary)
- num_epochs: 174 | num_steps: 16802 | peak_vram_mb: 1614.4 (source: run.log summary)

## Verification Results

### Conditions Checked

1. **Runs clean within wall guard** — PASS. Completed normally (full summary), `total_seconds 452.8` < 600 (not timeout-killed), `best_test_acc:` present. (source: run.log)
2. **Full training budget + whitening off-budget + prepare.py frozen** — PASS. `training_seconds 300.0` ≥ 295; `whitening_seconds 0.08` (small, off the budget, in startup); `git diff --quiet -- prepare.py` and `git diff --quiet <dev> -- prepare.py` both exit 0; `TIME_BUDGET_S = 300` intact. (source: run.log; git)
3. **Improves over baseline by ≥ +0.1pp** — PASS → **improvement**. `best_test_acc 95.87%` ≥ bar 95.82% (baseline 95.72%, +0.15pp). **Genuineness cross-check:** max per-epoch `best:` across the eval trace = 95.87% = summary value (not fabricated; came from `Eval.evaluate`). (source: run.log)
4. **Genuine, in-scope, no reward-hack** — PASS. `git diff --name-only <dev>` lists only `train.py`; global `torch.manual_seed(42)`/`cuda.manual_seed(42)` plus a **local** `torch.Generator().manual_seed(0)` for the patch subsample (no global-RNG perturbation, no seed search); exactly one `evaluator.evaluate(` call (L327), once per epoch; no `train=False`; only `CIFAR10(` is the train split (L194); no `.loader`/`test_set`/`testset` eval-internals reach; whitening conv `requires_grad=False`, excluded from the optimizer (smoke-verified), eigendecomposition off the budget (`whitening_seconds`), patch subset capped at 2000 imgs; per-step `synchronize`/`dt`/`total_training_time` budget loop unchanged from EXP-002; TTA still 2 `_forward_once` passes. (source: train.py; git diff)

**All necessary conditions passed → verdict: improvement.**

### Informational Metrics
- whitening_seconds: 0.08 (source: run.log summary) — off-budget eigendecomposition cost negligible.
- peak_vram_mb: 1614.4 (source: run.log) — essentially unchanged vs EXP-002 (frozen front-end is tiny).
- num_epochs: 174 | training_seconds: 300.0 (source: run.log) — 9 fewer epochs than EXP-002's 183 (whitening per-step overhead), yet higher best_acc.
- total_seconds: 452.8 (source: run.log) — wall vs 600s cap (comparable to EXP-002's 442.7s).
- Early-epoch deltas vs EXP-002 (interpretability): ep1 +3.11pp, ep10 +3.88pp, ep25 +9.49pp — whitening clearly accelerated early convergence as theorized. (source: run.log vs experiments/002/04-analysis.md)

## Errors & Dead Ends

<!-- none yet -->

## Human Notes

> (none — autopilot)
