# EXP-006: EMA weight averaging for evaluation

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-006.md
- **Plan**: plans/plan-006.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-006
- **Commit**: (none — no-improvement, changes discarded)
- **PR**: (none — no-improvement)
- **Outcome**: completed (clean run; verification cond 2 failed → no-improvement verdict in analyze)

## Implementation Notes

### Summary
Implemented the EMA-weight-averaging plan with four edits to `train.py` only (Milestone 1). (1) Imported
`AveragedModel, get_ema_multi_avg_fn` from `torch.optim.swa_utils` (core torch — no new dep). (2) Added
`EMA_DECAY = 0.999` to the hyperparameter block. (3) Built `ema_model = AveragedModel(model,
multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY), use_buffers=True)` right after the model is placed on device
(channels_last) and `num_params` printed. (4) Added `ema_model.update_parameters(model)` immediately after
`optimizer.step()` (inside the timed region so its cost is charged to the 300s budget), and changed the
per-epoch eval call from `evaluator.evaluate(model, device)` to `evaluator.evaluate(ema_model, device)`.
Parse-clean, ruff clean, and the EMA sanity check printed `4299866` (model param count unchanged — EMA is pure
averaging, adds no model capacity).

### Surprises & Discoveries
- None during implementation. Confirmed installed torch is 2.9.1+cu128 and that `AveragedModel` exposes the
  `use_buffers` parameter and `get_ema_multi_avg_fn` exists — both required for this plan and both present.
- `prepare.py`'s `Eval.evaluate(model, device)` calls `model.eval()` then `model(inputs)`; `AveragedModel` is a
  transparent drop-in (delegates forward to `.module`, supports `.eval()`), so no eval-side change is needed and
  the "eval once per epoch" constraint is preserved (still exactly one `evaluate` call per epoch).

### Decisions
- **Decay = 0.999**: chosen so the ~1000-step effective window (~2.5 epochs at ~390 steps/epoch over ~77 epochs)
  tracks the cosine-annealed low-LR tail rather than averaging in stale high-LR weights. Single hyperparameter;
  a sweep is deferred to a follow-up loop if this lands in the noise band.
- **`use_buffers=True`**: average BN running stats with the params so the EMA weights are evaluated with matching
  BN statistics — avoids a separate BN-recompute pass that would consume training budget. Standard practice.
- **EMA update placed inside the timed region** (before `torch.cuda.synchronize()`): honest budget accounting —
  the per-step lerp is a real (tiny) cost and is charged to the 300s budget.

## Experimental Adjustments

(none yet)

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID recorded below)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A (no WandB in this project)
- **Status**: completed
- **Started**: 2026-06-08
- **Ended**: 2026-06-08 (exit 0)

Description:
- Running the k=4 WideResNet + Cutout recipe unchanged, with the single intervention that the per-epoch eval
  uses an EMA copy of the weights (decay 0.999, BN buffers averaged) instead of the raw SGD iterate. Expect a
  clean ~77-epoch run within the 300s budget, `num_params` unchanged at 4,299,866, and `best_test_acc` ideally
  clearing the +0.1pp bar (≥96.10%, vs 96.00% baseline). The test of the hypothesis is whether trajectory
  averaging lands in a flatter, better-generalizing minimum than the final iterate.

Observations:
- Clean startup: `Device: cuda`, `params: 4,299,866` (unchanged vs EXP-003 — EMA is pure averaging, no model
  capacity added) (source: run.log L1-2).
- Loss decreasing normally through warmup, no NaN: 2.04 → 1.40 by step 600 (source: run.log L5-7).
- Throughput healthy/neutral: dt ~11ms/step, ~11,600 img/s — comparable to EXP-003's ~10ms (NOT the ~17ms jitter
  that cost EXP-005 epochs). Confirms the per-step EMA update adds negligible cost (source: run.log L5-7).
- Expected EMA cold-start: epoch-1 EMA eval is low (20.85%) because the average is still init-heavy; `best_acc`
  takes the per-epoch max and will reflect the converged late-training EMA (source: run.log L6).

Key Metrics:
- best_test_acc: **95.97%** @ best over 70 epochs (source: run.log `best_test_acc:` line) — BELOW baseline 96.00
  and the 96.10 bar.
- final_test_acc: 95.80%; final_test_loss: 0.2055 (≈ EXP-003's 0.204 — EMA did NOT lower eval loss).
- num_epochs: 70; num_steps: 27,020; training_seconds: 300.0; total_seconds: 377.7; startup: 1.2s.
- peak_vram_mb: 507.9 (≈ EXP-003 + one model copy, as predicted); num_params: 4,299,866 (unchanged).
- Late-epoch EMA evals very stable: 95.77–95.80 (ep 65–70), best 95.97 — EMA tracked the converged weights but
  settled marginally below the raw iterate (source: run.log `eval ep 65..70`).

## Verification Results

### Conditions Checked

- **Cond 1 — clean completion within budget**: PASS. `best_test_acc:` present (95.97%), `total_seconds` 377.7 <
  600, no traceback in tail (source: run.log).
- **Cond 2 — metric ≥ 96.10 (baseline 96.00 + 0.1)**: **FAIL**. best_test_acc 95.97 < 96.10 (also < 96.00
  baseline). → no-improvement.
- **Cond 3 — no constraint violations**: skipped — aborted after cond 2 failure (per verification protocol).

### Informational Metrics

- Not collected — only gathered when all necessary conditions pass (cond 2 failed). Values recorded above under
  Key Metrics for analysis: num_epochs 70 (throughput ~neutral, within the 65–77 noise band), final_test_loss
  0.2055 (no reduction vs EXP-003), num_params unchanged.

## Errors & Dead Ends

(none)

## Human Notes

> (none — autopilot)
