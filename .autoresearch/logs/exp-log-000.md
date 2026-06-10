# EXP-000: Modern training recipe — bf16 AMP + channels_last + budget-matched cosine schedule

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-000.md
- **Plan**: plans/plan-000.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-000
- **Commit**: de14baa (on autoresearch/exp-000, merged to autoresearch/dev)
- **PR**: not created — repo has no `origin` remote (intentional, per TASK.md § Git Rules: keep commits
  local to avoid baseline-improvement data leakage). Merged locally to integration branch instead.
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented all of plan-000's recipe changes in `train.py` only (Milestone 1). Six changes:
(1) added `import math`; (2) replaced the `LR=0.1` constant with `PEAK_LR=0.2` + `WARMUP_FRAC=0.05` +
`LABEL_SMOOTHING=0.1` and a module-level `lr_at_fraction(frac)` helper (linear warmup then cosine
anneal to ~0); (3) raised `MAX_STEPS` 64000 → 10_000_000 so the 300s time budget is the sole limiter;
(4) model `.to(device, memory_format=torch.channels_last)` and per-batch input conversion to
channels_last; (5) SGD now has `lr=PEAK_LR, nesterov=True` and the `MultiStepLR` scheduler + its
`scheduler.step()` were removed, replaced by per-step `pg["lr"] = lr_at_fraction(elapsed/budget)`;
(6) forward+loss wrapped in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` (no GradScaler
needed for bf16) and training loss uses `label_smoothing=0.1`. Syntax + `ruff check` pass.

### Surprises & Discoveries
- The per-step LR is computed from `total_training_time` accumulated *before* the current step, so the
  very first step runs at lr≈0 (frac=0). Harmless — it's the start of warmup.
- The existing logging line `lr = optimizer.param_groups[0]["lr"]` (L217) re-reads the LR I set at the
  top of the loop, so the printed `lr:` field stays accurate with no extra change.

### Decisions
- **Time-driven schedule + raised MAX_STEPS**: driving LR by elapsed-time fraction (not step count) keeps
  the anneal-to-zero correct even though bf16 changes how many steps fit in 300s; raising MAX_STEPS
  prevents an early stop (with LR not yet annealed) if throughput rises enough to hit the old 64000 cap.
- **Held fixed to isolate the recipe**: BATCH_SIZE=128, WEIGHT_DECAY=1e-4, seed=42 unchanged. Batch
  scaling and WD tuning are deliberately deferred to later experiments for clean attribution.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background bash ID bnzma1ibz (local, GPU 0)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 (resumed session)
- **Ended**: 2026-06-08

Description:
- Running the full modern-recipe `train.py` on GPU 0 under the fixed 300s training budget. Tests the
  EXP-000 hypothesis that a budget-matched cosine schedule (replacing the never-annealing MultiStepLR),
  bf16 AMP, channels_last, Nesterov, and label smoothing raise `best_test_acc` ≥ 91.83 (baseline 91.73
  + 0.1 pp). Expecting the schedule fix to drive most of the gain (likely into the ~93% range) with the
  run completing cleanly within budget (total < 600s).

Observations:
- Run started cleanly: `Device: cuda`, ResNet-20 269,722 params, Time budget 300s, 390 batches/epoch
  (source: run.log L1-4).
- Early throughput ~20,000 img/s, dt ~6-7ms/step vs baseline ~8.6ms/step — bf16 AMP + channels_last
  speedup is real; implies more steps/epochs fit in 300s (source: run.log early step lines).
- LR warmup working: lr rising 0.0195 → 0.0282 toward PEAK_LR=0.2; training loss decreasing
  2.04 → 1.72 over first 200 steps, no NaN/divergence (source: run.log early step lines).

Key Metrics:
- best_test_acc: **92.06%** @ epoch 104 (baseline 91.73%, +0.33 pp) (source: run.log summary)
- final_test_acc: 91.92% | final_test_loss: 0.3115 (label-smoothed training inflates eval loss vs
  baseline 0.28; irrelevant to accuracy metric) (source: run.log summary)
- training_seconds: 300.0 | total_seconds: 388.6 | startup_seconds: 1.1 (source: run.log summary)
- peak_vram_mb: 164.4 (baseline 330.1 — bf16 actually lowered VRAM) (source: run.log summary)
- num_epochs: 109 | num_steps: 42,156 (baseline 90 / 34,861 → +21% steps from AMP+channels_last throughput) (source: run.log summary)
- num_params: 269,722 (unchanged) (source: run.log summary)

## Verification Results

### Conditions Checked
- **Condition 1 — clean completion within budget**: PASS. `best_test_acc:` summary present,
  `total_seconds=388.6` < 600, 0 tracebacks/NaN in run.log (source: run.log summary; traceback grep=0).
- **Condition 2 — metric improvement (best_test_acc ≥ 91.83)**: PASS. best_test_acc = **92.06%**
  ≥ 91.83 (baseline 91.73 + 0.1 pp); +0.33 pp over baseline (source: run.log summary).
- **Condition 3 — no constraint violations**: PASS. `git diff --name-only autoresearch/dev` = only
  `train.py`; no diff on `pyproject.toml`/`uv.lock` (no new deps); 109 eval lines = 109 epochs (eval
  once/epoch); seed unchanged (`torch.manual_seed(42)`); no seed hacking.

All necessary conditions PASS → verified improvement.

### Informational Metrics
- peak_vram_mb: 164.4 (vs baseline 330.1 — bf16 reduced memory; huge headroom remains of 98 GB)
- num_epochs / num_steps: 109 / 42,156 (vs baseline 90 / 34,861 — +21% throughput from bf16+channels_last)
- img/s throughput: ~20,000 early-run (vs baseline ~14.9k implied) (source: run.log step lines)

## Errors & Dead Ends

## Human Notes

> (none — autopilot)
