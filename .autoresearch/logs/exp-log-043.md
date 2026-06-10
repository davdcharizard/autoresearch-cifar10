# Experiment Log EXP-043: AdamW optimizer-family swap

## Execution
- **Created**: 2026-06-09
- **Brainstorm**: brainstorm/brainstorm-043.md
- **Plan**: plans/plan-043.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-043
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed (clean run; verdict no-improvement — see Verification Results)

## Implementation Notes

### Summary
Three edits to `train.py` (Milestone 1): `PEAK_LR` 0.2→2e-3, `WEIGHT_DECAY` 1e-4→0.05 (decoupled), and the
optimizer constructor `optim.SGD(...nesterov)` → `optim.AdamW(model.parameters(), lr=PEAK_LR,
betas=(0.9,0.999), eps=1e-8, weight_decay=WEIGHT_DECAY)`. Everything else (model, augmentation, time-fraction
cosine+warmup schedule, label smoothing, batch, channels_last, bf16, torch.compile, seed 42, frozen eval)
unchanged. AST parses, ruff passes, `git diff --stat` = train.py only.

### Surprises & Discoveries
- None at implementation time. `MOMENTUM = 0.9` is now unused but left defined (module-level constant; ruff
  does not flag it) to minimize the diff.

### Decisions
- **lr=2e-3, wd=0.05**: literature-standard AdamW-from-scratch config so a regression is a real
  optimizer-family verdict, not a tuning miss. The per-step `pg["lr"] = lr` already drives AdamW's param
  group, so the cosine+warmup schedule transfers unchanged (just scaled by the new peak).

## Run Log

### Run 1
- **Description**: Intended run of EXP-043 on idle GPU 0 (both H20s 0 MiB/0% at launch). Tests whether AdamW
  (the first optimizer-FAMILY change in 44 experiments — all prior used SGD) beats the 96.32 bar under the
  otherwise-baseline TA+Cutout recipe. Watching for early-step Adam stability (loss not NaN) and dt ~8ms.
- **Job ID / PID**: local background
- **Log file**: run.log
- **WandB**: n/a
- **Status**: completed — CLEAN, FAIR (reported run)
- **Started**: 2026-06-09
- **Observations**: AdamW trained STABLY (no divergence) — smooth convergence (ep1 47%, ep41 92.3, ep71
  95.4, flat 95.7-95.9 tail ep81-91). dt steady 8ms (630×8ms + 70×9ms + 2×12 + 1×29 + 1×13) = baseline →
  91 ep, a fair throughput-neutral test. GPU0 uncontended. params unchanged 4,299,866; peak_vram 470 (≈
  baseline, AdamW's 2 moment buffers are small vs activations). 91 eval lines == 91 epochs (≤1/epoch).
- **Key Metrics**: best_test_acc 95.87% (baseline 96.22, −0.35pp), final_test_acc 95.84, final_test_loss
  0.2023 (vs baseline 0.195), 91 ep, 35226 steps, total 408.4s, exit 0.

## Experimental Adjustments
(none — single clean run, no adjustments needed)

## Errors & Dead Ends
(none — AdamW@2e-3 was stable; no divergence, no retries)

## Verification Results
### Conditions Checked
- **Cond 1 — primary metric ≥ baseline+0.1 (NECESSARY): FAILED.** best_test_acc = **95.87** < bar **96.32**
  (baseline 96.22 + 0.1); also −0.35pp below baseline. Source: run.log `best_test_acc:`. → verdict
  **no-improvement**.
- **Cond 2 — clean completion within budget: PASS.** Summary printed, exit 0, total_seconds 408.4 < 600,
  training_seconds 300.0, num_epochs 91. (Recorded for completeness.)
- **Cond 3 — no constraint violations: PASS.** `git diff --stat` = train.py only; eval 91 == num_epochs 91
  (≤1/epoch); AdamW is in torch.optim (no new deps); seed 42 unchanged; no seed hacking. (For completeness.)

### Informational Metrics
- peak_vram_mb: 470.2 (≈ baseline; AdamW moment buffers small relative to activations).
- num_epochs / num_steps: 91 / 35226 — throughput-neutral (baseline ~91), confirming a FAIR test.
- final_test_loss: 0.2023 (vs baseline 0.195 — slightly worse; AdamW gave neither a top-1 nor a loss win).
- dt distribution: 630×8ms + 70×9ms (steady 8ms = baseline; AdamW adds no measurable per-step cost).
