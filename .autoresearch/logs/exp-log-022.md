# Experiment Log EXP-022: Batch 1024 with √-scaled peak LR (0.566)

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-022.md
- **Plan**: plans/plan-022.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-022 (cut from autoresearch/dev @ 1990397)
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: failed (verification condition 1 not met — converged no-improvement, 96.57 vs 96.81 bar)

## Implementation Notes

### Summary
Two-constant edit to `train.py` per plan-022 Milestone 1: `BATCH_SIZE 512→1024` and `PEAK_LR 0.4→0.566` (= 0.4×√2, √-scaling per Hoffer et al. 1705.08741), with the L23 provenance comment updated from the linear-scaling note to the sqrt-scaling rationale. Everything else byte-identical to the EXP-006 baseline: default-mode torch.compile, foreach/nesterov SGD, time-keyed one-cycle (warmup 0.15), selective WD 5e-4, LS 0.1, TA+RE augmentation. Syntax check passed; `git diff --stat` confirms 1 file / 2 lines. The compile-warmup tensors are sized by `BATCH_SIZE` so warmup automatically matches the new batch shape; `lr_at()` reads `PEAK_LR` directly — no other code paths touch either constant.

### Surprises & Discoveries
- None at implementation time — the change surface is exactly the two module-level constants, as the plan predicted.

### Decisions
- PEAK_LR written as 0.566 (3 decimals) rather than `0.4 * 2**0.5` so the constant block stays literal like its neighbors; the comment carries the derivation.
- Contention thresholds for this run are scaled to the batch-1024 regime (60ms windowed instead of 30ms) because clean windowed dt at 1024 is ~41ms (EXP-012) — the baseline 30ms gate would self-kill a healthy run. Same 1.5×-clean ratio as the standard gate.

## Run Log

### Run 1
- **Description**: Full 300s-budget training run of the two-constant variant on GPU 0. Reproduces EXP-012's measured throughput configuration (batch 1024, ~41ms windowed dt, ~151 epochs) while replacing the linearly-scaled peak 0.8 — diagnosed in exp-report-012 as the trajectory-damaging defect (~18pp mid-run deficit, bouncy hot phase) — with √-scaled peak 0.566. Expected outcome per hypothesis: mid-run trajectory within ~1pp of baseline family, 148–153 epochs, VRAM ~2.6GB, best_test_acc ≥ 96.81 if the +12 epochs convert under unchanged numerics.
- **Job ID**: local background composite (pre-check + launch + inline watchdog)
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed, rc=0, clean signatures
- **Started**: 2026-06-10T12:12:30Z
- **Ended**: 2026-06-10T12:21:45Z
- **Observations**: Pristine run — GPU-0 pre-check clean; watchdog windows all 40.0–42.0ms (zero >60ms); post-hoc profile authoritative: 0/143 windows >60ms, mean win 41.5ms, expected 149.2 epochs vs 151 actual (within ±3). Throughput delivered exactly per EXP-012 (151 epochs, dt ~41.5ms windowed, startup 12.2s warm). Trajectory: smoother than EXP-012's linear-0.8 run (no ~18pp collapse), but still clearly below the baseline family mid-run (ep90 89.96, ep105 92.93); tail converged into a long flat plateau — final 7 evals span 96.50–96.57, final=96.52, best=96.57 at ep150. The plateau is genuinely converged (best within 0.05 of final), so the deficit is dynamics, not starvation.
- **Key Metrics**: best_test_acc 96.57 | final 96.52 | final_test_loss 0.1870 | training_seconds 300.0 | total 533.6s | startup 12.2s | VRAM 3134.6MB | 151 epochs | 7232 steps | 4,286,026 params | eval_lines 151 = num_epochs (≤1/epoch confirmed). Source: run.log summary block + task bjm5virkm output.

## Experimental Adjustments

## Errors & Dead Ends

## Verification Results

Pre-condition (contention profile, plan-022 §Verification): PASS — `windows>60ms: 0 of 143 | mean win 41.5 ms | expected epochs 149.2` vs 151 actual (±3 tolerance). Run is clean and analyzable.

### Conditions Checked

1. **best_test_acc ≥ 96.81 (baseline 96.71 + 0.1pp)** — FAILED. Actual: 96.57 (`grep "^best_test_acc:" run.log`). Deficit −0.14pp vs baseline, −0.24pp vs bar, from a fully converged plateau (final 7 evals 96.50–96.57). First-failure-stop: remaining conditions skipped.
2. **Run completes without crash ≤600s** — skipped (aborted after prior failure). For the record: rc=0, total_seconds 533.6 — would have passed.
3. **Validation ≤ once per epoch** — skipped (aborted after prior failure). For the record: 151 eval lines = 151 epochs — would have passed.

### Informational Metrics

## Human Notes
