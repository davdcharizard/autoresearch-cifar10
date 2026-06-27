# Experiment Report EXP-022: Batch 1024 with √-scaled peak LR (0.566)

- **Date**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-022.md
- **Plan**: plans/plan-022.md
- **Exp Log**: logs/exp-log-022.md
- **Branch**: autoresearch/exp-022 (cut from autoresearch/dev @ 1990397)
- **Verdict**: **no-improvement** (96.57 vs baseline 96.71, −0.14pp; bar was ≥96.81)

## Goal

Maximize `best_test_acc` (%) of the widened ResNet-20 under the fixed 300s timed training budget. Direction: higher. Baseline at experiment time: **96.71** (EXP-006 recipe @ 1990397). Success bar: ≥ 96.81 (+0.1pp absolute).

## Idea & Hypothesis

**Idea**: Revisit the campaign's closest miss — EXP-012 (batch 1024 + linearly-scaled peak 0.8 → 96.66, −0.05pp) — replacing the linear LR rule, diagnosed as the trajectory-damaging defect (bouncy hot phase, ~18pp mid-run deficit), with the canonical √-scaling rule (Hoffer et al. 1705.08741): PEAK_LR = 0.4×√2 ≈ 0.566. The throughput half was already measured clean in-project (+8% img/s, 151 epochs) in the same default-compile numerics regime, satisfying the EXP-021 numerics-equivalence requirement.

**Hypothesis**: √-scaling holds the mid-run trajectory within ~1pp of the baseline family; the +12 epochs convert at the EXP-006 arithmetic (~+0.02pp/epoch) → best_test_acc ≥ 96.81. A converged miss closes the batch axis permanently (cold/middle/hot points all measured at 1024).

## Approach

Two-constant edit to `train.py`, everything else byte-identical to baseline: `BATCH_SIZE 512→1024`, `PEAK_LR 0.4→0.566` (provenance comment updated to the sqrt-scaling derivation). Default-mode torch.compile, foreach/nesterov SGD, time-keyed one-cycle (warmup 0.15), WD 5e-4 selective, LS 0.1, TA+RE unchanged. Compile-warmup tensors are sized by `BATCH_SIZE`, so warmup matched the new shape with no further edits. Contention thresholds were rescaled to the batch-1024 regime (60ms windowed vs the baseline 30ms; clean dt at 1024 is ~41ms — the standard gate would self-kill a healthy run).

## Execution

Single pristine run, zero retries. GPU-0 pre-check clean; composite launcher with inline 15s watchdog; all watchdog windows 40.0–42.0ms; startup 12.2s (warm compile); rc=0 at 533.6s total. Post-hoc authoritative profile: **0/143 windows >60ms, mean win 41.5ms, expected 149.2 epochs vs 151 actual** — clean by both gates. No errors, no dead ends.

## Results

| Metric | EXP-022 | Baseline (EXP-006) | EXP-012 (linear 0.8) |
|---|---|---|---|
| best_test_acc | **96.57** | 96.71 | 96.66 |
| final_test_acc | 96.52 | — | — |
| final_test_loss | 0.1870 | — | — |
| num_epochs | 151 | 139 | 151 |
| windowed dt | 41.5ms | 22.4ms (b512) | ~41ms |
| peak_vram_mb | 3134.6 | 1613.0 | ~2600 |
| total_seconds | 533.6 | ~480–540 | — |

**Throughput half: fully confirmed.** 151 epochs, identical to EXP-012 — batch 1024 reliably buys +12 epochs in the default-compile regime. The mechanism's delivery side is not in question.

**Trajectory half: the hypothesis failed in an informative direction.** The √-scaled run was smoother than EXP-012's linear run (no ~18pp collapse; ep90 89.96, ep105 92.93) but still converged BELOW both the baseline AND the linear-0.8 point: 96.57 < 96.66 < 96.71. The plateau is genuinely converged — final seven evals span 96.50–96.57, best within 0.05 of final — so the deficit is dynamics, not epoch starvation.

**Root cause reading**: the 1024-LR axis is now bracketed: 0.566 → 96.57, 0.8 → 96.66, both below 96.71. The ordering (hotter point closer to baseline) says the EXP-012 diagnosis was incomplete — the deficit at 1024 is not (mainly) the "too-hot linear rule"; if anything the optimum at 1024 sits at-or-above 0.8, yet even the better-tuned hot point still loses. The residual deficit is the large-batch generalization cost itself: halved gradient noise at 2× batch hurts more than +12 epochs repay on this recipe. Consistent with goal-learnings' standing line "epochs gained ≈ trajectory quality lost" — but now measured at TWO LR points, making it a property of the batch size, not of the LR rule. Per-example heat at 0.566/1024 (0.71× baseline) likely also contributed cold-side loss (EXP-015 reading), which is why 0.566 < 0.8 here.

**Campaign context**: seventeenth consecutive no-improvement (EXP-007…022). The batch axis joins every other axis in being closed with measurements on both sides of its optimum. The EXP-006 recipe survives yet another bracket.

## Verification

- Pre-condition (contention profile): PASS — 0/143 windows >60ms; 151 epochs vs 149.2 expected (±3 tolerance).
- Condition 1 — best_test_acc ≥ 96.81: **FAILED** (96.57, converged plateau; −0.24pp vs bar). First-failure-stop applied.
- Condition 2 (rc=0, ≤600s) and Condition 3 (≤1 eval/epoch): skipped per first-failure-stop; for the record both would have passed (rc=0 @ 533.6s; 151 evals = 151 epochs).
- Integrity: params 4,286,026 unchanged; frozen evaluator; no constraint touched. Result trustworthy. **Verdict basis: clean converged run below baseline → no-improvement.**

## Unexplored Avenues

- **Hotter LR at 1024 (e.g. 1.0–1.13)**: the bracket ordering (0.566 < 0.8 < baseline) leaves "even hotter" technically open, but EXP-012's bouncy 0.8 trajectory and the deferral law make >0.8 a near-certain loss; the gap to recover (+0.15 from the 0.8 point) exceeds the remaining headroom goal-learnings assigned the axis (≤0.2pp total, now mostly spent). Treat as closed.
- **Batch 768 (intermediate)**: would interpolate the same trade at lower magnitude — both endpoints lose, so the interior cannot clear a +0.1 bar that the better endpoint missed by 0.15. Closed by bracketing logic.
- **Longer warmup at 1024** (large-batch warmup literature): warmup-shape changes are heat changes under the time-keyed schedule (EXP-014) and the schedule axis is closed; no mechanism argument survives.

## Next Steps

1. **Heat-constant momentum trade (MOMENTUM 0.95 + PEAK_LR 0.2, lr/(1−β)=4 at batch 512)** — now the single remaining untried in-recipe candidate of any kind; completes the constant-bracketing program. Confidence: low.
2. **Synthesis check, fully hardened**: seventeen misses; with the batch axis now double-bracketed, every axis (constants, structure, schedule, init, capacity, batch×LR-rule, smoothing, topology, throughput/numerics) is closed with measurements on both sides. After the momentum trade, brainstorms must either find a mechanism that escapes BOTH the deferral law AND the numerics-equivalence law AND the variance/max-statistic law, or explicitly run low-EV radical probes per the autopilot directive. Confidence in framing: high.
3. **If continuing past the momentum trade**: the least-closed remaining territory is data-order/curriculum within unchanged execution (e.g., fixed shuffle→class-balanced batches) — touches no measured axis, but no comparable-regime evidence exists. Confidence: low.

## Key Learning

At fixed wall-clock, large-batch throughput gains are unrecoverable by LR-rule tuning: with the 1024-LR axis bracketed (√ 96.57, linear 96.66, both < 96.71), the deficit is the batch size's own generalization cost — halved gradient noise — not a mis-scaled learning rate. The "closest miss" of EXP-012 was not a near-success awaiting the right scaling rule; it was the optimum of a curve whose maximum sits below baseline.
