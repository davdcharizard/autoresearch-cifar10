# Experiment Log EXP-030: Concat avg+max global pooling head

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-030.md
- **Plan**: plans/plan-030.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-030 (cut from autoresearch/dev @ 1990397)
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: failed (verification condition 1: best 95.80 < 96.81; clean Run 2 stands, Run 1 discarded as contention-killed)

## Implementation Notes

### Summary
Plan-030 Milestone 1 implemented in train.py (+4/−2): (1) `ResNet.__init__`: `self.fc = nn.Linear(2 * w3, num_classes)`; (2) `ResNet.forward`: `out = torch.cat([F.adaptive_avg_pool2d(out, 1), F.adaptive_max_pool2d(out, 1)], dim=1)` replacing the avg-only pool; the existing `.view`/`fc` lines unchanged. Optimizer groups pick up the wider fc automatically (ndim>1 → decay group); Kaiming init via existing `_weights_init`. Expected params 4,288,586 (+2,560). AST OK; diff reviewed — timed step body untouched.

### Surprises & Discoveries
- None at implementation time — exactly the planned 2-site change.

### Decisions
- Kept `.view` after the concat: the pooled tensor is (N, 512, 1, 1); flattening a 1×1-spatial tensor is layout-invariant, so no `.reshape` needed even under channels_last.

## Run Log

### Run 1
- **Description**: Full 300s-budget run on GPU 0. Single architecture change vs baseline: the classifier consumes concat(avg, max) pooled features (512-d) instead of avg-pooled only (256-d) — cifar10-fast-pedigree head modernization, the last law-compliant untried axis. Expected: dt 22.4±0.5ms (head ≈ free), ~139 epochs, family-tracking early trajectory, plateau shifted by the head's true effect; success iff best ≥ 96.81. Noise-band plateau (±0.15 of mean 96.57) closes the head axis.
- **Job ID**: local background composite, task bwvwwbte1 (gates: STARTUP_KILL tick 10; EARLY_DT_KILL 3 consecutive >27ms in first 7 ticks; CONTENTION_KILL 4 consecutive >30ms; NaN/inf guard)
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: CONTENTION_KILLED at ~62% (step ~10400): windows 22.7 → 26.0 → 40.0 → 44.4 → 42.0 → 42.0 (4 consecutive >30ms) — genuine foreign contention confirmed post-kill: PID 754819 live on GPU 0 (1014MiB, 34% util). Discarded per protocol; rerun once when GPU 0 frees.
- **Started**: 2026-06-10T15:23:12Z
- **Ended**: 2026-06-10T15:30:43Z (rc=143, killed)
- **Observations**: Pre-contention signal worth keeping: dt 22.3–23.3ms (head ≈ +0.4ms vs 22.4 — near-free as predicted), but the early trajectory shows a REAL deferral toll: ep1 19.87 (family ~38), ep5 49.74 (~64), ep10 67.79 (~78) — the re-initialized 512-d head + max-path takes longer to organize than predicted (EXP-018/020 pattern, milder). Early train loss also elevated (step 900: 1.56 vs family ~1.33), converging toward family by step ~6000. Plateau judgment must wait for the clean rerun.
- **Key Metrics**: partial only (killed): 107 evals, best-so-far at kill ~62% progress. Source: task bwvwwbte1 + run.log (discarded).

### Run 2 (contamination rerun, same code)
- **Job ID**: local background composite, task budkf6d0u — polls GPU 0 every 30s (up to 60 min) until the foreign PID clears, then launches with the same gates as Run 1
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed rc=0, profile CLEAN — `windows>27ms: 0 of 263 | mean win 22.7 ms | expected epochs 137.2` vs 137 actual. This run stands for verification. (GPU 0 freed after ~3.5 min of polling; user confirmed.)
- **Started**: 2026-06-10T~15:32Z (launch after wait; end stamp 15:40:45Z)
- **Ended**: 2026-06-10T15:40:45Z
- **Observations**: Replicates and extends Run 1's early signal: the head costs only +0.3ms dt (22.7 vs 22.4 → 137 epochs, −2) but imposes a TRAJECTORY-LONG optimization drag, not a one-time init toll — behind family at every waypoint (ep1 18.4 vs ~38; ep10 66.5 vs ~78; ep30 80.8 vs ~93; ep60 85.6 vs ~94.9; ep90 88.1 vs ~96; ep110 93.4 vs ~96.4) and STILL CLIMBING at cutoff (best 95.80 at ep136, the EXP-016 starvation signature). Final plateau 95.6–95.8, test_loss 0.2047 vs ~0.185. Mechanism reading: the max-pool pathway routes gradients only through per-channel argmax positions — discontinuous, high-variance credit assignment that destabilizes head learning during the long high-LR phase; the trunk's features also serve two heads' demands. The cifar10-fast pedigree comes from a ~10-epoch regime where final-iterate speed dominates; under our 137-epoch max-statistic the head never catches up.
- **Key Metrics**: best 95.80 | final 95.73 | final_test_loss 0.2047 | total 499.5s | startup 14.4s | VRAM 1637.0 | 137 epochs / 13209 steps | params 4,288,586 ✓ (+2,560 exact) | final-7 median 95.71. Source: task budkf6d0u + run.log.

## Experimental Adjustments

## Errors & Dead Ends

## Verification Results

First-failure-stop per plan-030 on Run 2 (the clean run; Run 1 contention-killed and discarded — foreign PID confirmed live at kill time).

### Conditions Checked
1. **best_test_acc ≥ 96.81**: **FAIL** — `grep "^best_test_acc:" run.log` → 95.80%. Pre-condition profile PASSED (0/263 windows >27ms; epochs 137 vs 137.2 expected). Gap −1.01 to the bar, −0.91 to baseline, −0.77 to the baseline mean — far outside the noise band; a genuine architecture deficit. First failure → stop.
2. **Completes within budget**: skipped (would pass: rc=0, total 499.5s ≤ 600).
3. **Validation ≤ once/epoch**: skipped (would pass: eval_lines 137 = num_epochs 137).

### Informational Metrics
- Head dt cost: 22.7 − 22.4 = **+0.3ms/step** (~1.3%) → 137 epochs (−2); throughput-free as predicted — the loss is optimization quality, not epochs.
- num_params 4,288,586 (+2,560 exact) | VRAM 1637.0 (+24) | startup 14.4s
- Deferral check: FAILED the hypothesis's "family-tracking early" prediction — behind at EVERY waypoint and still climbing at cutoff (best at ep136 = EXP-016's starvation signature); the toll is trajectory-long, not init-localized. Both runs agree on the early shape (Run 1 ep10 67.8 / Run 2 66.5).

## Human Notes
