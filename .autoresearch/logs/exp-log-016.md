# EXP-016: Linear-to-zero anneal (replace cosine post-warmup branch)

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-016.md
- **Plan**: plans/plan-016.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-016
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (necessary condition 1 failed: best_test_acc 96.21 < 96.81 bar; clean valid run — research no-improvement, not infra)

## Implementation Notes

### Summary

Milestone 1 executed per plan: branched `autoresearch/exp-016` from `autoresearch/dev` (clean @ 1990397), replaced the cosine return in `lr_at()` with `PEAK_LR * (1 - q)` and updated the function comment ("then linear to 0") — git diff --stat shows 1 file, +2/−2 (formula + comment lines). py_compile passed and the analytic shape spot-check verified all four anchors exactly: lr_at(0.075)=0.2 (warmup branch untouched), lr_at(0.15)=0.4 (peak), lr_at(0.575)=0.2 (anneal midpoint — equal to cosine's by symmetry), lr_at(1.0)=0.0 (endpoint). Launch via the composite protocol (pre-check + train + inline watchdog in one chain, task bnkebo0fk); GPU 0 was free at pre-check.

### Surprises & Discoveries

- None at implementation time. The `q` normalization made the family swap a genuine one-expression change; no other code depends on the anneal shape.

### Decisions

- Updated the comment line along with the formula (plan flagged this as optional) — keeps the file's documentation honest at the cost of a 2-line rather than 1-line diff; zero functional impact.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: task bnkebo0fk (local background; composite launcher + inline watchdog, kills run on 4 consecutive >30ms windows)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log
- **WandB**: N/A
- **Status**: completed (clean)
- **Started**: 2026-06-10 09:30
- **Ended**: 2026-06-10 09:39

Description:
- Single training run of the baseline recipe with the cosine anneal swapped for linear-to-zero at identical peak (0.4), warmup (0.15), and total LR-time integral — the first pure heat-DISTRIBUTION probe (every prior schedule change moved the integral, a certified-closed axis). Expected: byte-identical throughput signatures (~139 epochs, dt ~22.3ms, 1613MB, 4,286,026 params); hypothesis (Defazio et al. 2310.07831) predicts behind-then-crossover trajectory (cooler early-mid, hotter late-mid) and best_test_acc ≥ 96.81, with late-epoch eval variance at least baseline-level.

Observations:
- Clean execution: watchdog zero SLOW events; post-hoc windowed profile 0 of 266 windows > 30ms (mean 22.4ms); 138 epochs / 13362 steps (~projection; −1 epoch is normal jitter); total 510.9s; VRAM 1613.0, params 4,286,026 (source: task bnkebo0fk output; run.log windowed profile)
- HYPOTHESIS REFUTED — no crossover, and a NEW failure shape: the trajectory ran BEHIND baseline the entire schedule (ep 20: 76.1; ep 60: 87.1; ep 100: 91.5; ep 130: 95.5) and was STILL CLIMBING at cutoff — best 96.21 first reached at the FINAL epoch (138), last evals: 95.69, 95.52, 95.73, 95.73, 96.01, 96.02, 96.17, 96.21 (source: run.log eval lines)
- Mechanism: linear's hotter late-middle is a liability under this metric/budget. At ep 130 (p≈0.94) linear lr ≈ 0.028 vs cosine ≈ 0.005 — the network keeps taking large noisy steps until the very end and never settles into a converged plateau. The max-statistic then has only ONE near-peak eval to harvest; baseline's cosine cold-tail produces ~10 converged evals in the 96.4–96.7 range to max over. The Defazio result is calibrated on FINAL-value comparisons; under best-over-evals + fixed wall clock, cosine's "theoretically suboptimal" flat tail is load-bearing (source: run.log last-8 evals; lr arithmetic from lr_at)

Key Metrics:
- best_test_acc: 96.21% @ ep 138 — final epoch, still climbing (source: run.log summary; bar was 96.81)
- total_seconds: 510.9; training_seconds: 300.0; num_epochs: 138; num_steps: 13362; peak_vram_mb: 1613.0; num_params: 4,286,026; final_test_loss: 0.1934 (source: run.log summary block)

## Verification Results

<!-- Filled after the experiment completes successfully.
     If ANY necessary condition fails, remaining conditions are not evaluated. -->

### Conditions Checked

- **Pre-condition — contention sanity (Protocol Findings EXP-011/014)**: num_epochs 138 within ~10% of ~139; watchdog zero SLOW events; post-hoc profile 0/266 windows > 30ms (mean 22.4ms). **CLEAN — conditions evaluable.** (source: run.log windowed profile; task bnkebo0fk output)
- **Condition 1 — best_test_acc ≥ 96.81 (baseline 96.71 + 0.1)**: parsed 96.21 from `grep "^best_test_acc:" run.log`. **FAILED** (96.21 < 96.81; −0.50pp vs baseline). (source: run.log summary block)
- **Condition 2 — total ≤ 600s**: skipped — aborted after prior failure (observed informally: 510.9s would have passed)
- **Condition 3 — validation ≤ once/epoch**: skipped — aborted after prior failure (observed informally: 138 eval lines = 138 epochs would have passed)

### Informational Metrics

- Not collected per protocol (necessary condition failed). Informal: peak_vram_mb 1613.0 (= baseline), num_epochs 138 (≈ baseline), num_params 4,286,026 (= baseline), final_test_loss 0.1934 (slightly above baseline ~0.19 — consistent with not-yet-converged tail).

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
