# Log EXP-049: PEAK_LR 0.4 → 0.3 — the heat-down bracket
## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-049.md
- **Plan**: plans/plan-049.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-049
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
M1 executed exactly as planned: a single-line edit to train.py L23 — `PEAK_LR = 0.4` → `PEAK_LR = 0.3` with the comment updated to document the bracket. `git diff --stat` confirms 1 file / ±1 line. Static checks pass: AST parses; constants extracted from the module source give PEAK_LR=0.3, WARMUP_FRAC=0.15; lr_at semantics verified — lr_at(0.15)=0.3 (peak at warmup end), lr_at(0.075)=0.15 (linear warmup midpoint), lr_at(0.575)=0.15 (cosine midpoint = peak/2), lr_at(1.0)≈0 (anneal completes). No model code touched, so no param-count or smoke test needed beyond these (zero new code surface). M2: reusing /tmp/exp046_composite.sh verbatim (verified on disk, 4023 bytes, executable, gate 26ms).

### Surprises & Discoveries
- None. The change surface is one constant; everything matched the plan's projections at implementation time.

### Decisions
- Skipped a separate CPU sanity script (plan anticipated this): the change touches no module/class/loop code, so the AST + extracted-constant + lr_at checks above are the complete sanity surface. A model-building smoke would test unchanged code.

## Run Log

### Run 1
- **Description**: Single gated 300s-budget run of train.py with PEAK_LR=0.3 (0.75× integrated heat, all other constants and signatures byte-identical to the 96.71 baseline recipe). Purpose: sample the unmeasured shallow side of the LR optimum (heat axis bracketed only from above: 0.6 → −0.57 in EXP-010). Expected outcome per pre-registered branches: (i) ≥96.81 improvement if 0.4 sits above the optimum; (ii) mean band 96.42–96.72 → heat axis closed flat-below; (iii) <96.42 → shallow-side loss, 0.4 at interior optimum; (iv) gate/contention kills → infra relaunch. Expect family signatures: dt≈22.5ms, ~139 ep, ~13.4–13.5k steps, params 4,286,026.
- **Job ID / PID**: composite background task brof5a26t; train pid 1664016
- **Log file**: run.log (project root) + composite stdout (task output file)
- **WandB**: N/A
- **Status**: completed (RC=0, PROC_EXITED at tick 34)
- **Started**: 2026-06-10 23:15:20 (GATES_CLEAR poll 1: apps=0, load=5; GATE_DECISION D0=22.2ms, projected_epochs=139, contention_thresh=27.8ms — pristine, family dt as projected for a pure-LR change)
- **Ended**: 2026-06-10 ~23:23:30 (total_seconds 487.7)
- **Observations**: Pristine run end-to-end: 31 post-gate watchdog windows all 21.7–22.8ms, slow_streak never above 0, no kills. ep1 35.70 (within normal scatter per the trajectory criterion); mid-schedule evals tracked slightly below family as expected at 0.75× heat (ep7 63.06 vs family ~65); plateau converged-flat over the last 8 evals (96.25–96.52) at family test_loss (0.1855–0.1882) — NOT still-climbing, so the run was not heat-starved; it converged to a marginally lower basin. Source: composite task brof5a26t output; run.log.
- **Key Metrics**: best_test_acc 96.52 | final_test_acc 96.47 | final_test_loss 0.1882 | training_seconds 300.0 | total_seconds 487.7 | startup 12.1 | peak_vram_mb 1613.0 | num_epochs 139 | num_steps 13,456 | params 4,286,026

## Experimental Adjustments
(none yet)

## Errors & Dead Ends
(none yet)

## Verification Results

### Conditions Checked
- **Integrity pre-condition** — PASS. Windows: 31 post-gate ≥200-step windows, all 21.7–22.8ms (mean ≈22.3 ≤ 23.5; none > 27). num_epochs 139 ∈ [136,142]. num_steps 13,456 ∈ family ledger [13,428–13,515] (EXP-048 protocol). params 4,286,026 exact. training_seconds 300.0. eval count 139 ≤ num_epochs 139. Numerics by trajectory criterion: ep1 35.70, family-shaped climb, converged-flat plateau at family test_loss 0.1882 — normal. Source: composite task brof5a26t output; run.log.
- **Condition 1 (best_test_acc ≥ 96.81)** — FAIL. `grep "^best_test_acc:" run.log` → 96.52%. Below the replicate band [96.70, 96.80] as well, so no replicate pair triggered. First-failure-stop: conditions 2–3 not evaluated for the verdict.
- **Condition 2 (within budget)** — skipped per first-failure-stop (informationally passes: RC=0, total_seconds 487.7 ≤ 600).
- **Condition 3 (eval cadence)** — skipped per first-failure-stop (informationally passes: 139 evals ≤ 139 epochs).

### Informational Metrics
- final_test_loss 0.1882 (family ~0.185–0.187 — at family level, NOT elevated: the run converged; no undertrained signature)
- num_epochs 139 / num_steps 13,456 — exactly family (pure-LR isolation confirmed, as EXP-010 predicted)
- peak_vram_mb 1613.0 (family)
- Plateau shape: converged-flat over last 8 evals (96.25→96.52, best at ep137), family test_loss throughout — branch (ii) signature, not the still-climbing heat-starved shape of branch (iii)

## Human Notes
(autopilot — none)
