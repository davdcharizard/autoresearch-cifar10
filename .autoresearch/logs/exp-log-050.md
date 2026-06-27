# Log EXP-050: Additive logit margin on the true class (MARGIN = 0.75)
## Execution
- **Created**: 2026-06-11
- **Brainstorm**: brainstorm/brainstorm-050.md
- **Plan**: plans/plan-050.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-050
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
M1 per plan: three edits to train.py — new constant `MARGIN = 0.75` after LABEL_SMOOTHING; the timed-loop loss subtracts `MARGIN * F.one_hot(targets, NUM_CLASSES).to(outputs.dtype)` from outputs inside the autocast block; the warmup loss gets the identical form (via a `warm_out` temp) for compiled-graph identity. `git diff --stat`: 1 file, +8/−2. CPU sanity (/tmp/exp050_sanity.py, CUDA_VISIBLE_DEVICES=""): (a) params 4,286,026 exact; (b) m=0 identity holds; moderate-confidence logits pay more (1.514 → 2.000); the loss-vs-gap argmin shifts 4.512 → 5.262 (shift 0.750 — the mechanism, measured); (c) at gap = plain-optimum + 0.09, plain CE+LS gradient is +0.007 (pushes the gap DOWN) while margin-loss gradient is −0.071 (decisive upward push) — gradient liveness beyond the LS optimum confirmed; (d) 6-step smoke at lr 0.01 monotone 7.74 → 0.96.

### Surprises & Discoveries
- **LS × margin loss-VALUE interaction**: at gaps far ABOVE the LS optimum, the margin loss reads LOWER than plain CE+LS (the 0.01-weight non-target log-prob terms shrink when the dominant logit drops; p_true barely moves). First sanity draft asserted "margin loss > plain on confident logits" and failed (0.608 vs 0.623). This is correct loss-surface behavior, not a wiring bug — the meaningful invariant is the argmin SHIFT (verified exactly +0.750). Implication for monitoring: the printed train loss will NOT be uniformly inflated vs family; early training higher, late training comparable-or-mixed. Judge numerics on TEST evals only, as planned.

### Decisions
- Sanity check (b)/(c) assertions rewritten around the argmin-shift and gradient-sign invariants instead of naive loss-value comparisons (see Surprises). No production-code deviation from plan.

## Run Log

### Run 1
- **Description**: Single gated 300s-budget run of train.py with the additive true-class logit margin (MARGIN=0.75) in the training loss only — first probe of the loss-geometry class, targeting the measured decision-boundary-limited ceiling (EXP-011/032). All signatures projected byte-identical to baseline (the subtract is a fused elementwise op on 512×10). Expected outcome per pre-registered branches: (i) ≥96.81 improvement if boundary placement is the binding limitation; (ii) mean band 96.42–96.72 → static margin absorbed; (iii) <96.42 → margin harmful under heavy aug (destroyed-label amplification); (iv) gate/contention → infra.
- **Job ID / PID**: composite background task b1xu3r1o9; train pid 1737561
- **Log file**: run.log (project root) + composite stdout (task output file)
- **WandB**: N/A
- **Status**: completed (RC=0, PROC_EXITED at tick 33)
- **Started**: 2026-06-11 03:19:56 (GATES_CLEAR poll 1: apps=0, load=6; GATE_DECISION D0=22.7ms, projected_epochs=136, contention_thresh=28.4ms — family dt; margin op confirmed throughput-free)
- **Ended**: 2026-06-11 ~03:28:02 (total_seconds 485.6)
- **Observations**: Pristine run: 30 post-gate windows all 22.0–22.7ms, slow_streak 0 throughout, no kills. Plateau converged-flat over last 8 evals (96.05–96.19) — a uniformly DEPRESSED plateau, the same active-negative shape as EXP-047. The headline mechanistic datum: final_test_loss 0.1505 vs family ~0.185 — the margin substantially IMPROVED test cross-entropy while accuracy fell 0.4pp below the recipe mean. Boundary pressure moved logit geometry exactly as designed (wider gaps → higher p_true → lower CE) but moved argmaxes the WRONG way. Source: composite task b1xu3r1o9 output; run.log.
- **Key Metrics**: best_test_acc 96.19 | final_test_acc 96.15 | final_test_loss 0.1505 | training_seconds 300.0 | total_seconds 485.6 | startup 12.0 | peak_vram_mb 1613.0 | num_epochs 139 | num_steps 13,431 | params 4,286,026

## Experimental Adjustments
(none yet)

## Errors & Dead Ends
(none yet)

## Verification Results

### Conditions Checked
- **Integrity pre-condition** — PASS. 30 post-gate ≥200-step windows all 22.0–22.7ms (mean ≈22.3 ≤ 23.5; none > 27). num_epochs 139 ∈ [136,142]. num_steps 13,431 ∈ family ledger [13,428–13,515]. params 4,286,026 exact. training_seconds 300.0. eval count 139 ≤ 139. Numerics by trajectory criterion: family-shaped climb, converged-flat (depressed) plateau; test_loss shift is the DESIGNED effect (plan flagged test_loss informational, not an integrity gate, for this experiment). Source: composite task b1xu3r1o9; run.log.
- **Condition 1 (best_test_acc ≥ 96.81)** — FAIL. 96.19% — also below the replicate band [96.70, 96.80]. First-failure-stop: conditions 2–3 not evaluated for the verdict.
- **Condition 2 (within budget)** — skipped per first-failure-stop (informationally passes: RC=0, 485.6s ≤ 600).
- **Condition 3 (eval cadence)** — skipped per first-failure-stop (informationally passes: 139 ≤ 139).

### Informational Metrics
- final_test_loss 0.1505 (family ~0.185 — substantially LOWER; the margin's gap-widening improved CE while costing accuracy: improved-loss/worse-argmax, the inverse face of the EXP-011/032 signature)
- num_epochs 139 / num_steps 13,431 — family exactly (margin op throughput-free, as projected)
- peak_vram_mb 1613.0 (family)
- Plateau shape: converged-flat at a uniformly depressed level (96.05–96.19 last 8) — active negative, not noise

## Human Notes
(autopilot — none)
