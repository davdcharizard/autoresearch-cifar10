# Experiment Log EXP-065: Higher label smoothing (LABEL_SMOOTHING 0.1 → 0.15)

## Execution
- **Created**: 2026-06-09
- **Brainstorm**: brainstorm/brainstorm-065.md
- **Plan**: plans/plan-065.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-065
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
One-constant change (plan Milestone 1): train.py L27 `LABEL_SMOOTHING` 0.1 → 0.15, inline comment updated to note EXP-065. The constant feeds `F.cross_entropy(outputs, targets, label_smoothing=LABEL_SMOOTHING)` at L243. All else byte-identical to EXP-054.

### Surprises & Discoveries
None. Smoke checks: AST OK; `git diff --name-only` == train.py only; LABEL_SMOOTHING=0.15 confirmed feeding cross_entropy at L243.

### Decisions
No deviations. Probes the UPPER side of the LS optimum (EXP-023 tested only lower, 0.05, on the old TA recipe) on the current AugMix best — a RETUNE of an existing regularizer, compute-/throughput-neutral, cudagraph-safe (LS is a host-side scalar arg to cross_entropy, outside the compiled forward).

## Run Log

### Run 1
- **Description**: LS 0.1→0.15 on the EXP-054 AugMix-p0.5 best. Tests whether softer targets better match the soft, multi-chain-mixed AugMix inputs (50% of the batch) and clear the 96.55 bar — the LS×AugMix interaction EXP-023 (TA recipe, lower-direction only) did not probe. Expected: near-noise null or small regression (project-insights Medium: adding regularization hurts at this short budget; EXP-023's lower-side regression suggests 0.1 is near the peak), bracketing the LS optimum at 0.1 either way. Launched on idle GPU 1 (GPU 0 also idle).
- **Job ID**: (local, background bash)
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-09
- **Key Metrics**: best_test_acc 96.17% (best at ep89) | final_test_loss 0.2478 (≫ EXP-054's 0.1968 — higher-LS model is less confident → higher plain-CE eval loss; over-regularized) | **training_seconds 300.0 (compute budget EXACTLY respected)** | **total_seconds 602.5 (WALL BREACH: >600 by 2.5s)** | num_epochs 92 | num_steps 35780 | num_params 4,299,866 | peak_vram_mb 453.8. dt: 659×8ms + 56×9ms — uncontended, throughput identical to EXP-054 (LS change is compute-free). 0 NaN/error.

## Experimental Adjustments
(none)

## Errors & Dead Ends
(none)

## Verification Results

### Conditions Checked
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: best_test_acc = **96.17%** < 96.55. **FAILED.** (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: **WALL BREACH** — total_seconds 602.5 > 600 (by 2.5s). For the record: training_seconds 300.0 (the actively-gated compute budget EXACTLY respected), num_params 4,299,866 ✓, 0 NaN/error ✓. The 2.5s overrun is eval+dataloader wall (92 evals + AugMix CPU variance), NOT compute (dt clean 8ms), and NOT caused by the compute-free LS change — it is the AugMix recipe's documented run-to-run wall variance (2nd breach after EXP-061's 604.6s).
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == train.py only ✓; no new deps ✓; seed 42 unchanged ✓; uncontended ✓.

**Verdict deliberation (invalid vs no-improvement)**: the 600s wall is a hard constraint and 602.5 > 600, which could argue for `invalid`. Chose **no-improvement** (consistent with the EXP-061 precedent) because: (a) condition 1 (metric) fails FIRST and DECISIVELY on a fully trustworthy value (96.17 ≪ 96.55, best at ep89); (b) the actively-gated 300s COMPUTE budget was respected EXACTLY (training_seconds 300.0, clean 8ms dt) — the run trained fairly; (c) the 2.5s wall overrun is documented AugMix base-recipe variance (goal-learnings: "a replication may exceed 600s"), NOT caused by the compute-free LS change; (d) the breach does not make the metric untrustworthy. Classifying as `invalid` would wrongly DISCARD a genuine, informative negative result (higher LS hurts → LS optimum bracketed at 0.1) over a 2.5s base-recipe wall-variance overrun. The breach is documented prominently here, in the report, and strengthened in infra-errors (recurring 2nd breach).

**Verdict**: no-improvement — valid training run (300s compute respected) that decisively missed the accuracy bar (96.17 < 96.55, −0.28pp vs baseline). Higher LS over-regularized: BOTH top-1 (96.17) AND confidence (eval CE 0.2478 ≫ 0.1968) worse. Brackets the LS optimum at 0.1 (EXP-023's 0.05 hurt below; this 0.15 hurts above). Wall breach noted (602.5 > 600, base-recipe variance).

