# Experiment Log EXP-067: BN momentum reduction (0.1 → 0.05)

## Execution
- **Created**: 2026-06-09
- **Brainstorm**: brainstorm/brainstorm-067.md
- **Plan**: plans/plan-067.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-067
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented plan Milestone 1: added `momentum=0.05` to all four `nn.BatchNorm2d(...)` constructor sites — BasicBlock bn1 (L71), bn2 (L75), shortcut BN (L83), stem bn1 (L103), each with an inline EXP-067 note. AST OK; `grep -c "momentum=0.05"` == 4; `git diff --name-only` == train.py only. All else byte-identical to EXP-054.

### Surprises & Discoveries
None. The four BN sites are the only BatchNorm constructors in the net; the momentum kwarg is a host-side BN attribute that does not enter the compiled forward's compute graph, so no throughput/cudagraph effect is expected.

### Decisions
No deviations. momentum=0.05 applied uniformly to all BN layers (no per-layer differentiation) — the simplest single-variable test of a longer running-stat EMA window.

## Run Log

### Run 1
- **Description**: BN momentum 0.1→0.05 on the EXP-054 AugMix-p0.5 best — lengthens the running-stat EMA window to lower eval-time stat estimation variance over the noisy AugMix operating distribution. Compute-/throughput-neutral (BN momentum is a host-side attribute outside the compiled forward), single-graph, wall-safe. Expected: ~91 ep, dt 8ms, near-noise null or mild regression (the cosine-to-0 tail already stabilizes running stats); watch final_test_loss as the sensitive secondary signal. Launching on idle GPU 1 (GPU 0 also idle).
- **Job ID**: (local, background bash)
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-10
- **Key Metrics**: best_test_acc **96.15%** (best ep84; −0.30pp vs baseline 96.45, < 96.55 bar) | final_test_loss 0.2058 (> EXP-054's 0.1968) | **training_seconds 300.0** | **total_seconds 589.2 (< 600 — CLEAN, no wall breach)** | num_epochs 89 | num_steps 34570 | num_params 4,299,866 | peak_vram_mb 453.8. dt: 586×8ms + 101×9ms (steady, throughput unchanged — BN momentum is compute-free, single-graph confirmed). 0 NaN/error. GPU 0 idle throughout (uncontended).

## Experimental Adjustments
(none)

## Errors & Dead Ends
(none)

## Verification Results

### Conditions Checked
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: best_test_acc = **96.15%** < 96.55. **FAILED** (−0.30pp vs baseline 96.45). (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: total_seconds 589.2 < 600 ✓ (CLEAN — no wall breach, unlike EXP-065/066), training_seconds 300.0 ✓, num_params 4,299,866 ✓, summary printed ✓, 0 NaN/error ✓.
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == train.py only ✓; no new deps ✓; seed 42 unchanged ✓; evaluate() once/epoch ✓; uncontended (steady 8ms dt, GPU 0 idle) ✓.

**Verdict**: no-improvement — clean valid run (Σdt=300s respected, total wall 589.2 < 600, no caveats) that missed the accuracy bar (96.15 < 96.55, −0.30pp). The longer BN running-stat EMA window (momentum 0.05) mildly hurt BOTH top-1 (96.15) AND eval loss (0.2058 > 0.1968): with the cosine-to-0 near-frozen-weight tail, default-momentum running stats are already well-estimated, so a longer window folds slightly-staler higher-LR batch stats into the eval estimate — the predicted mild regression. Closes the BN-momentum knob (lower-than-default hurts).
