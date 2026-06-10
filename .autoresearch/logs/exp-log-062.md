# Experiment Log EXP-062: WARMUP_FRAC isolation (0.05 → 0.10)

## Execution
- **Created**: 2026-06-09
- **Brainstorm**: brainstorm/brainstorm-062.md
- **Plan**: plans/plan-062.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-062
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Single-constant change implementing plan Milestone 1: train.py L24 `WARMUP_FRAC` 0.05 → 0.10, inline comment updated to note EXP-062. The `lr_at_fraction` function (L35-41) is untouched — it reads WARMUP_FRAC, so doubling the constant lengthens the linear-warmup ramp (~4.5 ep → ~9 ep at the realized ~91-epoch throughput) and slightly shortens the high-LR cosine phase. All else byte-identical to EXP-054 (k=4 WideResNet-20, AugMix-p0.5, Cutout16, cosine peak0.2/Nesterov m0.9/WD1e-4/LS0.1, batch128, seed42, compile reduce-overhead).

### Surprises & Discoveries
None. Smoke checks confirmed: `ast.parse` OK; `git diff --name-only` == train.py only (one-line change); LR sanity `lr_at_fraction(0.0)=0.0000`, `lr_at_fraction(0.10)=0.2000` (peak reached exactly at frac=0.10), `lr_at_fraction(1.0)=0.000000` (still anneals to 0).

### Decisions
No deviations from the plan. Throughput- and wall-neutral change (warmup only redistributes the existing time-fraction LR schedule, adds zero work) — no EXP-061-style wall-overrun risk.

## Run Log

### Run 1
- **Description**: Single-variable LR-warmup probe — `WARMUP_FRAC` 0.05→0.10 on the EXP-054 best recipe. Tests whether a longer linear warmup stabilizes early training under noisy AugMix gradients at PEAK_LR=0.2, reaching a marginally better basin. Expected: near-noise null on this deeply-mapped plateau (LR regime finely balanced per EXP-016/017), small regression possible if longer warmup under-trains the mid-cosine phase. Launched on idle GPU 1 (GPU 0 has foreign proc PID 1200082, 814MiB).
- **Job ID**: (local, background bash)
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-09
- **Key Metrics**: best_test_acc 96.18% | final_test_loss 0.1975 | total_seconds 596.0 | num_epochs 91 | num_steps 35294 | num_params 4,299,866 | peak_vram_mb 453.8. dt distribution: 618×8ms + 86×9ms + 1×25ms (compile warmup) — uncontended, throughput identical to EXP-054. 0 NaN/error.

## Experimental Adjustments
(none)

## Errors & Dead Ends
(none)

## Verification Results

### Conditions Checked
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: best_test_acc = **96.18%** < 96.55. **FAILED.** (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: not evaluated (aborted after condition 1 failed). For the record: total_seconds 596.0 < 600 ✓, num_params 4,299,866 ✓, num_epochs 91 ✓, 0 NaN/error ✓ — the run WAS valid and within budget; it simply missed the accuracy bar.
3. **Necessary condition 3 — no hard-constraint violation**: not evaluated (aborted). For the record: `git diff --name-only` == train.py only ✓; uncontended dt (618×8ms) ✓.

**Verdict**: no-improvement — valid, in-budget, uncontended run that missed the accuracy bar (96.18 < 96.55), in fact a small regression vs baseline 96.45 (−0.27pp). Confirms the LR regime is finely balanced: doubling WARMUP_FRAC (0.05→0.10) eats into the mid-training high-LR cosine phase and slightly under-trains, consistent with the EXP-016/017 finding that ±LR perturbations cost ~0.5pp.

