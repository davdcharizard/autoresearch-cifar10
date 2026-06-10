# Experiment Report EXP-073: Nesterov momentum ON → OFF (vanilla heavy-ball SGD)

## Goal
Maximize CIFAR-10 `best_test_acc` (%), higher-is-better, editing only `train.py` within Σdt=300s GPU-compute + total wall ≤ 600s (single H20). Baseline at experiment time: **96.45** (EXP-054, commit 86161d9). Bar: **96.55**.

## Idea & Hypothesis
**Chosen idea**: Toggle `optim.SGD(..., nesterov=True)` → `nesterov=False` (vanilla heavy-ball momentum) at train.py L205, keeping momentum=0.9; everything else byte-identical to EXP-054. **Mechanism**: the last genuinely-untested optimizer cell. Nesterov evaluates the gradient at the look-ahead point (θ + m·v); heavy-ball evaluates at θ. The toggle isolates the look-ahead's contribution at an IDENTICAL effective step magnitude (~1/(1−m), unchanged since momentum is fixed) — so unlike a momentum-coefficient change it cannot perturb the effective LR into the closed LR axis, and unlike the EXP-070/071 architectural probes it cannot destabilize training. The safest possible knob.

**Hypothesis**: best_test_acc within ±0.25pp of 96.45 (clean null or small regression), closing the optimizer-internal axis; any movement is informative about how load-bearing the Nesterov look-ahead is.

## Approach
Single-flag edit `nesterov=True` → `nesterov=False`, momentum held at 0.9. Pre-launch smoke confirmed: AST OK, exactly one `nesterov=` line reading `False`, `optim.SGD(..., nesterov=False)` constructs with the flag set, `git diff --name-only` == train.py only. No deviations.

## Execution
Single clean run on idle GPU 1 (both GPUs idle, uncontended), exit 0, no retries, 0 NaN/error. dt steady **8ms** — heavy-ball is compute-identical to Nesterov (same SGD kernel), throughput-neutral as predicted. Early gate healthy: eval climbed normally (ep5 71.67%, ep6 77.88%), tracking EXP-054 within run-to-run variation. **92 epochs**, training_seconds 300.0, total_seconds 595.5 < 600, peak_vram 453.8 MB.

## Results
**best_test_acc 96.12% (−0.33pp vs baseline 96.45) — a clean regression squarely in the −0.2..−0.6pp scalar-knob band**, with **final_test_loss 0.2020 ALSO worse** than EXP-054's 0.1968. This is NOT a polish-vs-top1 split (where loss improves but top-1 stays flat) — BOTH top-1 and calibration degraded uniformly. The hypothesis (clean null or small regression) was borne out on the regression side, and the result is MORE informative than the expected null: **the Nesterov look-ahead is genuinely load-bearing in this tuned recipe** — it converges the high-LR cosine trajectory to a meaningfully better minimum (≈0.33pp top-1, ≈0.005 loss). Mechanistically, Nesterov's look-ahead gradient provides an effective damping/anticipation that better matches the aggressive peak-LR-0.2 cosine schedule; removing it makes the same schedule slightly less well-controlled, landing in a marginally worse basin. This is consistent with EXP-043 (SGD+Nesterov beat AdamW) — Nesterov is an integral part of the tuned winner, not an incidental default.

**Trajectory fit**: 20th consecutive no-improvement since EXP-054. Closes the LAST untested optimizer-internal cell — the optimizer axis is now fully mapped across family (AdamW EXP-043), dynamics (SAM EXP-036, GC EXP-030/031, grad-clip EXP-064), and now the Nesterov flag. Reinforces the High-importance "scalar/static-knob retune lands −0.2..−0.6pp" pattern (EXP-062/064/065/067/071/073) and the robust-ceiling verdict: every component of the EXP-054 recipe — including the optimizer's look-ahead flag — is at a tuned operating point.

## Verification
- **Necessary condition 1 — `best_test_acc >= 96.55`**: 96.12 < 96.55. **FAILED** (−0.43pp below bar). Stop at first failed condition.
- **Necessary condition 2 — clean completion within budget** (for completeness): training_seconds 300.0 ✓, total_seconds 595.5 < 600 ✓, num_params 4,299,866 UNCHANGED ✓, 0 NaN/error ✓, 92 ep.
- **Necessary condition 3 — no hard-constraint violation**: train.py only ✓; no new deps ✓; seed 42 ✓; evaluate() once/epoch ✓; uncontended (dt 8ms) ✓.

**Verdict: no-improvement.** Clean valid run that missed the bar. Results trustworthy — direct parse, 0 NaN, healthy trajectory. NOT invalid (no breach; optimizer flag, params unchanged) and NOT crash (real interpretable metric).

## Unexplored Avenues
- **Momentum coefficient 0.9→0.95 / 0.85** (brainstorm Ideas 2/3): the remaining optimizer scalars, both CONTRAINDICATED — they scale the effective LR (~1/(1−m)) into the closed LR axis (0.95 ≈ 2× → overshoot like EXP-016; 0.85 ≈ 0.67× → undershoot like EXP-017). Near-certain regressions, lower information than this run. Low priority.
- The optimizer axis is now CLOSED from every angle (family, dynamics, Nesterov flag). Do NOT propose further optimizer changes — SGD+Nesterov m0.9 is the tuned optimum and every perturbation regresses.

## Next Steps
1. **Accept 96.45 as the comprehensively-mapped k=4/300s ceiling** (high confidence) — 20 consecutive misses; every axis now closed including the full optimizer-internal map. The augmentation lever (only top-1 lever) is exhausted (EXP-072 op-set), and every scalar/architectural/optimizer perturbation regresses. Remaining probes are pure plateau-mapping with no positive EV.
2. **Momentum coefficient sweep** (low confidence) — only if a clean optimizer-scalar axis-closer is wanted; contraindicated (effective-LR proxy for the closed LR axis), expected −0.2..−0.5pp.
3. **Re-examine for any genuinely-novel mechanism** (low confidence) — the disciplined NEVER-STOP path: re-read in-scope files / papers for an untested generalization mechanism that is NOT augmentation/optimizer/schedule/capacity/normalization/head/regularizer (all closed). At a 300s/91-epoch budget on a shallow generalization-bound net, no such lever is currently known.
