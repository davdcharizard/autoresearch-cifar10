# Experiment Log EXP-069: AugMix mixing-concentration alpha 1.0 → 2.0

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-069.md
- **Plan**: plans/plan-069.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-069
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented plan Milestone 1 — a single-kwarg change to train.py L171: `transforms.RandomApply([transforms.AugMix()], p=0.5)` → `transforms.RandomApply([transforms.AugMix(alpha=2.0)], p=0.5)`, plus an explanatory comment. AugMix `alpha` (default 1.0) is the concentration of the Dirichlet(alpha,…) weights over the 3 augmentation chains AND the Beta(alpha,alpha) clean-mix weight; raising it to 2.0 concentrates both toward their means (mean-preserving: same average strength/coverage, lower per-image variance). All else byte-identical to EXP-054. AST OK; `transforms.AugMix(alpha=2.0)` constructs without error (torchvision 0.24.1, signature pre-verified); `git diff --name-only` == train.py only.

### Surprises & Discoveries
None. torchvision 0.24.1 `AugMix.__init__` signature confirmed `(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0, all_ops=True, ...)` — `alpha` is a first-class kwarg and `AugMix()` in EXP-054 is exactly w3/severity3/chain_depth-1/alpha1.0, so this is a clean single-variable isolation of alpha.

### Decisions
- alpha=2.0 (2× the default 1.0) — large enough to meaningfully reduce the mix variance, small enough to stay near the tuned operating point. Direction chosen UP (not down to 0.5) because higher alpha makes the Dirichlet less likely to collapse to a single dominant chain → more faithful 3-chain blending = the "multi-chain Dirichlet mixing" structure credited for AugMix's win (project-insights line 68), AND Beta(2,2) keeps the clean-mix near 0.5 → preserves full 50% effective coverage (avoids the EXP-055/057 coverage-drop failure mode that lower alpha would risk).

## Run Log

### Run 1
- **Description**: EXP-054 best recipe with the single change AugMix alpha 1.0→2.0 (concentrate the Dirichlet chain-weights + Beta clean-mix toward their means). Tests whether a more faithful, consistent 3-chain-blend-per-image (mean-preserving, wall-neutral) lifts top-1 past the 96.55 bar, or whether the augmentation mixing-distribution sub-axis is null like width/magnitude/coverage. Expected: ~91 ep, dt 8ms, wall ~590s (alpha adds no CPU op cost), best_test_acc near the 96.2–96.45 plateau; honest prior is a within-noise null. Launching on idle GPU 1 (GPU 0 lightly used at 8%/1GB — using GPU 1 which is fully idle).
- **Job ID**: (local, background bash)
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed (exit 0)
- **Started**: 2026-06-10
- **Ended**: 2026-06-10
- **Early gate (ep1, ~step300, run.log L5)**: params 4,299,866 ✓; dt steady 8-9ms ✓ (CPU-side alpha change does not affect the GPU step / compile graph); loss decreasing normally 2.21→1.68, no NaN ✓; img/s ~15,000 ✓ (no dataloader starvation — alpha is op-count-neutral, workers keep pace). Clean start matching the EXP-054 profile.
- **Key Metrics**: best_test_acc **96.25%** (best ep~84-89; −0.20pp vs baseline 96.45, < 96.55 bar) | final_test_acc 96.12% | final_test_loss **0.1998** (> EXP-054's 0.1968 — top-1 AND loss both mildly worse) | **training_seconds 300.0** | **total_seconds 595.9 (< 600 — CLEAN, no wall breach; confirms alpha is wall-neutral as planned)** | num_epochs **91 (= baseline → wall/throughput-neutral)** | num_steps 35316 | num_params 4,299,866 | peak_vram_mb 453.8 (≈ baseline, no extra buffers) | startup 1.9s. dt distribution: 615×8ms / 90×9ms / 1×10ms (single-graph, throughput-neutral). Converged, not underfit (best reached mid-late, final epochs flat 96.12-96.24). 0 NaN/error. GPU 0 lightly used (8%/1GB) but ran on idle GPU 1 (uncontended — dt stayed 8ms).

## Experimental Adjustments
(none)

## Errors & Dead Ends
(none)

## Verification Results

### Conditions Checked
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: best_test_acc = **96.25%** < 96.55. **FAILED** (−0.20pp vs baseline 96.45). (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: total_seconds 595.9 < 600 ✓ (CLEAN, no breach), training_seconds 300.0 ✓, num_params 4,299,866 ✓, num_epochs 91 (= baseline) ✓, summary printed ✓, `grep -ciaE "nan|traceback|error"` == 0 ✓. (Evaluated for completeness; condition 1 already determined the verdict.)
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == `train.py` only ✓; prepare.py/eval untouched ✓; no new deps (torchvision-native `alpha` kwarg) ✓; seed 42 unchanged ✓; evaluate() once/epoch ✓; ran uncontended on idle GPU 1 (dt steady 8ms) ✓.

**Verdict**: no-improvement — clean valid run (Σdt=300s respected, wall 595.9 < 600 with zero caveats, 91 ep = baseline so wall/throughput-neutral exactly as planned) that decisively missed the accuracy bar (96.25 < 96.55, −0.20pp). AugMix alpha 1.0→2.0 (concentrating the Dirichlet/Beta mixing toward their means) mildly regressed BOTH top-1 (96.25) AND eval loss (0.1998 > 0.1968). The lower-variance, more-consistent 3-chain blend slightly UNDER-augmented (averaging 3 chains partially cancels distortions → softer net image), behaving like a mild strength reduction rather than amplifying useful diversity — the −0.20pp sits within the ±0.25pp scalar-knob noise band. Closes the AugMix mixing-weight-distribution (alpha) sub-axis on the UP side; the augmentation-diversity lever is null on its internal-distribution dimension too (consistent with severity EXP-053 and the broader "augmentation exhausted" finding).
