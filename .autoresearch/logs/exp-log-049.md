# EXP-049: Augmentation cooldown (EXP-034) + Gradient Centralization (EXP-031) combined

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-049.md
- **Plan**: plans/plan-049.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-049
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented plan-049 Milestone 1: applied BOTH proven throughput-neutral near-misses to `train.py`, unchanged from their originals, with no code interaction between them. **Gradient Centralization (EXP-031, 3 edits)**: added module-level `_gradient_centralize(grads)` (out-of-place per-output-unit mean-subtract over fan-in dims) wrapped as `_gc_compiled = torch.compile(...)` in DEFAULT mode; hoisted `gc_params = [p for p in model.parameters() if p.ndim > 1]` once after the model/compile setup; inserted the call site between `loss.backward()` and `optimizer.step()` (compute centralized grads, reassign `p.grad`). **Augmentation cooldown (EXP-034, 4 edits)**: added `COOLDOWN_FRAC = 0.10`; added `train_tf_clean` (full pipeline minus `TrivialAugmentWide`); added `aug_cooled` flag with an epoch-boundary swap of `train_set.transform` once `total_training_time/TIME_BUDGET_S ≥ 0.90` (with a `>>> aug cooldown ON` marker); gated the GPU Cutout call behind `if not aug_cooled`. Smoke test passed: AST clean, `git diff --name-only` = `train.py` only, num_params 4,299,866 (unchanged), GC targets = 23 (`ndim>1`), COOLDOWN_FRAC 0.1, GC per-output-unit mean after centralization ≈ 3e-8 (zeroed), `train_tf_clean` contains no TrivialAugment.

### Surprises & Discoveries
- None during implementation — both edit sets are verbatim re-applications of mechanisms already validated in EXP-031 and EXP-034. The bare `python` interpreter lacks `torchvision`; the project env is reached via `uv run python` (as expected — runs use `uv run train.py`).

### Decisions
- Left the tail LR schedule untouched (frozen near-zero, as in EXP-034) — NOT reheated — per the EXP-035 caution that reheating the clean tail regresses (96.12 < baseline).
- GC runs every step including the clean cooldown phase (it is a full-run regularizer; the cooldown only changes the input pipeline, not the gradient processing).
- GC uses a SEPARATE DEFAULT-mode compiled callable, independent of the model's `reduce-overhead` compile, because `zero_grad` reallocates grads each step (CUDA-graph static addresses would be invalid) — exactly as EXP-031 established.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID at launch — background Bash)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09
- **Ended**: 2026-06-09 (402.3s total wall)

Description:
- Running the baseline k=4 ResNet-20 recipe with BOTH the EXP-034 augmentation cooldown (@0.10, clean tail from frac 0.90) AND the EXP-031 compiled+hoisted Gradient Centralization applied together, on idle GPU 1. Tests whether the two orthogonal sub-noise near-misses add to clear the +0.1 bar (96.32): GC lowers loss / better-conditions the weights, and the clean cooldown tail is hypothesized to let that better-conditioned state surface as top-1 by removing the aug-train↔clean-test mismatch. Expect throughput-neutral dt~8ms / ~91 epochs (no epoch confound), the cooldown marker firing once near frac 0.90, and best_test_acc somewhere in ~96.0–96.5.

Observations:
- **Throughput-neutral, clean fair test**: dt steady 8ms (647×8ms, 55×9ms), num_epochs 91 = baseline → NO epoch confound, NO CUDA-graph break (would be 14-16ms). The GC out-of-place grad reassignment did not perturb the model's reduce-overhead forward graph, as predicted (EXP-031). (source: run.log, dt extraction)
- **Cooldown fired correctly once** at `>>> aug cooldown ON at ep 82 frac 0.90` — the clean tail DID lift top-1: pre-cooldown base 95.77 (ep80) → post-cooldown peak 96.13 (ep86), a +0.36 climb (healthy, comparable to EXP-034's tail climb). The cooldown mechanism worked. (source: run.log eval lines L80-91)
- **Combination REGRESSED vs both components**: best_test_acc 96.13 < baseline 96.22 (−0.09pp) AND < EXP-034 cooldown-alone 96.26 (−0.13pp). final_test_loss 0.1983 ≈ baseline 0.195 — crucially NOT EXP-031's lower 0.1894, so GC's loss advantage WASHED OUT in the combination. (source: run.log summary)
- Two reasons the combination underperformed: (a) the pre-cooldown base here (95.77 @ep80) was LOWER than EXP-034's (96.05 @ep83) — GC did not raise the augmented-phase base; (b) GC's standalone loss benefit (0.1894) did not persist (0.1983), so there was no better-conditioned state for the cooldown to "cash in." The two sub-noise levers did NOT add. (source: run.log)
- Clean completion: training_seconds 300.0, total_seconds 402.3 < 600, startup 1.1s, peak_vram 469.8 MB, no NaN/traceback. (source: run.log summary)

Key Metrics:
- best_test_acc: 96.13% (−0.09pp vs baseline 96.22; −0.19pp vs bar 96.32) @ ep86; final_test_acc 96.07% @ ep91; final_test_loss 0.1983
- num_epochs: 91; num_steps: 35,108; training_seconds: 300.0; total_seconds: 402.3; startup_seconds: 1.1; peak_vram_mb: 469.8; num_params: 4,299,866 (unchanged)
- dt: steady 8ms (647×8ms, 55×9ms) — throughput-neutral, identical to baseline/EXP-031/EXP-034

## Verification Results

### Conditions Checked
1. **Primary necessary condition (`best_test_acc ≥ 96.32`)** — FAIL. best_test_acc 96.13 < 96.32 (−0.19pp vs bar, −0.09pp vs baseline 96.22). → no-improvement. (source: run.log `best_test_acc: 96.13%`)
2. **Run completes cleanly within budget** — PASS. Summary printed; total_seconds 402.3 < 600; training_seconds 300.0; num_params 4,299,866 (unchanged); no crash/NaN. (source: run.log summary)
3. **No hard-constraint violations** — PASS. `git diff --name-only` = `train.py` only; prepare.py/eval untouched; `evaluate()` once/epoch (loop structure unchanged); no new deps (GC + cooldown use only torch/torchvision already present); seed 42 unchanged; deterministic — no seed hacking. Cooldown fired once (`grep -c` = 1). (source: git diff, run.log)

Verdict: **no-improvement** (primary condition fails, −0.09pp vs baseline; clean fair throughput-neutral test at matched epochs). Run completed cleanly → Outcome = completed.

### Informational Metrics
- num_epochs / num_steps: 91 / 35,108 (throughput-neutral, = baseline ~91).
- peak_vram_mb: 469.8 (≈ baseline).
- final_test_loss: 0.1983 (≈ baseline 0.195; NOT EXP-031's 0.1894 — GC's loss benefit did not survive the combination).
- pre→post cooldown climb: 95.77 (ep80) → 96.13 (ep86), +0.36 over ~6 clean epochs (cooldown mechanism worked; base was just too low).

## Errors & Dead Ends

## Human Notes

> (none — autopilot)
