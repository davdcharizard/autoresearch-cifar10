# Experiment Log EXP-064: Gradient-norm clipping at a permissive threshold (max_norm=2.0)

## Execution
- **Created**: 2026-06-09
- **Brainstorm**: brainstorm/brainstorm-064.md
- **Plan**: plans/plan-064.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-064
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
One-line change (plan Milestone 1): inserted `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)` between `loss.backward()` (L245) and `optimizer.step()` (now L248), with an inline EXP-064 comment. All else byte-identical to EXP-054.

### Surprises & Discoveries
None. Smoke checks: AST OK; `git diff --name-only` == train.py only; clip sits between backward and step (L247) on the eager `model.parameters()`.

### Decisions
Clip on `model.parameters()` (the eager handle) — grads live there after backward, and compiled_model shares the same parameter tensors. The clip runs eagerly OUTSIDE the compiled forward, so it cannot break the reduce-overhead CUDA graph (contrast EXP-042's in-graph branch; and unlike EXP-031, the clip is not compiled, so the changing-grad-address concern does not apply). max_norm=2.0 is permissive (clips only outlier spikes). Modern clip_grad_norm_ avoids a host sync, and the loop already syncs each step for timing — throughput expected to stay 8ms.

## Run Log

### Run 1
- **Description**: Gradient-norm clip at max_norm=2.0 on the EXP-054 AugMix-p0.5 best. Tests whether bounding outlier gradient spikes (from heavily-distorted AugMix batches) during the high-LR (0.2) plateau — where EXP-016/017 placed the recipe at/above its stability edge — smooths convergence to a marginally better basin and clears the 96.55 bar. Expected: near-noise null (this net never diverged in 64 runs; optimizer/gradient-dynamics polish is a closed family). Launched on idle GPU 1 (GPU 0 also idle at launch).
- **Job ID**: (local, background bash)
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-09
- **Key Metrics**: best_test_acc 96.34% | final_test_loss **0.1939** (LOWER than EXP-054's 0.1968 — polish-vs-top1 signature) | total_seconds 574.1 | num_epochs 91 | num_steps 35218 | num_params 4,299,866 | peak_vram_mb 453.8. dt: 609×8ms + 93×9ms + 1×10ms + 1×11ms (warmup) — uncontended, throughput identical to EXP-054 (clip adds no measurable dt). 0 NaN/error.

## Experimental Adjustments
(none)

## Errors & Dead Ends
(none)

## Verification Results

### Conditions Checked
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: best_test_acc = **96.34%** < 96.55. **FAILED.** (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: not formally evaluated (aborted after condition 1), but for the record all pass: total_seconds 574.1 < 600 ✓, num_params 4,299,866 ✓, num_epochs 91 ✓, 0 NaN/error ✓.
3. **Necessary condition 3 — no hard-constraint violation**: not formally evaluated (aborted), but for the record: `git diff --name-only` == train.py only ✓; clip_grad_norm_ is stock torch (no new dep) ✓; eval once/epoch unchanged ✓; uncontended dt (609×8ms) ✓.

**Verdict**: no-improvement — valid, in-budget, uncontended run that missed the accuracy bar (96.34 < 96.55), a −0.11pp regression vs baseline 96.45. Textbook polish-vs-top1 result: gradient clipping LOWERED final_test_loss to 0.1939 (< EXP-054's 0.1968) but did NOT lift top-1 — exactly the project-insights Medium pattern (GC/SAM/AdamW/PolyLoss all did the same). Gradient clipping joins the closed optimizer/gradient-dynamics polish family; the max_norm=2.0 clip smoothed/better-conditioned the trajectory without converting to accuracy.

