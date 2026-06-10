# Plan EXP-064: Gradient-norm clipping at a permissive threshold (max_norm=2.0)

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-064.md

## Closed-axis check
project-insights Medium Importance flags the optimizer/gradient-dynamics POLISH family as closed (GC EXP-030/031, SAM EXP-036, AdamW EXP-043 — none moved top-1). Gradient clipping is a genuinely DISTINCT, never-tested operation: GC centralizes (subtracts per-filter grad mean), clipping bounds the global grad NORM only on outlier steps. It is not a retry of any closed sub-lever — it is the last untested gradient-side knob. The plan acknowledges the honest expectation (near-noise null, per the polish-family pattern) but runs it to definitively close the lever per NEVER-STOP. Single-variable, throughput-neutral, cudagraph-safe.

## Milestones

### Milestone 1: Code change + smoke
- [ ] train.py: insert `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)` between `loss.backward()` (L245) and `optimizer.step()` (L246). Add an inline comment noting EXP-064.
- [ ] Smoke: `ast.parse` OK; `git diff --name-only` == train.py only (one-line insert); confirm the clip is on the eager `model.parameters()` (grads live on the eager params after backward; compiled_model shares them) and sits OUTSIDE the compiled forward.

### Milestone 2: Launch on idle GPU + early gate
- [ ] Pre-launch `nvidia-smi` idle-GPU check; launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background).
- [ ] Gate (~ep8): dt steady ~8ms (clip is one eager norm-reduction/step, no host sync in modern PyTorch clip_grad_norm_ — and the loop already syncs each step for timing, so any cost is absorbed; expect identical throughput), no NaN, loss descending. If dt jumps ≫8ms (unexpected recompile or sync stall), note it.

### Milestone 3: Completion + verification
- [ ] Run exits 0, prints summary; extract metrics, compare to baseline. Expect ~91 ep (throughput unchanged).

## Code Changes
- **train.py (between L245-L246)**: add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)` after `loss.backward()` and before `optimizer.step()`. Why: clips the global gradient norm to 2.0 only on outlier (heavily-distorted AugMix) batches, leaving normal steps untouched, to tame spikes during the high-LR (0.2) plateau where EXP-016/017 placed the recipe at/above its stability edge. Risk/edge case: the clip runs EAGERLY on `model.parameters()` grads AFTER backward, OUTSIDE the compiled forward — so it cannot break the reduce-overhead CUDA graph (contrast EXP-042, which branched INSIDE the compiled forward; and contrast EXP-031's note about COMPILING a grad op — here the clip is eager, so the changing-grad-address concern does not apply). max_norm=2.0 is permissive (clips only true outliers); too-low a threshold would under-step and regress.

## Configuration Changes
- New op: `clip_grad_norm_(max_norm=2.0)`. All else byte-identical to EXP-054 (k=4 WideResNet-20, AugMix-p0.5, Cutout16, cosine peak0.2/warmup0.05/Nesterov m0.9/WD1e-4/LS0.1, batch128, seed42, compile reduce-overhead). num_params unchanged (4,299,866). Rationale for 2.0: permissive — clips outlier spikes without throttling the ~99% of normal steps; consistent with standard practice for a stably-training net (a tighter clip would risk under-stepping, the EXP-016/017-style regression).

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background bash.
- Resources: single idle H20 (pre-check nvidia-smi; relaunch on contention per infra-errors).
- Estimated runtime: ~91 epochs, dt ~8ms, Σdt ~300s, wall ~593s (< 600s; same recipe as EXP-054, clip adds negligible compute).
- Log output: `run.log` in project root.
- Tool skill: none (local).

## Abort Criteria
- Loss NaN/inf or not descending by ep5.
- dt drifts ≫ 8ms (contention, or an unexpected recompile/sync stall from the clip — should not happen): kill, relaunch on clean idle GPU; if the clip itself inflates dt on a clean GPU, that is a real finding (note it), not a contention abort.
- No output / hung > 3 min.

## Verification Protocol

### Verification Procedure
Baseline = **96.45** (from `exp-index.sh baseline`); bar = **96.55**.
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: after exit, `grep -aE "^best_test_acc:" run.log`; parse float; PASS iff `>= 96.55`. (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: `grep -aE "^total_seconds:|^num_epochs:|^num_params:" run.log`; confirm summary printed, `total_seconds < 600`, total wall < 10 min, `num_params == 4,299,866`, `grep -ciaE "nan|traceback|error" run.log` == 0.
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == train.py only; prepare.py/eval untouched; evaluate() once/epoch (loop unchanged); no new deps (clip_grad_norm_ is stock torch); seed 42 unchanged; ran uncontended (steady ~8ms dt).
- Verdict: improvement iff all three pass; no-improvement if a necessary condition fails on a valid run; invalid on scope/dep breach; crash if no metrics.
- Timeout: 10 min wall. Cleanup: `rm run.log` after recording.

### Informational Metrics (Optional)
- peak_vram_mb, num_epochs/num_steps, final_test_loss: `grep -aE "^peak_vram_mb:|^num_epochs:|^num_steps:|^final_test_loss:" run.log` — confirm ~91 ep (throughput unchanged) and compare loss to EXP-054's 0.1968 (a lower loss with flat top-1 would be the polish-vs-top1 signature again).
