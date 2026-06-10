# Plan EXP-073: Nesterov momentum ON → OFF (vanilla heavy-ball SGD)
- **Created**: 2026-06-10
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-073.md

## Closed-axis check
This is the last genuinely-UNTESTED optimizer cell, not a closed-axis retry. The optimizer FAMILY (AdamW EXP-043) and dynamics (SAM EXP-036, GC EXP-030/031, grad-clip EXP-064) are closed, but the Nesterov flag itself has never been toggled — SGD+Nesterov has been the fixed tuned setting since EXP-000. It does NOT contradict a High-importance insight: it is NOT an add-a-regularizer move (convergence-bound closure, project-insights line 68 — it changes nothing about regularization), NOT a logit-scale/architectural perturbation (unlike EXP-070/071), and — critically — NOT an effective-LR change (unlike the momentum-coefficient Ideas 2/3 which scale step ≈1/(1−m) into the closed LR axis). Nesterov vs vanilla heavy-ball changes ONLY the point at which the gradient is evaluated (look-ahead θ+m·v vs θ) at an identical effective step magnitude, so it is the SAFEST possible knob — it cannot overshoot/undershoot the finely-tuned cosine anneal and cannot destabilize training. Honest framing (carried from brainstorm): plateau-mapping with ~nil upside — 19 straight misses, the augmentation lever (only top-1 lever) exhausted including op-set (EXP-072), every other axis closed. Its value is COMPLETING the optimizer-internal axis map with the cleanest-possible failure mode, the correct disciplined NEVER-STOP probe.

## Milestones

### Milestone 1: Code change + smoke
- [ ] train.py L205: `nesterov=True` → `nesterov=False` inside the `optim.SGD(...)` call. Single-flag change; no other edit.
- [ ] Smoke: `uv run python -c "import ast; ast.parse(open('train.py').read())"` OK; construct an `optim.SGD([torch.zeros(1,requires_grad=True)], lr=0.2, momentum=0.9, weight_decay=1e-4, nesterov=False)` and assert `opt.param_groups[0]['nesterov'] is False`; grep-confirm train.py has exactly one `nesterov=` and it reads `False`; `git diff --name-only` == `train.py` only.

### Milestone 2: Launch on idle GPU + early gate
- [ ] Pre-launch `nvidia-smi` idle-GPU check; launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background).
- [ ] Gate (~ep8): dt steady ~8ms (Nesterov vs heavy-ball is the same SGD kernel cost — no throughput change), no NaN, eval test_acc climbing normally (trajectory should track EXP-054 very closely — identical effective step, only the gradient-eval point differs). num_params printed 4,299,866.

### Milestone 3: Completion + verification
- [ ] Run exits 0, prints summary; extract best_test_acc, compare to baseline 96.45 / bar 96.55. Expect ~91 ep, dt 8ms, wall ~585-595s, num_params 4,299,866. Most-likely best_test_acc within ±0.25pp of 96.45 (clean null or small regression).

## Code Changes
- **train.py (L205, single flag)**: `nesterov=True` → `nesterov=False`. Why this tests the hypothesis: with `nesterov=True` the SGD update evaluates the gradient at the look-ahead point (θ + momentum·velocity); `nesterov=False` is vanilla heavy-ball momentum (gradient at θ). Toggling it isolates how much the Nesterov look-ahead contributes to the tuned recipe's convergence/generalization. Risks/edge cases: NONE structural — the flag changes neither tensor shapes, params, FLOPs, the compile graph, the effective step magnitude (~1/(1−m), unchanged since momentum is fixed at 0.9), nor logit scale; it cannot destabilize training (unlike EXP-070) or alter the effective LR (unlike a momentum-coefficient change). The only "risk" is the expected within-noise null or small regression from removing the tuned look-ahead.

## Configuration Changes
- SGD `nesterov`: `True` (tuned default since EXP-000) → `False` (vanilla heavy-ball). MOMENTUM stays 0.9, all else byte-identical to EXP-054 (AugMix-p0.5 default all_ops, GPU Cutout16, cosine peak0.2/warmup0.05/WD1e-4/LS0.1, batch128, seed42, compile reduce-overhead). num_params unchanged (4,299,866) — an optimizer flag, not a model parameter.

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background bash.
- Resources: single idle H20 (pre-check `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`; Σdt=300s budget REQUIRES an uncontended GPU — relaunch on contention).
- Estimated runtime: ~91 epochs, dt ~8ms, Σdt ~300s, wall ~585-595s (< 600s; Nesterov vs heavy-ball is compute-identical — no throughput change vs EXP-054's 593s). Monitor the 600s wall (recipe is wall-tight) but this change adds no cost.
- Log output: `run.log` in project root.
- Tool skill: none (local).

## Abort Criteria
- Loss NaN/inf (essentially impossible — removing the look-ahead is, if anything, marginally MORE conservative, not less stable) or eval test_acc not climbing by ~ep5.
- dt ≥13ms sustained (cudagraph break / contention — NOT expected from an optimizer-flag change; indicates GPU contention): kill, relaunch on a clean idle GPU.
- No output / hung > 3 min.

## Verification Protocol

### Verification Procedure
Baseline = **96.45** (from `exp-index.sh baseline`); bar = **96.55**.
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: after exit, `grep -aE "^best_test_acc:" run.log`; parse the float; PASS iff `>= 96.55`. (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: `grep -aE "^total_seconds:|^num_epochs:|^num_params:|^training_seconds:" run.log`; confirm summary printed, `training_seconds ≈ 300`, `total_seconds < 600`, `num_params == 4,299,866` (UNCHANGED — optimizer flag adds no params), and `grep -ciaE "nan|traceback|error" run.log` == 0. (If total_seconds breaches 600 by a small margin while training_seconds≈300 and metric trustworthy → no-improvement per EXP-061/065 precedent — Nesterov is compute-free so any breach is base-recipe wall variance.)
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == `train.py` only; prepare.py/eval untouched; evaluate() called once/epoch; no new deps; seed 42 unchanged; ran uncontended (steady ~8ms dt).
- Verdict: improvement iff all three pass; no-improvement if a necessary condition fails on a valid run; invalid on scope/dep breach; crash if no metrics / abort.
- Timeout: 12 min wall. Cleanup: `rm run.log` after recording.

### Informational Metrics (Optional)
- `num_epochs`/`num_steps`, `final_test_loss`, `peak_vram_mb`: `grep -aE "^num_epochs:|^num_steps:|^final_test_loss:|^peak_vram_mb:" run.log` — confirm ~91 ep (throughput unchanged) and whether the run tracks EXP-054 (best_test_acc vs 96.45, final_test_loss vs 0.1968). A meaningful deviation in EITHER direction is informative about how load-bearing the Nesterov look-ahead is on this recipe.
