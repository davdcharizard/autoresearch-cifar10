# Plan EXP-065: Higher label smoothing (LABEL_SMOOTHING 0.1 → 0.15)

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-065.md

## Closed-axis check
EXP-023 probed LS only DOWNWARD (0.1→0.05, regressed) and concluded "0.1 near-optimal" — but on the OLD TrivialAugment recipe (commit 6c417a4) and WITHOUT testing the upper side. This plan probes the UNTESTED upper side (0.15) on the CURRENT AugMix-p0.5 best, with a recipe-specific mechanism (target softness ↔ AugMix soft-mixed inputs). It is a RETUNE of an existing regularizer (not adding a new penalty, so it does not directly contradict the "adding regularizers hurts" Medium insight — though that insight does lean against a gain). Single-variable, compute-/throughput-neutral, cudagraph-safe. The std-normalization avenue from brainstorm-064 is CONFIRMED CLOSED (goal-learnings Protocol Findings: frozen eval pins std=(1,1,1)) and is NOT pursued.

## Milestones

### Milestone 1: Code change + smoke
- [ ] train.py L27: `LABEL_SMOOTHING = 0.1` → `LABEL_SMOOTHING = 0.15`. Update the inline note to reference EXP-065.
- [ ] Smoke: `ast.parse` OK; `git diff --name-only` == train.py only (one-line change); confirm LABEL_SMOOTHING feeds `F.cross_entropy(..., label_smoothing=LABEL_SMOOTHING)` (train.py L242-244) and nothing else changed.

### Milestone 2: Launch on idle GPU + early gate
- [ ] Pre-launch `nvidia-smi` idle-GPU check; launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background).
- [ ] Gate (~ep8): dt steady ~8ms (no structural change → identical throughput), no NaN, loss descending. NOTE: train loss is the LS-regularized CE so its absolute value will be slightly higher than baseline (more smoothing) — that is expected, not a regression signal; judge by the per-epoch eval test_acc trend.

### Milestone 3: Completion + verification
- [ ] Run exits 0, prints summary; extract metrics, compare to baseline. Expect ~91 ep (throughput unchanged).

## Code Changes
- **train.py (L27)**: `LABEL_SMOOTHING` 0.1 → 0.15. The constant feeds `F.cross_entropy(outputs, targets, label_smoothing=LABEL_SMOOTHING)` in the training loop (L242-244). Why: tests whether softer targets better match the soft, multi-chain-mixed AugMix inputs (50% of the batch) on the current best recipe — an interaction EXP-023 (TrivialAugment recipe, lower-direction only) did not probe. Risk/edge case: more label smoothing is more regularization; at the short 300s/91-ep budget the recipe is convergence-bound (project-insights Medium), so over-smoothing could mildly under-train → small regression. No structural/throughput/scope risk (LS is a host-side scalar arg to cross_entropy, outside the compiled forward → cudagraph-safe).

## Configuration Changes
- `LABEL_SMOOTHING`: 0.1 → 0.15. Rationale: 0.15 is a modest step up (the standard LS range is 0.05–0.2); large enough to register an effect, small enough not to grossly over-smooth. All else byte-identical to EXP-054 (k=4 WideResNet-20, AugMix-p0.5, Cutout16, cosine peak0.2/warmup0.05/Nesterov/WD1e-4, batch128, seed42, compile reduce-overhead). num_params unchanged (4,299,866).

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background bash.
- Resources: single idle H20 (pre-check nvidia-smi; relaunch on contention per infra-errors).
- Estimated runtime: ~91 epochs, dt ~8ms, Σdt ~300s, wall ~593s (< 600s; same recipe as EXP-054 — LS change is compute-free).
- Log output: `run.log` in project root.
- Tool skill: none (local).

## Abort Criteria
- Loss NaN/inf or not descending by ep5 (judge eval test_acc, not the LS-inflated train loss).
- dt drifts ≫ 8ms (contention — should not happen, no structural change): kill, relaunch on clean idle GPU.
- No output / hung > 3 min.

## Verification Protocol

### Verification Procedure
Baseline = **96.45** (from `exp-index.sh baseline`); bar = **96.55**.
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: after exit, `grep -aE "^best_test_acc:" run.log`; parse float; PASS iff `>= 96.55`. (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: `grep -aE "^total_seconds:|^num_epochs:|^num_params:" run.log`; confirm summary printed, `total_seconds < 600`, total wall < 10 min, `num_params == 4,299,866`, `grep -ciaE "nan|traceback|error" run.log` == 0.
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == train.py only; prepare.py/eval untouched; evaluate() once/epoch (loop unchanged); no new deps; seed 42 unchanged; ran uncontended (steady ~8ms dt).
- Verdict: improvement iff all three pass; no-improvement if a necessary condition fails on a valid run; invalid on scope/dep breach; crash if no metrics.
- Timeout: 10 min wall. Cleanup: `rm run.log` after recording.

### Informational Metrics (Optional)
- peak_vram_mb, num_epochs/num_steps, final_test_loss: `grep -aE "^peak_vram_mb:|^num_epochs:|^num_steps:|^final_test_loss:" run.log` — confirm ~91 ep (throughput unchanged). NOTE: final_test_loss is eval CE WITHOUT label smoothing (Eval uses plain `F.cross_entropy`), so it IS comparable to EXP-054's 0.1968 (higher train-time LS does not directly inflate the eval-CE metric — a fair comparison).
