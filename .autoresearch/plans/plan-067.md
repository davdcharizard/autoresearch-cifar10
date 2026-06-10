# Plan EXP-067: BN momentum reduction (0.1 → 0.05)

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-067.md

## Closed-axis check
"Normalization" appears in the project-insights closure list, but that closure covered NORMALIZATION-AS-REGULARIZER mechanisms (BN-as-regularizer, GroupNorm-style, eval-BN recalibration EXP-061) — NOT the BN running-stat ESTIMATOR hyperparameters (momentum, eps), which were never tuned (confirmed in brainstorm-065/066/067 history scan). This is a genuinely-untested knob. It does NOT contradict any High-importance insight (it adds no compute, no layers, no 2nd graph, no augmentation change). It is mechanistically DISTINCT from EXP-061 (clean-data recalib changed the stat DISTRIBUTION; this changes only the EMA window over the SAME augmented distribution). Single static kwarg → cudagraph-safe, wall-safe (critical given 3 recorded wall breaches + the EXP-066 multi-graph penalty).

## Milestones

### Milestone 1: Code change + smoke
- [ ] train.py: add `momentum=0.05` to all four `nn.BatchNorm2d(...)` constructor sites — BasicBlock `bn1` (L71), BasicBlock `bn2` (L75), BasicBlock shortcut BN (L83), stem `bn1` (L103). Update one inline comment to note EXP-067.
- [ ] Smoke: `python -c "import ast; ast.parse(open('train.py').read())"` OK; `git diff --name-only` == train.py only; `grep -c "momentum=0.05" train.py` == 4; confirm no other change.

### Milestone 2: Launch on idle GPU + early gate
- [ ] Pre-launch `nvidia-smi` idle-GPU check; launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background).
- [ ] Gate (~ep8): dt steady ~8ms (no structural change → identical throughput, single graph), no NaN, eval test_acc climbing normally. BN momentum does not affect the forward compute graph, so dt must be unchanged.

### Milestone 3: Completion + verification
- [ ] Run exits 0, prints summary; extract metrics, compare to baseline 96.45 / bar 96.55. Expect ~91 ep (throughput unchanged).

## Code Changes
- **train.py (L71, L75, L83, L103)**: add the `momentum=0.05` kwarg to each `nn.BatchNorm2d(...)` call (default is 0.1). Why: lengthens the running-stat EMA window, lowering eval-time estimation variance over the noisy AugMix operating distribution. Risk/edge case: with cosine-to-0, the final epochs are near-frozen-weight so running stats are already stable — a longer window could fold slightly-staler higher-LR batches into the eval stats → mild regression possible. No structural/throughput/scope/wall risk (momentum is a host-side BN attribute, does not enter the compiled forward's compute).

## Configuration Changes
- BN `momentum`: 0.1 → 0.05 (all 4 sites). Rationale: 0.05 is a standard "longer window" value (doubles the effective EMA horizon); large enough to register an effect, conservative enough not to destabilize the running stats. All else byte-identical to EXP-054 (k=4 WideResNet-20, AugMix-p0.5, GPU Cutout16, cosine peak0.2/warmup0.05/Nesterov/LS0.1/WD1e-4, batch128, seed42, compile reduce-overhead). num_params unchanged (4,299,866).

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background bash.
- Resources: single idle H20 (pre-check nvidia-smi; relaunch on contention per infra-errors — Σdt budget REQUIRES an uncontended GPU).
- Estimated runtime: ~91 epochs, dt ~8ms, Σdt ~300s, wall ~593s (< 600s; same recipe/throughput as EXP-054 — BN momentum is compute-free). Note: AugMix recipe is wall-tight (3 prior breaches on run-to-run variance) — monitor total_seconds but the change adds no wall.
- Log output: `run.log` in project root.
- Tool skill: none (local).

## Abort Criteria
- Loss NaN/inf or eval test_acc not climbing by ep5.
- dt drifts ≫ 8ms (contention — should not happen, no structural change): kill, relaunch on clean idle GPU.
- No output / hung > 3 min.

## Verification Protocol

### Verification Procedure
Baseline = **96.45** (from `exp-index.sh baseline`); bar = **96.55**.
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: after exit, `grep -aE "^best_test_acc:" run.log`; parse float; PASS iff `>= 96.55`. (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: `grep -aE "^total_seconds:|^num_epochs:|^num_params:|^training_seconds:" run.log`; confirm summary printed, `training_seconds ≈ 300`, `total_seconds < 600`, `num_params == 4,299,866`, `grep -ciaE "nan|traceback|error" run.log` == 0. (If total_seconds breaches 600 by a small margin while training_seconds≈300 and metric trustworthy → no-improvement per EXP-061/065 precedent, NOT invalid — but the compute-free change should not add wall.)
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == train.py only; prepare.py/eval untouched; evaluate() once/epoch (loop unchanged); no new deps; seed 42 unchanged; ran uncontended (steady ~8ms dt).
- Verdict: improvement iff all three pass; no-improvement if a necessary condition fails on a valid run; invalid on scope/dep breach; crash if no metrics.
- Timeout: 12 min wall. Cleanup: `rm run.log` after recording.

### Informational Metrics (Optional)
- `num_epochs`/`num_steps`, `final_test_loss`, `peak_vram_mb`: `grep -aE "^num_epochs:|^num_steps:|^final_test_loss:|^peak_vram_mb:" run.log` — confirm ~91 ep (throughput unchanged) and compare final_test_loss to EXP-054's 0.1968 (a longer BN window most directly affects eval-stat variance → watch loss as the sensitive secondary signal).
