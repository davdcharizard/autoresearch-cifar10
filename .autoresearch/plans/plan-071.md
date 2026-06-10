# Plan EXP-071: BatchNorm eps 1e-5 → 1e-3
- **Created**: 2026-06-10
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-071.md

## Closed-axis check
This is an explicitly-flagged UNTESTED cell, not a closed-axis retry. goal-learnings line 181 (BN-estimator axis): "Only **BN eps (untested)** and momentum-UP (near-certain null) remain — effectively closed." BN momentum-DOWN was tested (EXP-067, −0.30pp) and clean-BN recalib (EXP-061, −1.6pp); eps is the one untouched BN knob. It does NOT contradict a High-importance insight: eps is NOT a strong regularizer (so it dodges the convergence-bound "adding regularizers fails" closure, project-insights line 68) and NOT a logit-scale or effective-LR change (so it dodges the EXP-070 destabilization and the momentum/LR closures). It is the SAFEST possible knob — strictly the BN numerical floor — optimization-stable, dt-neutral, param-neutral. Honest framing (carried from brainstorm): this is plateau-mapping with near-zero upside; its value is completing the BN-estimator axis map with a benign clean-null failure mode, the correct disciplined NEVER-STOP probe given every positive-EV lever is closed (16 straight misses, 96.45 = k=4/300s ceiling).

## Milestones

### Milestone 1: Code change + smoke
- [ ] train.py: add a constant near the other hyperparameters (after L28 `CUTOUT_SIZE`): `BN_EPS = 1e-3  # BatchNorm numerical floor (default 1e-5); EXP-071 probe of the last untested BN-estimator knob`.
- [ ] train.py: pass `eps=BN_EPS` to ALL FOUR `nn.BatchNorm2d(...)` construction sites: BasicBlock `bn1` (L71), `bn2` (L75), the downsample-shortcut BN (L83), and the ResNet stem `bn1` (L103). (All four must change — a partial change would mix eps values inconsistently.)
- [ ] Smoke: `uv run python -c "import ast; ast.parse(open('train.py').read())"` OK; instantiate `ResNet(3,10,width_mult=4)`, forward a `(2,3,32,32)` tensor → assert `(2,10)`, print param count (expect UNCHANGED 4,299,866 — eps adds no params), and assert every BatchNorm2d module has `.eps == 1e-3` (e.g. `all(m.eps==1e-3 for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))`); `git diff --name-only` == `train.py` only.

### Milestone 2: Launch on idle GPU + early gate
- [ ] Pre-launch `nvidia-smi` idle-GPU check; launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background).
- [ ] Gate (~ep8): dt steady ~8ms (eps is a scalar in the BN epilogue — no graph/throughput change), no NaN, eval test_acc climbing normally (the trajectory should track EXP-054 almost exactly — eps is near-inert). num_params printed 4,299,866.

### Milestone 3: Completion + verification
- [ ] Run exits 0, prints summary; extract best_test_acc, compare to baseline 96.45 / bar 96.55. Expect ~91 ep, dt 8ms, wall ~590s, num_params 4,299,866, best_test_acc within ±0.25pp of 96.45.

## Code Changes
- **train.py (new constant after L28)**: `BN_EPS = 1e-3`. Why: a single named knob for the eps probe (vs default 1e-5), applied consistently to all BN layers.
- **train.py (4 BatchNorm2d sites: L71, L75, L83, L103)**: `nn.BatchNorm2d(C)` → `nn.BatchNorm2d(C, eps=BN_EPS)`. Why this tests the hypothesis: eps is the numerical floor in `(x−μ)/sqrt(σ²+eps)`; raising it 1e-5→1e-3 mildly dampens low-variance channels (a tiny smoothing). It is the last untested BN-estimator knob; the test is whether this floor is inert (near-certain null) on this well-conditioned net. Risks/edge cases: NONE structural — eps cannot change shapes, params, FLOPs, the compile graph, or logit scale; it cannot destabilize training (unlike EXP-070) or alter the effective LR (unlike a momentum change). The only "risk" is the expected within-noise null. Must ensure all 4 sites change (a mixed 1e-5/1e-3 net would be an inconsistent, uninterpretable config — the Milestone 1 smoke-check asserts all BN modules have eps==1e-3).

## Configuration Changes
- BN `eps`: 1e-5 (PyTorch default, EXP-054) → 1e-3 (a standard "high-eps" value, e.g. TensorFlow's default). All else byte-identical to EXP-054 (AugMix-p0.5, GPU Cutout16, cosine peak0.2/warmup0.05/Nesterov m0.9/WD1e-4/LS0.1, batch128, seed42, compile reduce-overhead). num_params unchanged (4,299,866) — eps is a buffer scalar, not a parameter.

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background bash.
- Resources: single idle H20 (pre-check `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`; Σdt=300s budget REQUIRES an uncontended GPU — relaunch on contention).
- Estimated runtime: ~91 epochs, dt ~8ms, Σdt ~300s, wall ~590s (< 600s; eps is compute-free — no throughput change vs EXP-054's 593s). Monitor the 600s wall (recipe is wall-tight) but this change adds no cost.
- Log output: `run.log` in project root.
- Tool skill: none (local).

## Abort Criteria
- Loss NaN/inf (would be shocking from an eps increase — a LARGER eps is strictly MORE numerically stable, so NaN is essentially impossible) or eval test_acc not climbing by ~ep5.
- dt ≥13ms sustained (cudagraph break / contention — NOT expected from a scalar eps change; indicates GPU contention): kill, relaunch on a clean idle GPU.
- No output / hung > 3 min.

## Verification Protocol

### Verification Procedure
Baseline = **96.45** (from `exp-index.sh baseline`); bar = **96.55**.
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: after exit, `grep -aE "^best_test_acc:" run.log`; parse the float; PASS iff `>= 96.55`. (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: `grep -aE "^total_seconds:|^num_epochs:|^num_params:|^training_seconds:" run.log`; confirm summary printed, `training_seconds ≈ 300`, `total_seconds < 600`, `num_params == 4,299,866` (UNCHANGED — eps adds no params), and `grep -ciaE "nan|traceback|error" run.log` == 0. (If total_seconds breaches 600 by a small margin while training_seconds≈300 and metric trustworthy → no-improvement per EXP-061/065 precedent — eps is compute-free so any breach is base-recipe wall variance.)
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == `train.py` only; prepare.py/eval untouched; evaluate() called once/epoch; no new deps; seed 42 unchanged; ran uncontended (steady ~8ms dt).
- Verdict: improvement iff all three pass; no-improvement if a necessary condition fails on a valid run; invalid on scope/dep breach; crash if no metrics / abort.
- Timeout: 12 min wall. Cleanup: `rm run.log` after recording.

### Informational Metrics (Optional)
- `num_epochs`/`num_steps`, `final_test_loss`, `peak_vram_mb`: `grep -aE "^num_epochs:|^num_steps:|^final_test_loss:|^peak_vram_mb:" run.log` — confirm ~91 ep (throughput unchanged) and that the run tracks EXP-054 (best_test_acc within ±0.25pp of 96.45, final_test_loss ≈ 0.1968). A meaningful deviation in EITHER direction would itself be informative (it would mean the BN floor is NOT inert here, contrary to expectation).
