# Plan EXP-054: Intermittent full-strength AugMix via RandomApply(p=0.5) — push chain-count diversity
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-054.md

Baseline = **96.34%** (EXP-052, 292a9e2); bar = baseline + 0.1 = **96.44%**. Push the live augmentation-diversity lever (chain COUNT, per EXP-053) by delivering the literature-validated full AugMix (mixture_width=3, chain_depth=-1) to ~50% of images via `RandomApply` — the feasible way to expose training to genuine 3-chain diversity under the 600s wall (uniform w3 is infeasible, ~792s).

## Milestones

### Milestone 1: Code change implemented and smoke-tested
- [ ] In `train_tf`, replace `transforms.AugMix(mixture_width=2, chain_depth=1)` with `transforms.RandomApply([transforms.AugMix()], p=0.5)` (full default AugMix on ~50% of images; the rest get only RandomCrop+Flip, with GPU Cutout still applied in the train loop).
- [ ] Smoke: `uv run python -c "import ast; ast.parse(open('train.py').read())"` passes; `git diff` = the single transform line (+ comment); imports resolve (no new dep).
- [ ] Smoke: instantiate train_set, pull ~10 augmented samples (RandomApply+AugMix runs without error, shape (3,32,32) after ToTensor); num_params unchanged (4,299,866).

### Milestone 2: Experiment running and FEASIBILITY (wall-clock) confirmed
- [ ] Launch `uv run train.py > run.log 2>&1` on the idle GPU; confirm run.log is written.
- [ ] **Feasibility check** (probed isolated: p=0.5 = 12.9ms/batch → ~585s, tight; calibration: EXP-052 w2,d1 isolated 12.6ms → actual 571.9s, so the isolated probe is well-calibrated here). After ~60-90s wall, take a real-load eval-inclusive wall measurement (steps/wall over a window). Project total wall = effective ms/step × est-total-steps + ~30s compile. **If projected > 585s → ABORT and go to contingency (Run 2 at p=0.4, probed 11.1ms → ~517s, safe).**
- [ ] Early signal: dt steady ~8ms (GPU step unchanged — AugMix is CPU-side), ep1 test_acc normal (~45%), no NaN.

### Milestone 3: Run completes and is verified
- [ ] Run prints the summary; `total_seconds < 600`.
- [ ] Extract best_test_acc, num_epochs, dt dist, total_seconds, peak_vram_mb; compare to bar 96.44.

## Code Changes
- **train.py** (one line, in `train_tf`): `transforms.AugMix(mixture_width=2, chain_depth=1)` → `transforms.RandomApply([transforms.AugMix()], p=0.5)`.
  - **Why this tests the hypothesis**: EXP-053 showed the diversity lever is chain COUNT, not magnitude; EXP-052 showed AugMix mixing works — but the rich w3 config (3 chains) is wall-infeasible uniformly. RandomApply applies the FULL w3 AugMix to a random ~50% of images at an average CPU cost (~12.9ms/batch) that fits the wall, exposing training to genuine 3-chain diversity. Tests whether richer-but-intermittent diversity beats uniform-but-shallow (w2,d1).
  - **Risks / edge cases**: (a) **coverage reduction** — ~half the images get no photometric/geometric aug (still get crop+flip+Cutout); "aug on every image" has been the working regime → could wash the diversity gain (within-noise null / mild regression). (b) **wall-clock** — p=0.5 is tight (~585s est); per-batch cost varies (random # augmented samples) → wall variance; gated by Milestone-2 check + p=0.4 contingency. (c) confounded (coverage↓ and diversity↑ co-move) — documented for analysis.

### Contingency (Run 2, only if Run 1 breaches wall feasibility)
- If Run 1 projects wall > 585s at the Milestone-2 check: re-run at `RandomApply([AugMix()], p=0.4)` (probed 11.1ms/batch → ~517s, comfortable margin). Same mechanism, 40% coverage. If that also breaches, record infeasibility and proceed to analysis.

## Configuration Changes
- Augmentation: `AugMix(w2,d1)` (all images) → `RandomApply([AugMix() default w3,d-1], p=0.5)` (full AugMix on ~50%). No model/optimizer/schedule/seed/batch/compile changes. num_params unchanged.

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background (`run_in_background: true`).
- Resources: single NVIDIA H20. Shared node GPUs 0/1; check `nvidia-smi`, launch on idle GPU.
- Estimated runtime: ~300s Σdt budget; wall ~585s at p=0.5 (target < 600s, monitored at Milestone 2; p=0.4 fallback ~517s).
- Log output: stdout/stderr → `run.log` in project root.
- Tool skill: none (local run).

## Abort Criteria
- **Wall-clock projection > 585s** at the Milestone-2 real-load check → abort, go to p=0.4 contingency.
- Loss NaN/inf or diverging.
- dt rises well above 8ms and stays (GPU-side issue — not expected, augmentation is CPU-side).
- No output / log not advancing > 3 min after launch.
- Total wall-clock actually reaching ~590s without a summary → kill (constraint breach).

## Verification Protocol

### Verification Procedure
Run after completion; stop at the first failed necessary condition.

1. **Baseline**: `bash .../exp-index.sh baseline experiment-indices/improve-cifar10-test-accuracy.tsv` → 96.34, bar **96.44**.
2. **Necessary condition 1 — `best_test_acc >= 96.44`**: `grep -aE "^best_test_acc:" run.log` → parse float. PASS iff `>= 96.44`; else no-improvement. (Absent ⇒ crash → `tail -n 50 run.log`.)
3. **Necessary condition 2 — clean completion within budget**: `grep -aE "^best_test_acc:|^total_seconds:|^num_params:" run.log` → summary printed, **`total_seconds < 600`** (binding here), `num_params == 4,299,866`. No NaN/traceback.
4. **Necessary condition 3 — no hard-constraint violations**: `git diff --name-only` = `train.py` only; prepare.py/eval untouched; `evaluate()` once/epoch (loop unchanged); no new deps (AugMix/RandomApply are torchvision); seed 42 unchanged; no seed hacking.
5. Remove `run.log` before the next experiment.

### Informational Metrics (Optional)
- best_test_acc / delta vs 96.34: `grep -aE "^best_test_acc:" run.log`.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — expect ~91 (Σdt budget unaffected).
- total_seconds (wall): `grep -aE "^total_seconds:" run.log` — feasibility-critical (~585s expected).
- dt distribution: `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c` — expect steady 8ms.
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — compare to EXP-052's 0.2010 / baseline 0.195.
