# Plan EXP-053: AugMix(w2,d1) severity 3→6 — push op magnitude on the new winner
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-053.md

Baseline = **96.34%** (EXP-052, 292a9e2); bar = baseline + 0.1 = **96.44%**. Single-variable push on the just-validated augmentation-diversity lever: raise AugMix `severity` 3→6 on the `mixture_width=2, chain_depth=1` winner, keeping Cutout and everything else fixed. severity is the only diversity dial that is CPU-neutral (probed feasible ~12ms/batch) and preserves all-image coverage.

## Milestones

### Milestone 1: Code change implemented and smoke-tested
- [ ] In `train_tf`, change `transforms.AugMix(mixture_width=2, chain_depth=1)` → `transforms.AugMix(mixture_width=2, chain_depth=1, severity=6)`.
- [ ] Smoke: `uv run python -c "import ast; ast.parse(open('train.py').read())"` passes; `git diff` = the single `severity=6` addition (+ comment); `from torchvision.transforms import AugMix` resolves (no new dep).
- [ ] Smoke: instantiate train_set, pull ~5 augmented samples (AugMix runs at severity=6 without error, shape (3,32,32) after ToTensor); num_params unchanged (4,299,866).

### Milestone 2: Experiment running and feasibility confirmed
- [ ] Launch `uv run train.py > run.log 2>&1` on the idle GPU; confirm run.log is written.
- [ ] Feasibility (lower risk than EXP-052: severity is CPU-neutral, w2,d1,sev5 probed at 12.1ms/batch ≈ w2,d1 12.6ms): after ~60-90s wall, take a real-load eval-inclusive wall measurement (steps/wall over a window). Project total wall = effective ms/step × est-total-steps + ~30s compile. If projected > 585s → abort (should NOT happen — severity adds no CPU cost).
- [ ] Early signal: dt steady ~8ms (GPU step unchanged), ep1 test_acc normal (~45%), no NaN.

### Milestone 3: Run completes and is verified
- [ ] Run prints the summary; `total_seconds < 600`.
- [ ] Extract best_test_acc, num_epochs, dt dist, total_seconds, peak_vram_mb; compare to bar 96.44.

## Code Changes
- **train.py** (one keyword, in `train_tf`): `transforms.AugMix(mixture_width=2, chain_depth=1)` → `transforms.AugMix(mixture_width=2, chain_depth=1, severity=6)`.
  - **Why this tests the hypothesis**: severity scales per-op magnitude (default 3, range 1-10). Raising it to 6 produces stronger per-op distortions → a more spread-out augmented distribution → more diversity in the strength dimension, the validated lever (EXP-012/052). w2,d1 chain structure and all-image coverage are unchanged → clean single-variable test of magnitude.
  - **Risks / edge cases**: (a) magnitude-knob interior-optimum (Cutout strength EXP-013/021 was interior-optimal both directions) → severity=3 may already be near-optimal → within-noise null; (b) over-augmentation at sev=6 → mild regression (AugMix's clean-image convex mix bounds the shift, mitigating this). No feasibility/crash risk — severity is CPU-neutral, all-coverage. Update the inline comment to note severity=6.

## Configuration Changes
- Augmentation: AugMix `severity` 3 (default) → 6. No model/optimizer/schedule/seed/batch/compile/coverage changes. num_params unchanged (4,299,866).

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background (`run_in_background: true`).
- Resources: single NVIDIA H20. Shared node GPUs 0/1; check `nvidia-smi`, launch on idle GPU.
- Estimated runtime: ~300s Σdt budget; wall ~550-575s (≈ EXP-052's 571.9s, severity is CPU-neutral). Target < 600s, monitored at Milestone 2.
- Log output: stdout/stderr → `run.log` in project root.
- Tool skill: none (local run).

## Abort Criteria
- Wall-clock projection > ~585s at the Milestone-2 check (not expected — CPU-neutral vs EXP-052's feasible 571.9s).
- Loss NaN/inf or diverging.
- dt rises well above 8ms and stays (would indicate a GPU-side issue — not expected, augmentation is CPU-side).
- No output / log not advancing > 3 min after launch.
- Total wall-clock actually reaching ~590s without a summary → kill (constraint breach).

## Verification Protocol

### Verification Procedure
Run after completion; stop at the first failed necessary condition.

1. **Baseline**: `bash .../exp-index.sh baseline experiment-indices/improve-cifar10-test-accuracy.tsv` → 96.34, bar **96.44**.
2. **Necessary condition 1 — `best_test_acc >= 96.44`**: `grep -aE "^best_test_acc:" run.log` → parse float. PASS iff `>= 96.44`; else no-improvement. (Absent ⇒ crash → `tail -n 50 run.log`.)
3. **Necessary condition 2 — clean completion within budget**: `grep -aE "^best_test_acc:|^total_seconds:|^num_params:" run.log` → summary printed, `total_seconds < 600`, `num_params == 4,299,866`. No NaN/traceback.
4. **Necessary condition 3 — no hard-constraint violations**: `git diff --name-only` = `train.py` only; prepare.py/eval untouched; `evaluate()` once/epoch (loop unchanged); no new deps (AugMix is torchvision 0.24.1); seed 42 unchanged; no seed hacking.
5. Remove `run.log` before the next experiment.

### Informational Metrics (Optional)
- best_test_acc / delta vs 96.34: `grep -aE "^best_test_acc:" run.log`.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — expect ~91 (CPU-neutral, same as EXP-052).
- total_seconds (wall): `grep -aE "^total_seconds:" run.log` — expect ~570s.
- dt distribution: `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c` — expect steady 8ms.
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — compare to EXP-052's 0.2010 / baseline 0.195.
