# Plan EXP-052: AugMix replacing TrivialAugmentWide (strongest diverse augmentation)
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-052.md

Baseline = **96.22%** (EXP-012, 6c417a4); bar = baseline + 0.1 = **96.32%**. Tests the only untried variant of the only lever that ever broke a plateau here (strong diverse augmentation, EXP-012): swap the single-chain TrivialAugmentWide for AugMix (mix-of-3-chains, torchvision, no new dep), keeping Cutout. Endorsed by the High-Importance "test the strongest diverse augmentation before closing the axis" insight.

## Milestones

### Milestone 1: Code change implemented and smoke-tested
- [ ] In `train_tf`, replace `transforms.TrivialAugmentWide()` with `transforms.AugMix()` (torchvision defaults: severity=3, mixture_width=3, chain_depth=-1, alpha=1.0). Same pipeline position (PIL stage, before `ToTensor`).
- [ ] Smoke check: `python -c "import ast; ast.parse(open('train.py').read())"` passes; `git diff` = the single transform line; `from torchvision.transforms import AugMix` resolves (no new dep — verified torchvision 0.24.1).
- [ ] Smoke check: instantiate the train_set + pull ~5 augmented samples to confirm AugMix runs without error on CIFAR PIL images and yields correct tensor shape after ToTensor; num_params unchanged (4,299,866 — augmentation doesn't touch the model).

### Milestone 2: Experiment running and FEASIBILITY (wall-clock) confirmed
- [ ] Launch `uv run train.py > run.log 2>&1` on the idle GPU; confirm run.log is written.
- [ ] **Critical feasibility check** (AugMix is ~3× TA's CPU cost): after ~60-90s wall, read `pct_done` (= total_training_time/300) and actual wall elapsed. The Σdt budget fills regardless, but if the dataloader can't keep up the GPU starves and WALL-clock balloons. Project final wall ≈ wall_elapsed / (pct_done). If projected wall > ~560s → the run will breach the 600s limit → ABORT and go to the contingency (Run 2, lighter AugMix).
- [ ] Early signal: dt steady ~8ms (GPU step unchanged — AugMix is CPU-side), ep1 test_acc normal, no NaN.

### Milestone 3: Run completes and is verified
- [ ] Run prints the summary; `total_seconds < 600`.
- [ ] Extract `best_test_acc`, `num_epochs`, dt dist, `total_seconds`, `peak_vram_mb`; compare to bar 96.32.

## Code Changes
- **train.py** (one line, in `train_tf`): `transforms.TrivialAugmentWide()` → `transforms.AugMix()`.
  - **Why this tests the hypothesis**: AugMix superimposes 3 independently-sampled augmentation chains (mixed with the original via random convex weights), producing strictly more diverse augmented samples than the single-chain TrivialAugment — directly increasing augmentation diversity, the proven-effective intervention class (EXP-012). Cutout (occlusion, orthogonal) and everything else stay fixed → clean single-policy swap (mirrors EXP-014's TA↔RandAugment test).
  - **Risks / edge cases**: (a) **CPU cost / wall-clock** — AugMix's 3-chain cost is heavier; if it starves the GPU, wall-clock (not Σdt) balloons toward the 600s hard limit. Mitigated by the Milestone-2 feasibility check + contingency. (b) AugMix expects PIL/uint8 input at this pipeline position (before ToTensor) — same as TrivialAugmentWide, so the position is correct. (c) **Policy-saturation null** (EXP-014): may land within ±0.25pp.

### Contingency (Run 2, only if Run 1 breaches wall-clock feasibility)
- If Run 1 projects wall > ~560s: re-run with a lighter AugMix — `transforms.AugMix(mixture_width=2, chain_depth=2)` — to roughly halve the CPU cost while preserving the mix-of-chains diversity. If that still breaches feasibility, record as an infeasibility finding (AugMix too CPU-expensive at this budget) and proceed to analysis.

## Configuration Changes
- Augmentation: `TrivialAugmentWide()` → `AugMix()` (defaults). No model/optimizer/schedule/seed/batch/compile changes. num_params unchanged.

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background (`run_in_background: true`).
- Resources: single NVIDIA H20. Shared node GPUs 0/1; check `nvidia-smi`, launch on idle GPU.
- Estimated runtime: ~300s GPU-time; wall-clock UNCERTAIN (AugMix CPU cost) — target < 600s, monitored at Milestone 2.
- Log output: stdout/stderr → `run.log` in project root.
- Tool skill: none (local run).

## Abort Criteria
- **Wall-clock projection > ~560s** at the Milestone-2 check (CPU-bound dataloader starving the GPU) → abort, go to contingency.
- Loss NaN/inf or diverging.
- dt rises well above 8ms and stays (would indicate a GPU-side issue — not expected since AugMix is CPU-side).
- No output / log not advancing > 3 min after launch.
- Total wall-clock actually reaching ~590s without a summary → kill (constraint breach).

## Verification Protocol

### Verification Procedure
Run after completion; stop at the first failed necessary condition.

1. **Baseline**: `bash .../exp-index.sh baseline experiment-indices/improve-cifar10-test-accuracy.tsv` → 96.22, bar **96.32**.
2. **Necessary condition 1 — `best_test_acc >= 96.32`**: `grep -aE "^best_test_acc:" run.log` → parse float. PASS iff `>= 96.32`; else no-improvement. (Absent ⇒ crash → `tail -n 50 run.log`.)
3. **Necessary condition 2 — clean completion within budget**: `grep -aE "^best_test_acc:|^total_seconds:|^num_params:" run.log` → summary printed, **`total_seconds < 600`** (the binding constraint for this experiment), `num_params == 4,299,866`. No NaN/traceback.
4. **Necessary condition 3 — no hard-constraint violations**: `git diff --name-only` = `train.py` only; prepare.py/eval untouched; `evaluate()` once/epoch (loop unchanged); no new deps (AugMix is in the installed torchvision 0.24.1); seed 42 unchanged; no seed hacking.
5. Remove `run.log` before the next experiment.

### Informational Metrics (Optional)
- best_test_acc / delta vs 96.22: `grep -aE "^best_test_acc:" run.log`.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — if AugMix is CPU-bound, epochs may drop (fewer steps fit before wall/Σdt limits).
- total_seconds (wall): `grep -aE "^total_seconds:" run.log` — the feasibility-critical metric (AugMix CPU cost).
- dt distribution: `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c` — expect steady 8ms (GPU step unchanged).
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — compare to baseline 0.195.
