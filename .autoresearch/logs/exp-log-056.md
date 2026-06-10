# EXP-056: GPU-batched diverse augmentation (affine + photometric), full-coverage — the throughput unlock

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-056.md
- **Plan**: plans/plan-056.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-056
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Three edits to train.py: (1) added `gpu_augment(x)` after `cutout_batch` — vectorized per-sample random affine (rotation ±12°, shear ±0.1, anisotropic scale [0.9,1.1]) via `F.affine_grid`+`F.grid_sample` (bilinear, reflection pad) + photometric (brightness ±0.1 additive, contrast [0.85,1.15] about per-image mean), all seeded/batched/no-CPU-sync, returns channels_last; (2) wired `inputs = gpu_augment(inputs)` into the train loop immediately before `cutout_batch`; (3) removed the CPU `RandomApply([AugMix()], p=0.5)` from `train_tf` (CPU pipeline now crop+flip+ToTensor+Normalize). This moves the augmentation-diversity lever off the wall-limited 8-worker CPU dataloader onto the idle GPU at full coverage. Smoke tests passed: AST OK, scope=train.py only, gpu_augment output (128,3,32,32) finite float32 channels_last, **~0.52ms/batch standalone** (cheap → small dt premium expected).

### Surprises & Discoveries
gpu_augment standalone cost is only ~0.52ms/batch on the H20 — well under the ~1ms budgeted, so the epoch-wall risk looks low (expect dt 8→~8.5ms, epochs ~86-88). The real test is whether full-coverage GPU geometric+photometric diversity matches/beats the 50%-subset CPU AugMix it replaces.

### Decisions
Included photometric (brightness/contrast) alongside affine in v1 (not affine-only): both are mathematically valid on the std=(1,1,1) normalized tensor (brightness=additive shift, contrast=scale about per-image mean) and add diversity for ~free, improving win probability. Skipped posterize/solarize/equalize/saturation (fiddly/incorrect on normalized data) — if v1 is a near-miss, those are the next increment. Applied gpu_augment BEFORE cutout (geometric/photometric first, occlusion last — mirrors the old CPU-AugMix→GPU-Cutout order).

## Experimental Adjustments

- **Run 1 ABORTED (GPU contention) → Run 2 relaunched on GPU 1**: Run 1 (GPU 0) hit transient contention — another user's process (PID 194920, 1.4GB, 85% util) landed on GPU 0 after launch, and CPU load avg ~8 (competing `pdflatex`) starved the dataloader workers → at step 2450, wall 384s vs only 36s of Σdt consumed (wall/Σdt ≈ 10× vs the healthy ~1.5-2×), projecting ~3200s total wall ≫ 600s. NOT a code issue (the lightened CPU pipeline + cheap 0.52ms gpu_augment are fine) — the known shared-node hazard (goal-learnings: fair dt-budgeted runs require an idle GPU). Aborted (TaskStop), relaunched identical config on idle GPU 1. (ref: Run 1 nvidia-smi PID 194920 + uptime load 7.5)

## Run Log

### Run 1 (ABORTED — GPU contention, not a code/research failure)

Metadata:
- **Job ID**: background bash ID bpp7zuael (local) — STOPPED
- **Log file(s)**: run.log (overwritten by Run 2)
- **Status**: aborted (infra contention)
- **Started/Ended**: 2026-06-09 (aborted at ~step 2450)

Observations:
- GPU 0 contended post-launch (PID 194920, 1.4GB, 85% util) + CPU load ~8 → wall/Σdt ≈ 10× → projected ~3200s wall. Aborted per the wall-overrun abort criterion. dt itself oscillated 9-15ms (contention), gpu_augment functioned (loss descending normally, ep6 test_acc 79.31%). (source: run.log Run 1; nvidia-smi)

### Run 2

Metadata:
- **Job ID**: background bash ID bqbi38wco (local)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09
- **Ended**: 2026-06-09 (exit 0, 390.5s wall)

Description:
- Identical config to Run 1, relaunched on idle GPU 1 (both GPUs idle at relaunch; contention cleared). Full-coverage GPU augmentation (gpu_augment: affine+photometric) + GPU Cutout, CPU lightened to crop+flip. Tests whether moving the diversity lever off the starved CPU dataloader onto the idle GPU (EXP-003 precedent) at 100% coverage beats the 50%-subset CPU AugMix (96.45). Expect dt 8→~8.5ms, epochs ~86, wall ~350-420s. Early gate: abort if dt>~11ms (epochs<~76) OR wall/Σdt ≫ 2 (contention recurred). Bar = 96.55.

Observations:
- EARLY GATE PASSED on GPU 1 (no contention). Window at step 3650 / 50s wall / 11.9% budget: mean **dt 9.9ms/step** (≈+1.9ms gpu_augment premium over the 8ms baseline), projected **~78 epochs** (above the 76 abort floor; below baseline's 91 — the GPU-aug epoch cost), wall/Σdt = 1.39× (healthy vs Run 1's ~10× contention), projected total wall ~400s (comfortable < 600s). dt steady 9-10ms, loss descending normally, no NaN. (source: run.log Run 2; ps etimes)
- The experiment's crux: full-coverage GPU geometric+photometric diversity must offset the ~13-epoch loss (78 vs 91) to clear 96.55.

Key Metrics:
- best_test_acc: **95.39%** — **REGRESSION**, −1.06pp vs baseline 96.45, far below bar 96.55 (below even pre-AugMix 96.00). (source: run.log Run 2 summary)
- final_test_acc: 95.27%; final_test_loss: **0.2240** (much worse than EXP-054's 0.1968 — model fighting an over-distorted train distribution). (source: run.log)
- total_seconds: 390.5s (wall comfortable, dt-bound as predicted — CPU no longer starves). training_seconds 300.0 (Σdt budget hit). num_epochs: 84 (vs baseline 91 — the GPU-aug epoch premium); num_steps 32,621; num_params 4,299,866; peak_vram 452.9 MB (unchanged). (source: run.log)
- dt dist: 635×9ms, 13×10ms, few 11-13ms, 1×47ms (steady ~9ms = +1ms gpu_augment premium over 8ms baseline; GPU-aug infrastructure is cheap & works). (source: run.log)
- Ran uncontended (GPU 1 solo throughout; GPU 0's contention was an unrelated job). (source: nvidia-smi)

## Verification Results

### Conditions Checked
- **Cond 1 — best_test_acc ≥ 96.55 (baseline 96.45 + 0.1)**: 95.39% → **FAIL** (−1.06pp vs baseline; −1.16pp below bar). Verdict → no-improvement. (source: run.log summary)
- **Cond 2 — clean completion within budget**: NOT EVALUATED (skipped — aborted after Cond 1 failure). [Informationally: would PASS — summary printed, total_seconds 390.5 < 600, num_params 4,299,866, no NaN/traceback (grep 0).]
- **Cond 3 — no hard-constraint violations**: NOT EVALUATED (skipped). [Informationally: would PASS — `git diff --name-only` = train.py only; eval/prepare untouched; affine_grid/grid_sample core torch (no new dep); seed 42 unchanged; evaluate() once/epoch.]
- **Cond 1 (necessary) FAILED → Outcome: completed (clean run); verdict no-improvement (regression).**

### Informational Metrics
- delta vs baseline 96.45: **−1.06pp** (large regression). final_test_loss 0.2240 ≫ EXP-054's 0.1968. num_epochs 84 (−7 vs 91, the GPU-aug Σdt premium). The epoch loss alone (~7) cannot explain −1.06pp — the dominant cause is the augmentation POLICY: full-coverage, every-step, STACKED affine(rotate+shear+scale)+photometric(brightness+contrast) — 5 simultaneous distortions per image with NO clean-image convex mixing — is far harsher than TrivialAugment (1 op/image → 96.22) or AugMix (clean-mixed chains on a subset → 96.45). The high test loss is the signature of an over-distorted training distribution. The GPU-augmentation INFRASTRUCTURE is validated (cheap ~9ms dt, correct, uncontended); the policy is the problem.

## Errors & Dead Ends

## Human Notes

> {none — autopilot}
