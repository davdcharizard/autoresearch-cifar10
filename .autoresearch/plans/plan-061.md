# Plan EXP-061: Clean-data BN recalibration before eval in the final epochs

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-061.md

## Why this is a NEW lever (not a closed axis)
project-insights High Importance says the next gain must come from a NEW lever (NOT aug/EMA/SWA/schedule/capacity/normalization/optimizer/head). This qualifies:
- It is NOT a regularizer or training-dynamics change. The model, optimizer, schedule, augmentation, and per-step training are byte-identical to EXP-054. The ONLY change is how the BN *running statistics* (used solely at inference) are computed before the eval forward pass.
- "Normalization closed" = GhostBN (normalization-AS-REGULARIZER, EXP-047); "EMA/SWA closed" = weight-AVERAGING (EXP-006/019/020). Recomputing BN running mean/var on clean data to match the eval input distribution is neither — it corrects an inference-time statistics mismatch, an axis no prior experiment touched. goal-learnings' Precise-BN dismissal was convergence-framed (~35k converged updates), NOT about the Cutout/AugMix train→eval DISTRIBUTION shift this targets.

## Milestones

### Milestone 1: Code implemented + smoke-checked
- [ ] Add `recalibrate_bn(model, loader, device, n_batches)` helper: snapshot each BatchNorm2d's (running_mean, running_var, num_batches_tracked, momentum); `reset_running_stats()` + set `momentum=None` (cumulative); `model.train()`; forward-only (`torch.no_grad()` + bf16 autocast) over `n_batches` clean batches on the EAGER `model`; return (bn_modules, backup) for restoration. Add `restore_bn(bn, backup)` to copy the snapshot back.
- [ ] Build a clean recalibration DataLoader: CIFAR10 train split, transform = `ToTensor + Normalize(mean, std)` ONLY (NO crop/flip/AugMix — matches the frozen eval transform in prepare.py L15-19), batch 128, shuffle True, num_workers=2 (light, to not starve the main 8-worker loader), drop_last False.
- [ ] In the training loop, before the existing `evaluator.evaluate(model, device)` (train.py L275): if `total_training_time / TIME_BUDGET_S > 1 - BN_RECAL_FRAC` (tail epochs), call `recalibrate_bn(...)`, run the (already-present, single) evaluate(), then `restore_bn(...)` so the next epoch's training resumes from the augmented running stats. NO added evaluate() call — exactly one eval per epoch preserved.
- [ ] Constants: `BN_RECAL_FRAC = 0.025` (last ~2 epochs), `BN_RECAL_BATCHES = 16` (2048 images — ample for stable BN population stats on balanced CIFAR-10).
- [ ] Smoke: `ast.parse` OK; `git diff --name-only` = train.py only; unit-check the helper on a tiny run — confirm a BN layer's running_mean CHANGES after recalibrate_bn and is byte-restored after restore_bn; confirm `num_params` unchanged (4,299,866).

### Milestone 2: Launch on idle GPU + early gates
- [ ] Pre-launch `nvidia-smi` idle-GPU check; launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background).
- [ ] dt/compile gate (~ep5): confirm dt steady ~8ms (the compiled training forward is UNCHANGED — recalibration is eager-only, so no cudagraph break per infra-errors EXP-042). No NaN, loss descending.
- [ ] WALL gate (~ep8): project base wall = real ms/step × est-steps + ~100s startup/eval. AugMix recipe is wall-tight (~593s, EXP-054). Recalibration adds only ~2 epochs × 16 batches × ~11ms ≈ 0.35s (negligible). If projected base wall > ~596s, reduce BN_RECAL_FRAC to final-epoch-only and/or relaunch — but base wall should match EXP-054.

### Milestone 3: Completion + verification
- [ ] Run exits 0, prints summary; extract metrics, compare to baseline, render conditions.
- [ ] Confirm `evaluate()` was called exactly `num_epochs` times (grep `eval ep` count == num_epochs) — the ≤1-eval/epoch constraint.

## Code Changes
- **train.py**: (1) add `recalibrate_bn` + `restore_bn` helpers (module level, near `cutout_batch`); (2) add two constants `BN_RECAL_FRAC`, `BN_RECAL_BATCHES`; (3) build `clean_loader` in `main()` after `train_loader`; (4) wrap the existing per-epoch `evaluate()` (L275) with a tail-epoch recalibrate→eval→restore. Conceptually: in the final ~2 epochs, the eval-time BN running stats are recomputed on clean (un-augmented, un-occluded) training images so the eval normalization matches the clean test distribution instead of the Cutout(zeros ~25%)+AugMix(distorts ~50%) training distribution. Risks/edge cases: (a) run recalibration ONLY on the eager `model` (never `compiled_model`) — avoids the EXP-042 cudagraph break; (b) restore augmented stats after eval so training is unperturbed; (c) the 2-worker clean loader must not starve the main loader (only iterated between epochs, when the main loader is idle); (d) best_test_acc is a max over epochs, so replacing the last ~2 flat-tail evals with clean-BN evals can only help or be neutral (small masking risk mitigated by limiting to the flat tail, leaving the natural augmented-BN peak at ~ep87-89 intact).

## Configuration Changes
- New: `BN_RECAL_FRAC = 0.025`, `BN_RECAL_BATCHES = 16`. Rationale: last ~2 epochs (cosine→0, most-converged weights) get the clean-BN eval; 16 batches = 2048 images is ample for stable BN population statistics on balanced 10-class CIFAR-10 while keeping wall cost ~0.35s (protecting the wall-tight AugMix recipe). Everything else byte-identical to EXP-054 (k=4 WideResNet-20, AugMix-p0.5, Cutout16, cosine peak0.2/warmup0.05, Nesterov m0.9, WD1e-4, LS0.1, batch128, seed42, compile reduce-overhead). num_params unchanged.

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background bash.
- Resources: single idle H20 (pre-check nvidia-smi; relaunch on contention per infra-errors).
- Estimated runtime: ~91 epochs, dt ~8ms, Σdt ~300s, wall ~593s + ~0.35s recalib ≈ ~593s (< 600s).
- Log output: `run.log` in project root.
- Tool skill: none (local).

## Abort Criteria
- dt drifts ≫ 8ms steady from ep1 (would indicate a cudagraph break — should NOT happen since the compiled forward is untouched; if it does, the recalibration code accidentally perturbed the compiled path → fix and relaunch).
- Projected base wall > ~600s at the gate → reduce recalibration to final-epoch-only or relaunch.
- Loss NaN/inf or not descending by ep5.
- GPU contention mid-run (wall/Σdt ≫ 2.5×): kill, relaunch on a clean idle GPU.
- Recalibration error (BN reset/restore exception): fix the helper and relaunch (code error, single retry).

## Verification Protocol

### Verification Procedure
Baseline = **96.45** (from `exp-index.sh baseline`); bar = **96.55**.
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: after exit, `grep -aE "^best_test_acc:" run.log`; parse float; PASS iff `>= 96.55`. (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: `grep -aE "^total_seconds:|^num_epochs:|^num_params:" run.log`; confirm summary printed, `total_seconds < 600`, total wall < 10 min, `num_params == 4,299,866`, and `grep -ciaE "nan|traceback|error" run.log` == 0.
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == train.py only; prepare.py/Eval.evaluate() untouched; **evaluate() called exactly once per epoch** — verify `eval-ep` line count == `num_epochs` (recalibration adds NO evaluate() call); no new deps (only torch/torchvision already present); seed 42 unchanged; ran uncontended (steady ~8ms dt, wall/Σdt ≲1.5×).
- Verdict: improvement iff all three pass; no-improvement if a necessary condition fails on a valid run; invalid on scope/dep/eval-count breach; crash if no metrics.
- Timeout: 10 min wall. Cleanup: `rm run.log` after recording.

### Informational Metrics (Optional)
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log`
- num_epochs/num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — confirm ~91 (unchanged; recalibration is off the Σdt timer).
- final_test_loss + tail eval lines: compare clean-BN tail-epoch acc vs the augmented-BN peak to isolate whether recalibration helped/hurt/null.
