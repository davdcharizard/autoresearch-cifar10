# Plan EXP-063: Augmentation cooldown @0.10 on the AugMix-p0.5 best recipe

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-063.md

## Closed-axis check
project-insights High Importance declares the augmentation axis (strength/policy/coverage/delivery) closed and says "the next gain must come from a NEW lever (NOT augmentation either path...)". **This plan does NOT retry that axis.** Aug cooldown does not change augmentation strength, policy, coverage, or delivery during training — it is a train→eval **distribution-matching schedule**: in the final 10% of the budget it removes augmentation so the model weights AND BN running stats jointly re-adapt to the near-clean distribution that `Eval.evaluate()` sees. This is a DISTINCT mechanism, documented separately in the project's own history (EXP-033/034/035 cooldown family; EXP-061 showed this JOINT adaptation is real and distinct from BN-recalib-alone, which hurts −1.6pp). It is also explicitly the un-combined near-miss: all four prior cooldown runs (EXP-033/034/035/049) were on the superseded TrivialAugment recipe (commit 6c417a4); cooldown has NEVER been applied to the EXP-054 AugMix-p0.5 best. Single-variable, throughput-neutral, wall-neutral-to-faster (removes tail work → no EXP-061 wall-overrun risk).

## Milestones

### Milestone 1: Code change + smoke
- [ ] Add `COOLDOWN_FRAC = 0.10` constant near the other hyperparameters (train.py ~L24).
- [ ] After `train_loader` is built (train.py ~L187), build a clean cooldown transform (`cooldown_tf` = RandomCrop+Flip+ToTensor+Normalize, i.e. `train_tf` minus the `RandomApply([AugMix()], p=0.5)` line) and a `cooldown_loader` (second CIFAR10 instance, `download=False`, same batch/workers/pin/drop_last as `train_loader`).
- [ ] In the training loop: at the top of each epoch compute `cooldown_active = (total_training_time / TIME_BUDGET_S) >= (1.0 - COOLDOWN_FRAC)`; iterate `cooldown_loader if cooldown_active else train_loader`; gate the GPU Cutout call `inputs = cutout_batch(...)` behind `if not cooldown_active:`.
- [ ] Print a one-time marker when cooldown first fires (so the log shows the firing epoch/fraction, matching EXP-034's "fired ep83/frac0.91" record).
- [ ] Smoke: `ast.parse` OK; `git diff --name-only` == train.py only; confirm `cooldown_tf` has NO AugMix and the eval path (`evaluator.evaluate(model, device)`) is unchanged.

### Milestone 2: Launch on idle GPU + early gate
- [ ] Pre-launch `nvidia-smi` idle-GPU check; launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background).
- [ ] Gate (~ep8): dt steady ~8ms (no structural change → identical throughput — the compiled forward is byte-identical), no NaN, loss descending. Cooldown has NOT yet fired this early (fires ~ep82, frac>0.90). Wall projects ≤ ~593s (cooldown lightens the tail; if base wall projects > ~596s before cooldown, note it).

### Milestone 3: Completion + verification
- [ ] Run exits 0, prints summary and the cooldown-fired marker (~ep82, frac~0.90); extract metrics, compare to baseline. Confirm a clean-tail eval climb in the per-epoch eval lines after the cooldown epoch.

## Code Changes
- **train.py — add `COOLDOWN_FRAC = 0.10`** (one constant). Rationale: EXP-034 (@0.10) was the best of the cooldown family; @0.15 (EXP-033) was too long.
- **train.py — `cooldown_tf` + `cooldown_loader`** (after L187): a clean crop+flip-only transform and a second CIFAR10 DataLoader. Why: switching the dataloader to the clean transform is how AugMix is removed in the tail (AugMix is a CPU dataloader transform; the cleanest, least-error-prone switch is a second loader rather than mutating `train_set.transform` under workers). Edge case: `persistent_workers` is unset (default False) so each epoch's `for` creates fresh workers — only one loader is iterated per epoch, so no worker doubling/contention; the second in-memory CIFAR10 (~170MB) is trivial vs 98GB VRAM/host. `download=False` (data already present from `train_loader`).
- **train.py — epoch-level cooldown gate** in the training loop (L221-285): compute `cooldown_active` once per epoch (epoch granularity matches EXP-034's epoch-boundary firing); select the loader; gate the GPU `cutout_batch` call behind `if not cooldown_active:`. Why this tests the hypothesis: in the tail both AugMix (clean loader) AND Cutout (gated) are off, so the model trains on the near-clean eval distribution and weights+BN jointly re-adapt. Risk/edge case: the Cutout gate is a HOST-level Python branch on a bool, OUTSIDE the compiled forward, operating on the input tensor before `compiled_model(inputs)` — the compiled graph's input shape/dtype is unchanged (128,3,32,32 bf16), so NO recompile and NO cudagraph break (contrast EXP-042, which branched INSIDE the compiled forward). Throughput stays 8ms.

## Configuration Changes
- New constant `COOLDOWN_FRAC = 0.10`. All else byte-identical to EXP-054 (k=4 WideResNet-20, AugMix-p0.5, Cutout16, cosine peak0.2/warmup0.05/Nesterov m0.9/WD1e-4/LS0.1, batch128, seed42, compile reduce-overhead). num_params unchanged (4,299,866) — cooldown adds no parameters.

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background bash.
- Resources: single idle H20 (pre-check nvidia-smi; relaunch on contention per infra-errors).
- Estimated runtime: ~91 epochs, dt ~8ms, Σdt ~300s, wall ≤ ~593s (cooldown lightens the tail's CPU augmentation, so wall ≤ EXP-054; < 600s).
- Log output: `run.log` in project root.
- Tool skill: none (local).

## Abort Criteria
- Loss NaN/inf or not descending by ep5.
- dt drifts ≫ 8ms (contention, or an unexpected recompile from the loader switch — should not happen, shapes identical): kill, relaunch on clean idle GPU; if the loader switch itself triggers a recompile/dt-jump in the tail, that is a real finding (note it), not a contention abort.
- No output / hung > 3 min.

## Verification Protocol

### Verification Procedure
Baseline = **96.45** (from `exp-index.sh baseline`); bar = **96.55**.
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: after exit, `grep -aE "^best_test_acc:" run.log`; parse float; PASS iff `>= 96.55`. (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: `grep -aE "^total_seconds:|^num_epochs:|^num_params:" run.log`; confirm summary printed, `total_seconds < 600`, total wall < 10 min, `num_params == 4,299,866`, `grep -ciaE "nan|traceback|error" run.log` == 0.
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == train.py only; prepare.py/eval untouched; evaluate() once/epoch (loop unchanged — eval still at end of each epoch on the eager `model`); no new deps; seed 42 unchanged; ran uncontended (steady ~8ms dt); cooldown-fired marker present (confirms the intended mechanism actually engaged, not a silent no-op).
- Verdict: improvement iff all three pass; no-improvement if a necessary condition fails on a valid run; invalid on scope/dep breach; crash if no metrics.
- Timeout: 10 min wall. Cleanup: `rm run.log` after recording.

### Informational Metrics (Optional)
- peak_vram_mb, num_epochs/num_steps, final_test_loss: `grep -aE "^peak_vram_mb:|^num_epochs:|^num_steps:|^final_test_loss:" run.log` — confirm ~91 ep (throughput unchanged) and compare loss to EXP-054's 0.1968. Also inspect per-epoch eval lines around the cooldown-fired epoch (~ep82) for a clean-tail accuracy climb (the EXP-033/034 signature).
