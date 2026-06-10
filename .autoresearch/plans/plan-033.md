# Plan EXP-033: Augmentation cooldown — disable TrivialAugment + Cutout for the final ~15%

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-033.md

## Milestones

### Milestone 1: Code changes implemented and passing local checks
- [ ] Add `COOLDOWN_FRAC = 0.15` to the hyperparameter block (train.py ~L28, near CUTOUT_SIZE).
- [ ] In `main()`, build a SECOND CPU transform pipeline `train_tf_clean` = `RandomCrop(32, padding=4)` + `RandomHorizontalFlip()` + `ToTensor()` + `Normalize(mean, std)` — i.e. the full pipeline MINUS `TrivialAugmentWide()`. Keep the existing `train_tf` (full) as the default `train_set.transform`.
- [ ] In the training loop, at the TOP of each epoch (before iterating the loader), compute `frac = total_training_time / TIME_BUDGET_S`; once `frac >= 1.0 - COOLDOWN_FRAC` and not already switched, set `train_set.transform = train_tf_clean`, set a boolean `aug_cooled = True`, and print a one-line marker (e.g. `>>> aug cooldown ON at ep {epoch} frac {frac:.2f}`) so the switch is observable in run.log.
- [ ] Gate Cutout by the SAME flag: in the step body, apply `cutout_batch(inputs, CUTOUT_SIZE)` only `if not aug_cooled`. (TA and Cutout switch off together at the same epoch boundary.)
- [ ] `git diff --name-only` shows ONLY `train.py`.
- [ ] AST parse clean (`uv run python -c "import ast; ast.parse(open('train.py').read())"`).
- [ ] Smoke check (`uv run python`): (a) build both transforms and confirm `train_tf_clean` has no `TrivialAugmentWide` in its `.transforms` list while `train_tf` does; (b) confirm `train_set.transform` reassignment is a plain attribute set (no error); (c) model params still **4,299,866** (architecture untouched).

### Milestone 2: Experiment launched and confirmed running
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background).
- [ ] Within ~60s: `run.log` shows `params: 4,299,866`, clean compile, step lines, no NaN, loss decreasing, dt ~8ms.

### Milestone 3: Run completes; cooldown fired; throughput-neutrality confirmed
- [ ] Run exits 0 and prints the summary block.
- [ ] **Confirm the cooldown actually fired**: `grep -a "aug cooldown ON" run.log` returns exactly one line, at `frac ≈ 0.85` (i.e. an epoch ~85% through, around ep ~78 of ~91). If it never fired or fired at the wrong time → mechanism bug, note and treat the result as not-a-clean-test.
- [ ] **Confirm throughput-neutral** — `num_epochs ≈ 91` and `dt ≈ 8ms`. Disabling a CPU transform should keep throughput flat or marginally faster (TA runs in dataloader workers, overlapped with GPU compute). If epochs drop materially → unexpected; note as a confound.
- [ ] `total_seconds < 600`.

## Code Changes
- **train.py — hyperparameter block (~L28)**: add `COOLDOWN_FRAC = 0.15` with a comment: final fraction of the time budget during which the strong distribution-shifting augs (TrivialAugment + Cutout) are disabled, leaving only RandomCrop+Flip.
- **train.py — `main()` transform setup (~L156-167)**: keep `train_tf` (full, with `TrivialAugmentWide()`). Add `train_tf_clean` = same Compose WITHOUT `TrivialAugmentWide()`. `train_set` is built with `transform=train_tf` (unchanged default).
- **train.py — training loop (~L213-223)**: introduce `aug_cooled = False` before the `while`. At the top of each epoch (right after `epoch += 1; model.train()`), compute `frac = total_training_time / TIME_BUDGET_S` and, if `not aug_cooled and frac >= 1.0 - COOLDOWN_FRAC`, do `train_set.transform = train_tf_clean; aug_cooled = True; print(">>> aug cooldown ON ...")`. In the step body, wrap the Cutout call: `if not aug_cooled: inputs = cutout_batch(inputs, CUTOUT_SIZE)`.

  **Why this tests the hypothesis**: time-varying augmentation is the one untouched lever (every prior run used a static pipeline). Removing the strong, distribution-shifting augs (TA+Cutout) only during the low-LR cosine tail turns the final ~15% into a clean-data fine-tune that aligns weights AND BN running statistics with the test (clean) distribution — a top-1-relevant mechanism (YOLOX close-mosaic; openreview ZcKPWuhG6wy). It keeps FULL augmentation through the entire high-LR learning phase, so it is NOT the "add/reduce a regularizer" failure mode (EXP-005/011/018/022/023) and adds no compute (no epoch wall, project-insights High).

  **Transform-switch mechanism (the one subtlety)**: the DataLoader uses `num_workers > 0` with `persistent_workers` defaulting to False, so each epoch's `for ... in train_loader` creates a fresh iterator that spawns new workers. On Linux (fork start method) the workers receive a copy of `train_set` AT SPAWN TIME, so mutating `train_set.transform` on the main process BEFORE the next epoch's iterator is created propagates to that epoch's workers. This is the standard YOLOX-style "close mosaic" callback pattern. The Milestone-3 grep on the `aug cooldown ON` marker + the epoch-of-firing check confirms it took effect.

  **Risks/edge cases**: (a) Effect may be small → no-improvement within ±0.2pp noise (the net is regularization-saturated). (b) Clean-tail overfitting could mildly regress; bounded by the tiny tail LR (≲0.012) and retained Crop+Flip. (c) If the start method were 'spawn' or `persistent_workers=True`, the mutation might not propagate — neither is the case here (defaults), and the marker+epoch check catches any failure to apply. (d) `num_params` UNCHANGED (4,299,866) — architecture untouched.

## Configuration Changes
- New: `COOLDOWN_FRAC = 0.15` (final 15% of the time budget runs without TA+Cutout). Rationale: aligns the clean-data phase with the low-LR cosine tail (lr ≲ 0.012 over the last 15%, computed from `lr_at_fraction`), mirroring YOLOX's ~5-15-epoch close-mosaic window scaled to this short schedule. First principled probe; a clean test of the mechanism.
- All else UNCHANGED: PEAK_LR 0.2, WARMUP_FRAC 0.05, batch 128, WD 1e-4, LS 0.1, CUTOUT_SIZE 16, momentum 0.9 Nesterov, cosine-to-0, seed 42, `torch.compile(reduce-overhead)`, architecture (k=4, params 4,299,866).

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`, background, with a `Monitor` on run.log for per-epoch evals, the cooldown marker, the summary, and NaN/error/dt.
- Resources: single NVIDIA H20 (GPU 0). VRAM ~0.5 GB (≈ baseline; no new tensors).
- Estimated runtime: ~380-405s total. Must stay < 600s.
- Log output: `run.log` in project root via redirection; source of truth.
- Tool skill: none (local run).

## Abort Criteria
- **NaN/inf loss** at any point → kill, treat as failed.
- **Compile/runtime error** at step 1 (traceback in run.log) → kill, fix, re-launch (counts as the code-fix retry).
- **Loss not decreasing after warmup** → kill.
- **No output / hang**: no new step lines for >120s → kill.
- **Wall-clock runaway**: process past ~580s → kill.
- CUDA OOM (not expected) → kill, treat as failed.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline`) = **96.22**, pass threshold **best_test_acc ≥ 96.32**. Run conditions in order; stop at first failure.

1. **Cond 1 — primary metric clears bar.** After completion:
   `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^num_epochs:|^num_steps:|^num_params:" run.log`
   PASS iff `best_test_acc ≥ 96.32`. Empty `best_test_acc:` ⇒ crash (`tail -n 50 run.log`) → crash verdict.
2. **Cond 2 — clean completion within budget.** PASS iff the summary block printed, `grep -c Traceback run.log` == 0, and `total_seconds < 600`.
3. **Cond 3 — no constraint violations.** PASS iff: `git diff --name-only` lists only `train.py`; `num_params == 4,299,866` (architecture untouched); eval-count == num_epochs (`grep -c "eval ep" run.log` == num_epochs, i.e. ≤ once/epoch); no new deps (core torch only); seed 42 unchanged.

**MANDATORY cooldown + throughput attribution note (EXP-015/024/030/031/032):**
- Record whether the cooldown fired and when: `grep -a "aug cooldown ON" run.log` — expect exactly one line at frac ≈ 0.85. If it did NOT fire (or fired at the wrong fraction), the run did not test the hypothesis → flag the mechanism bug; do not attribute the metric to the cooldown idea.
- Record `num_epochs` and mean `dt`. The change is compute-neutral, so epochs SHOULD be ~91 / dt ~8ms.
  - epochs ~91 & dt ~8ms & cooldown fired at ~0.85 → clean fair test of augmentation cooldown.
  - epochs materially < ~88 or dt risen → unexpected throughput shift → note as a (mild) confound.

### Informational Metrics (Optional)
- peak_vram_mb: `grep -a "^peak_vram_mb:" run.log` — expect ~0.5 GB.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — expect ~91 / ~35,000 (throughput-neutral check).
- final_test_loss: `grep -a "^final_test_loss:" run.log` — compare to baseline 0.195 (cooldown may lower it as BN re-aligns to clean data).
