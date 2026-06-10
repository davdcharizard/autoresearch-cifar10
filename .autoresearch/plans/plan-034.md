# Plan EXP-034: Later/shorter augmentation cooldown (COOLDOWN_FRAC 0.15 → 0.10)

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-034.md

## Milestones

### Milestone 1: Code changes implemented and passing local checks
- [ ] Re-apply the EXP-033 augmentation-cooldown edits to `train.py` (they were discarded on the EXP-033 no-improvement), with `COOLDOWN_FRAC = 0.10` (NOT 0.15):
  - Add `COOLDOWN_FRAC = 0.10` to the hyperparameter block (~L28, after CUTOUT_SIZE), with a comment noting the final-fraction clean cooldown (TA+Cutout off, RandomCrop+Flip kept).
  - Add `train_tf_clean` = full pipeline minus `TrivialAugmentWide()` (RandomCrop + Flip + ToTensor + Normalize) alongside `train_tf`. `train_set` keeps `transform=train_tf`.
  - Add `aug_cooled = False` before the `while` loop; at the top of each epoch (after `epoch += 1; model.train()`), if `not aug_cooled and total_training_time / TIME_BUDGET_S >= 1.0 - COOLDOWN_FRAC`: set `train_set.transform = train_tf_clean`, `aug_cooled = True`, print `>>> aug cooldown ON at ep {epoch} frac {...:.2f} ...`.
  - Gate Cutout: wrap the in-loop call as `if not aug_cooled: inputs = cutout_batch(inputs, CUTOUT_SIZE)`.
- [ ] `git diff --name-only` shows ONLY `train.py`.
- [ ] AST parse clean (`uv run python -c "import ast; ast.parse(open('train.py').read())"`).
- [ ] Smoke check (`uv run python`): (a) `train_tf_clean` has no `TrivialAugmentWide` while `train_tf` does; (b) model params == **4,299,866** (architecture untouched); (c) `COOLDOWN_FRAC == 0.10`.

### Milestone 2: Experiment launched and confirmed running
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background).
- [ ] Within ~60s: `run.log` shows `params: 4,299,866`, clean compile, step lines, no NaN, loss decreasing, dt ~8ms.

### Milestone 3: Run completes; cooldown fired LATER; throughput-neutrality confirmed
- [ ] Run exits 0 and prints the summary block.
- [ ] **Confirm the cooldown fired at the LATER fraction**: `grep -a "aug cooldown ON" run.log` returns exactly one line at `frac ≈ 0.90` (an epoch ~90% through, around ep ~81 of ~90) — NOT 0.85. If it fires at the wrong fraction → mechanism bug.
- [ ] **Confirm throughput-neutral** — `num_epochs ≈ 90` and `dt ≈ 8ms` (same as EXP-033; disabling a CPU transform keeps the GPU-bound step time flat). If epochs drop materially → note as a confound.
- [ ] `total_seconds < 600`.

## Code Changes
- **train.py** — identical to the EXP-033 diff EXCEPT the constant is `COOLDOWN_FRAC = 0.10` (was 0.15). Four edits: (1) add `COOLDOWN_FRAC = 0.10`; (2) add `train_tf_clean` (full pipeline minus TrivialAugment); (3) add `aug_cooled` flag + epoch-boundary `train_set.transform` swap with an observable marker; (4) gate `cutout_batch` behind `if not aug_cooled`.

  **Why this tests the hypothesis**: EXP-033 (COOLDOWN_FRAC 0.15, start frac 0.85) proved the cooldown mechanism produces a real clean-data climb (+0.67pp over ~9 epochs) and is throughput-neutral, but it started too early — cutting strong aug at ep77 (95.43%) sacrificed ~5 epochs of productive strong-aug training. Starting later (frac 0.90, ~9 clean epochs) preserves that strong-aug training so the clean fine-tune lifts from a higher pre-cooldown base, plausibly clearing the 96.32 bar. This changes exactly ONE variable vs EXP-033 (the start fraction) for clean attribution.

  **Transform-switch mechanism (same as EXP-033, verified working there)**: `num_workers>0` with `persistent_workers` defaulting to False → each epoch's loader iterator spawns fresh (forked) workers that snapshot `train_set` at spawn; mutating `train_set.transform` at the epoch boundary before the iterator is created propagates to that epoch's workers. EXP-033 confirmed the marker fired correctly and the tail behaved as expected.

  **Risks/edge cases**: (a) ~9 clean epochs (vs EXP-033's ~14) may be too few for the climb to fully develop → smaller lift → no-improvement. (b) The +0.67 lift may not be fully additive on a higher base (diminishing returns near the ceiling). (c) Run-to-run epoch variance ±0.2pp. (d) `num_params` UNCHANGED (4,299,866). No crash/invalid path.

## Configuration Changes
- `COOLDOWN_FRAC`: 0.15 (EXP-033) → **0.10**. Rationale: start the cooldown at frac 0.90 (~9 clean epochs, matching EXP-033's observed ~9-10-epoch climb-to-peak duration) so the clean fine-tune both completes within budget AND starts from a stronger pre-cooldown base. The new constant is the only change.
- All else UNCHANGED: PEAK_LR 0.2, WARMUP_FRAC 0.05, batch 128, WD 1e-4, LS 0.1, CUTOUT_SIZE 16, momentum 0.9 Nesterov, cosine-to-0, seed 42, `torch.compile(reduce-overhead)`, architecture (k=4, params 4,299,866).

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`, background, with a `Monitor` on run.log for per-epoch evals, the cooldown marker, the summary, and NaN/error/dt.
- Resources: single NVIDIA H20 (GPU 0). VRAM ~0.5 GB (≈ baseline).
- Estimated runtime: ~380-411s total. Must stay < 600s.
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
2. **Cond 2 — clean completion within budget.** PASS iff summary block printed, `grep -c Traceback run.log` == 0, `total_seconds < 600`.
3. **Cond 3 — no constraint violations.** PASS iff: `git diff --name-only` lists only `train.py`; `num_params == 4,299,866` (architecture untouched); eval-count == num_epochs (`grep -c "eval ep" run.log` == num_epochs); no new deps (core torch only); seed 42 unchanged.

**MANDATORY cooldown + throughput attribution note:**
- `grep -a "aug cooldown ON" run.log` — expect exactly one line at frac ≈ 0.90 (LATER than EXP-033's 0.85). If it did not fire / fired at the wrong fraction, the run did not test the hypothesis → flag the mechanism bug.
- Record `num_epochs` and mean `dt`. Compute-neutral → epochs ~90 / dt ~8ms → clean fair test of the 0.10-cooldown variant.
- Compare best_test_acc to EXP-033's 96.10 AND the 96.22 baseline / 96.32 bar.

### Informational Metrics (Optional)
- peak_vram_mb: `grep -a "^peak_vram_mb:" run.log` — expect ~0.5 GB.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — expect ~90 / ~35,000.
- final_test_loss: `grep -a "^final_test_loss:" run.log` — compare to baseline 0.195 and EXP-033's 0.2000.
