# Plan EXP-035: Clean-tail LR reheat (aug cooldown @0.10 + re-annealed LR 0.02→0 on the clean phase)
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-035.md

## Milestones

### Milestone 1: Code changes implemented and smoke-checked
- [ ] Add `COOLDOWN_FRAC = 0.10` and `CLEAN_LR0 = 0.02` constants after `CUTOUT_SIZE = 16` (L28).
- [ ] Add `train_tf_clean` Compose (full pipeline minus `TrivialAugmentWide()`) after the `train_tf` block.
- [ ] Add `aug_cooled = False` to the loop-init block (with `best_acc` etc.).
- [ ] Add the epoch-boundary cooldown trigger (swap `train_set.transform`, set flag, print observable marker) after `epoch += 1; model.train()`.
- [ ] Replace the LR-assignment block so that when `aug_cooled` is True the LR follows a re-annealed cosine `CLEAN_LR0 → 0` over the clean phase; otherwise the unchanged `lr_at_fraction(frac)`.
- [ ] Gate `inputs = cutout_batch(...)` behind `if not aug_cooled:`.
- [ ] Smoke check: `python -c "import ast; ast.parse(open('train.py').read())"` (AST clean); confirm `git diff --name-only` shows only `train.py`; confirm constants present (`grep -nE "COOLDOWN_FRAC|CLEAN_LR0" train.py`).

### Milestone 2: Experiment runs and cooldown+reheat fire correctly
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background).
- [ ] Confirm exactly one `>>> aug cooldown ON ...` marker fires at frac ≈ 0.90–0.91 (~ep83), and the printed `lr:` in the clean-phase step lines is ≈ 0.02 decaying toward 0 (NOT the ~0.001–0.005 cosine-tail value).
- [ ] Confirm no NaN/inf, dt steady ~8ms, run exits 0.

### Milestone 3: Metrics extracted and verified
- [ ] Parse summary block; record best_test_acc, num_epochs, num_params, total_seconds, final_test_loss.
- [ ] Run verification protocol; render against bar 96.32 and baseline 96.22.

## Code Changes
- **train.py**: Re-apply the EXP-034 augmentation-cooldown scaffold (verified-working transform swap that propagates to forked DataLoader workers at the epoch boundary) and add ONE new element — a clean-phase LR override. Specifically:
  1. **Constants** (after L28 `CUTOUT_SIZE = 16`):
     ```python
     COOLDOWN_FRAC = 0.10  # disable TrivialAugment+Cutout for the final 10% of the budget (clean-data tail)
     CLEAN_LR0 = 0.02      # EXP-035: peak of a re-annealed cosine LR applied DURING the clean tail so the
                           # clean fine-tune has real LR budget (the global cosine has annealed to ~0 by frac 0.90)
     ```
  2. **`train_tf_clean`** (after the `train_tf` Compose, before `train_set`): identical to `train_tf` but WITHOUT `transforms.TrivialAugmentWide()` — keeps RandomCrop(32,pad4)+RandomHorizontalFlip+ToTensor+Normalize.
  3. **Loop init** (with `best_acc = 0.0` ~L211): add `aug_cooled = False  # flips True once the augmentation cooldown begins`.
  4. **Epoch-boundary trigger** (after `epoch += 1` / `model.train()` ~L214-215):
     ```python
     if not aug_cooled and total_training_time / TIME_BUDGET_S >= 1.0 - COOLDOWN_FRAC:
         train_set.transform = train_tf_clean
         aug_cooled = True
         print(
             f"\n>>> aug cooldown ON at ep {epoch} frac {total_training_time / TIME_BUDGET_S:.2f} "
             f"(TrivialAugment+Cutout OFF; Crop+Flip kept; clean-phase LR reheat CLEAN_LR0={CLEAN_LR0})"
         )
     ```
  5. **LR override** (replace L227-229, the `lr = lr_at_fraction(...)` + param-group assignment):
     ```python
     frac = total_training_time / TIME_BUDGET_S
     if aug_cooled:
         # EXP-035: re-annealed cosine CLEAN_LR0 -> 0 over the clean phase (final COOLDOWN_FRAC of budget).
         clean_progress = min(max((frac - (1.0 - COOLDOWN_FRAC)) / COOLDOWN_FRAC, 0.0), 1.0)
         lr = CLEAN_LR0 * 0.5 * (1.0 + math.cos(math.pi * clean_progress))
     else:
         lr = lr_at_fraction(frac)
     for pg in optimizer.param_groups:
         pg["lr"] = lr
     ```
  6. **Gate Cutout** (L223): wrap `inputs = cutout_batch(inputs, CUTOUT_SIZE)` in `if not aug_cooled:`.
- **Why this tests the hypothesis**: items 1-4,6 reproduce the proven EXP-034 cooldown (best non-baseline 96.26). Item 5 is the single new variable — it gives the clean-data tail a real LR (~0.02 decaying to 0) instead of the near-frozen ~0.001–0.005 global-cosine value, so the model can take meaningful gradient steps toward the clean/test distribution optimum. If the cooldown's +0.04 cap was LR-starvation, this lifts best_test_acc above 96.26 toward the 96.32 bar.
- **Risks/edge cases**: (a) the 0.02 reheat is a small upward LR step (~4× the cosine-tail value) at the switch — could mildly perturb the converged solution (SGDR lesson), but it is 10× smaller than SGDR's 0.2 restart and re-anneals to 0, so worst case is a small regression, not divergence. (b) `clean_progress` is clamped to [0,1] so the LR is well-defined for the whole clean phase. (c) No FLOP/param change → throughput-neutral; transform swap verified neutral in EXP-033/034.

## Configuration Changes
- COOLDOWN_FRAC: (new) → 0.10 (matches EXP-034, the best cooldown window; later/shorter is strictly better per goal-learnings).
- CLEAN_LR0: (new) → 0.02 (grounded in EXP-020's best SWA floor; 10% of peak LR — large enough to enable clean-data adaptation, small enough to be gentle on the converged solution).
- All else unchanged: PEAK_LR 0.2, WARMUP_FRAC 0.05, CUTOUT_SIZE 16, BATCH_SIZE 128, LS 0.1, WD 1e-4, seed 42, k=4, torch.compile(reduce-overhead). Params stay 4,299,866.

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`, launched in background.
- Resources: single NVIDIA H20 GPU; ~454 MB peak VRAM (well within budget).
- Estimated runtime: ~405–411s wall (300s training + ~startup/compile + eval), well under the 600s ceiling.
- Log output: stdout+stderr → `run.log` in the project root. Step lines use `\r`; to read them: `tr '\r' '\n' < run.log | grep -oE "lr: [0-9.]+"`. Summary block printed at end.
- Tool skill: none (local run).

## Abort Criteria
- NaN/inf in loss (the 0.02 reheat destabilizes) → kill, set Outcome failed.
- No `run.log` growth for >120s after launch → kill, inspect.
- dt blows up (>15ms steady) or epochs collapse far below ~85 → throughput regression, note (would confound the test).
- Total wall-clock approaching 600s → kill (budget ceiling).
- More than one `>>> aug cooldown ON` marker, or marker fails to fire → logic bug, kill and fix.

## Verification Protocol

### Verification Procedure
Baseline = **96.22** (commit 6c417a4), bar = baseline + 0.1 = **96.32** (from `exp-index.sh baseline`).

1. **Cond 1 — primary metric clears bar** (`best_test_acc >= 96.32`):
   `grep -aE "^best_test_acc:" run.log` → parse the %. PASS iff ≥ 96.32. (Improvement-over-baseline-by-0.1 is exactly the 96.32 bar.)
2. **Cond 2 — clean completion within budget**:
   - `grep -c -a "Traceback" run.log` must be 0; summary block (`best_test_acc:`, `num_epochs:`, etc.) present.
   - `grep -aE "^total_seconds:" run.log` < 600.
   - timeout for the run: 600s wall; treat a hang as infra failure.
3. **Cond 3 — no constraint violations**:
   - `git diff --name-only` lists only `train.py`.
   - `grep -aE "^num_params:" run.log` == 4,299,866 (architecture untouched).
   - eval called once/epoch: count `grep -c -a "eval ep" run.log` == `num_epochs` from summary.
   - core torch only (no new imports added); seed 42 unchanged.
4. **Attribution check (trustworthiness)**: confirm exactly one cooldown marker at frac ≈0.90–0.91, clean-phase `lr:` ≈0.02→0 (via `tr '\r' '\n' < run.log | grep -oE "lr: [0-9.]+"` around the marker), num_epochs ≈ 90-91 and dt ~8ms (throughput-neutral fair test). If throughput is off (epochs ≪88), flag as compute-confounded.

First failed necessary condition ⇒ stop and classify (Cond1 fail with valid run ⇒ no-improvement; Cond3 fail ⇒ invalid).

### Informational Metrics (Optional)
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log`
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — throughput-neutrality check (~91 ep target)
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — compare vs EXP-034's 0.1951 / baseline 0.195
- pre-cooldown base acc: the `best:` value at the eval just before the marker fires (compare vs EXP-034's 96.05)
