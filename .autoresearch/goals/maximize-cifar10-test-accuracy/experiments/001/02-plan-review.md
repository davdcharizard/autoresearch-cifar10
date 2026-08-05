## Prioritized Concerns

1. **Plan §Execution Environment / §Abort Criteria / §Verification: wall-clock failure is not actually enforced.**  
   The command is plain `CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1`, while the pass condition relies on printed `total_seconds` (`02-plan.md:133-150`). That does not kill at 600s, and current `train.py` starts `t_start` inside `main` after top-level `evaluator = Eval()` (`train.py:25`, `train.py:107`), so printed `total_seconds` can undercount real process wall time.  
   **Fix:** run under an external wall timeout, e.g. `timeout 600s bash -lc 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'`, and treat exit `124` as failure. Also either move evaluator construction inside the timed region or verify wall time externally with `/usr/bin/time`.

2. **Plan §Execution Environment: eval-per-epoch overhead is under-specified and may break the 10-minute cap.**  
   The plan estimates `~60-120s eval` (`02-plan.md:135`) but keeps one frozen `Eval.evaluate` after every epoch (`02-plan.md:15`, `train.py:205`). With batch 512 there are only 97 train batches/epoch, so a fast H20 run could produce many epochs. Frozen eval uses `num_workers=8` with default non-persistent workers (`prepare.py:24-30`), meaning repeated eval iterator startup can dominate wall clock. The abort note about `total_seconds` “trending” is not actionable because `total_seconds` prints only at the end (`02-plan.md:141`, `train.py:232`).  
   **Fix:** add per-epoch wall/eval timing prints or external elapsed-time monitoring from process start. If projected wall time exceeds 600s, reduce eval cadence while staying within “at most one validation per epoch,” or otherwise reduce epoch count before the official run.

3. **Plan §Verification: fixed training budget is not a required pass condition.**  
   Verification only requires process exit, `total_seconds < 600`, and a non-empty `best_test_acc` (`02-plan.md:150`). `training_seconds` is only informational (`02-plan.md:156-158`), so the protocol would not catch a timer bug, early stop, or budget extension in edited `train.py`.  
   **Fix:** make `training_seconds` a hard check, e.g. require it to be close to `TIME_BUDGET_S=300` with a small one-step tolerance, and verify `TIME_BUDGET_S`/`prepare.py` are unchanged.

4. **Plan §Code Changes step 5: the LR schedule does not actually “anneal LR→0 exactly at the 300s budget.”**  
   LR is computed from `total_training_time` before the step (`02-plan.md:94-104`). The first update uses `lr=0`, and the final update that crosses 300s uses the pre-step progress, so it can still have positive LR after the nominal budget. This is small but contradicts the claimed exactness.  
   **Fix:** either weaken the claim and accept one-step overshoot, or track previous step time and set LR from an end/mid-step progress estimate; also enforce a hard pre-step budget check before applying another optimizer update.

5. **Plan §Code Changes step 2 / existing `train.py` loop: augmentation/data-loading time remains outside the training timer.**  
   `Cutout` runs in the DataLoader transform before the loop body starts timing (`02-plan.md:34-48`, `train.py:166-180`). That means extra CPU augmentation can increase real wall time while not counting toward `training_seconds`. This is a reward-hacking smell if the plan later adds heavier preprocessing.  
   **Fix:** keep augmentation lightweight, report external wall time, and preferably set `persistent_workers=True` on the train DataLoader to reduce off-budget worker churn. For stricter accounting, restructure iteration so batch fetch time is included in the training timer.

6. **Plan §Milestone 1: the proposed compile/import check is too weak.**  
   `uv run python -c "import train"` (`02-plan.md:16`) does not run `ResNet9.forward`, channels-last conversion, bf16 autocast, loss, or backward. It also triggers top-level `Eval()` side effects in the current file. Spatial or dtype bugs could survive this check and fail only during the full run.  
   **Fix:** add a synthetic one-batch smoke test: instantiate `ResNet9`, send a `[4,3,32,32]` CUDA tensor in channels-last format through bf16 autocast, compute CE loss, run backward and optimizer step. Keep `py_compile` as the pure syntax check.

7. **Plan §Verification step 3: reward-hacking checks are too narrow.**  
   Static verification checks only changed files, seed, and “exactly one `evaluator.evaluate` call per epoch” (`02-plan.md:152`). It would not catch direct access to `evaluator.loader`, an extra `datasets.CIFAR10(train=False)` in `train.py`, or test-label use outside `Eval.evaluate`.  
   **Fix:** add static checks that `train.py` does not instantiate or iterate the test set except through the single frozen `Eval.evaluate` path, and does not reference `evaluator.loader` directly.

8. **Plan §Verification step 2: metric parsing/comparison is underspecified.**  
   `BEST=$(grep ... | grep -oE "[0-9]+\.[0-9]+")` (`02-plan.md:151`) extracts the number, but shell comparison of floats is not defined unless the plan specifies `awk`, Python, or `bc`.  
   **Fix:** use an explicit numeric comparator, e.g. `python - <<'PY' ... assert best >= 91.67`.
