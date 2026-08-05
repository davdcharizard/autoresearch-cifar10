1. **Final default can still be the baseline** (`02-plan.md:37,44,47-50`; goal `01-definition.md:55`): the plan runs verdict cells via env vars but explicitly keeps bare defaults as `MODEL=davidnet, USE_COMPILE=0`. The goal’s verification procedure runs bare `uv run train.py`; a “winning” WRN env run would not be what the final frozen harness measures unless the selected WRN/compile config is baked into `train.py` and re-verified bare.

2. **Same seed does not mean same data/augmentation stream** (`train.py:198-230`; `02-plan.md:86`): DavidNet and WRN initialize different numbers of random tensors before the first `DataLoader` iteration. With no dedicated `DataLoader(generator=...)` or RNG restore, shuffle order and worker transform seeds can differ between c0 and cA. A 0.1-0.2pp delta can then be augmentation-order noise, not backbone effect.

3. **Compile warmup is not the real training graph** (`02-plan.md:41`; `train.py:300-303`): the plan’s warmup does not state `with torch.autocast("cuda", dtype=torch.bfloat16)` around `train_fwd`, loss, and backward. If implemented literally, it warms/compiles fp32 while the timed loop uses bf16 autocast, causing first real bf16 compile/autotune inside the 300s budget.

4. **Warmup can leak into timed training** (`02-plan.md:41`; `train.py:312-314`): the plan does not require `torch.cuda.synchronize()` after the warmup and before `t_start_training`. Async warmup kernels can be charged to the first timed step, corrupting `training_seconds`, smoke throughput, and the off-budget compile claim.

5. **Sizing smoke likely overestimates official epochs** (`02-plan.md:42`; `train.py:286,308-310`): a 25s smoke collecting `dt` after step 20 may never include EMA updates if `progress` remains keyed to `TIME_BUDGET_S=300`. Official runs spend ~85% of training with `ema_model.update_parameters(model)` every step, with cost scaling by WRN size. `projected_epochs` can pass 130 while the real run falls below.

6. **Anneal gate uses a weak/ambiguous counter** (`02-plan.md:23,28,63,85`; `train.py:275-336`): `num_epochs` increments before the batch loop, so the final partial epoch is counted as full. The plan also has inconsistent thresholds: pick ≥130, abort only <110, later call <130 suspect. Use of `num_steps >= 130 * len(train_loader)` is the needed hard gate; otherwise 110-129 or partial-130 runs can slip into interpretation.

7. **WRN-vs-DavidNet verdict is heavily confounded** (`02-plan.md:16,38-40,50`; brainstorm `01-brainstorm.md:43,54,95`): cA changes backbone, removes whitening, removes `scale_out`, switches max-pool head to GAP, and keeps/tunes LR under a different logit/gradient scale. A null would not isolate WRN, and a win would be a bundle win, not a clean backbone verdict.

8. **LR contingency creates a test-set tuning hole** (`02-plan.md:50`): “if ep1-5 trajectory is broken, check {0.2,0.1} and adopt the best” is underspecified and likely uses validation/test accuracy from the frozen evaluator. That gives cA extra hyperparameter selection against the benchmark while c0 remains fixed, making the comparison less trustworthy.

9. **Wall-cap risk is not screened by the size smoke** (`02-plan.md:42,56,65`; goal `01-definition.md:43-45`): smoke skips eval, but official WRN runs evaluate every epoch, with EMA and tail TTA. A size can satisfy training-time projected epochs and still exceed the 600s wall cap due to slower WRN evals.

10. **Compile leak monitoring is weaker than prior validated recipe** (`02-plan.md:72-80`; EXP-021 notes): the plan omits first-step/per-epoch dt logging. Its eval-boundary smoke checks the uncompiled eval path, not whether toggling `model.eval()` for raw early eval causes the next compiled train call to recompile. A silent per-epoch recompile would present as under-anneal.

11. **Referenced orchestrator can run the wrong experiment if reused literally** (`02-plan.md:27,58`; `/tmp/exp021_orchestrate.sh`): the existing script launches `DEPTH=0/1` and writes `exp021` logs. The plan says reuse the EXP-021 orchestrator pattern but does not spell out the adapted c0/cA env matrix; literal reuse would not run WRN at all.

12. **Default summary can crash when compile is off** (`02-plan.md:41,43,44`): summary printing adds `warmup_seconds`, but the plan only defines it inside the `if USE_COMPILE` block. The default invariant path `USE_COMPILE=0` can hit an unbound variable unless initialized to `0.0`.
