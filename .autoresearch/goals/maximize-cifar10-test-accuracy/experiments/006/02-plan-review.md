1. **[Abort Criteria, `02-plan.md:55-56`]** The fallback changes `TTA_START_FRAC` from `0.8` to `0.9`, but the plan’s scope says `ResNet9.forward` only and C2 explicitly forbids `TTA_START_FRAC` changes. If this fallback is used, the run cannot honestly pass the plan’s own confinement check.

2. **[Execution Environment / C1, `02-plan.md:49-51`, `train.py:31`, `train.py:198`]** The wall estimate is based on printed `total_seconds`, but `timeout 600` measures process wall time including `uv`/Python startup and top-level `evaluator = Eval()` before `t_start`. The claimed `<600s with margin` can be optimistic even if `total_seconds` later prints under 600.

3. **[C3 Genuineness, `02-plan.md:80-83`, summary prints `train.py:371-381`]** Verification trusts the summary `best_test_acc` but does not cross-check it against the max per-epoch `best:`/`test_acc` lines produced immediately after `Eval.evaluate`. Prior reports used this check; without it, the metric is less tightly tied to the frozen harness output.

4. **[C2 Frozen Harness Check, `02-plan.md:74`]** `git diff --quiet -- prepare.py` only checks unstaged working-tree changes against the index. A staged `prepare.py` edit would pass this specific command. The broader branch diff may catch it, but the frozen-harness check itself is not robust.

5. **[C2 Training-Unchanged Cross-Check, `02-plan.md:77-78`]** A large `num_epochs` deviation is described as a signal that the training path was perturbed, but it is only informational and omitted from the pass condition. If the run clears accuracy with an abnormal step/epoch count, the protocol can still mark C2 passed.

6. **[Milestone 1 Scope Command, `02-plan.md:9`]** `git diff --name-only <dev>` is not executable as written. C2 later names the real branch, but the milestone command is a placeholder and can fail or be skipped during execution.

7. **[C3 Eval-Frequency Check, `02-plan.md:82`]** `grep -c 'evaluator.evaluate(' train.py` is a weak proxy for “at most one validation run per epoch”; it counts text lines, not runtime behavior, aliases, wrappers, or loop placement. It is acceptable only if the manual diff confinement is actually enforced.
