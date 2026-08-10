# Claude Adversarial Plan Review — EXP-005

## Prioritized Concerns

1. The plan specified the `tau^2` KL multiplier but did not explicitly require dividing both teacher and student logits by `tau`. Existing zero/gradient smokes would not catch that omission.
2. The 299.5-301.0-second window needed evidence that the parent checks the budget per step rather than only at epoch boundaries, especially after doubling natural epoch length.
3. Exact `eval-count == num_epochs` needed evidence for the final partial epoch's counter and evaluation behavior.
4. Attribution must be to the combined repeat-view sampler plus KL mechanism, not KL self-distillation in isolation; an alpha-zero control would be needed to isolate contributions.
5. Fuzzy integrity phrases such as CutMix "near 0.5" and SAM start "approximately 0.75" require numeric tolerances before the result is visible.
6. The DataLoader worker generator must be wired explicitly through `DataLoader(generator=...)`, and the changed parent global-CPU RNG consumption belongs in the confound.
7. Applying `ruff format` could reformat inherited code outside the experimental diff; verify the parent first or omit the formatting gate.
8. The literal plugin-cache `tree.sh` path must be verified before it is relied on.

## Resolution

- Added explicit `softmax(cached/tau)` and `log_softmax(current/tau)` semantics plus a non-identical-logit tau-three-versus-tau-one smoke.
- Verified from the inherited loop that charged time is accumulated after every synchronized step and checked before the next step; the largest atomic overshoot is one step, so the existing 299.5-301.0-second window remains conservative.
- Verified that `epoch` increments before each outer pass and the budget-exhausted partial epoch is evaluated before exit; therefore `eval-count == num_epochs` is the expected exact relation.
- Bound any success claim to the combined repeat-view sampler plus clean-gated KL mechanism and preregistered a future alpha-zero control as the isolation test.
- Added numeric CutMix and SAM tolerances.
- Required `DataLoader(generator=dlb_worker_generator)` explicitly and expanded the RNG/data-schedule confound.
- Ran the inherited-file checks before editing: `ruff check` passes, but `ruff format --check` would reformat the parent. The formatting gate is therefore omitted to avoid unrelated churn.
- Verified the literal executable `tree.sh` path resolves in the current environment and added that precheck to the plan.
