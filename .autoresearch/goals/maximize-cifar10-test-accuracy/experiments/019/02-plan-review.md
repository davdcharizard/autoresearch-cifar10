1. [P0 | Milestones 3-4 / Verification] `cB` is outside the chosen hypothesis but eligible to win. The brainstorm selects SE at layer2+layer3, while the plan runs `SE_LAYERS="123"` and then confirms the “best SE cell.” That adds placement search on the test metric and can credit an unapproved layer1+2+3 variant.

2. [P1 | Milestone 1 Smoke A] The identity-init smoke is mis-specified. The shown `SE` class does not zero `fc2` in a fresh module; zeroing only happens later inside `ResNet9` after `self.apply`. So `torch.equal(se(x), x)` on `SE(c)` will fail as written, and even if fixed in `SE.__init__`, it would not catch the actual post-`apply` clobber risk in the model.

3. [P2 | Milestones 1-2] The plan calls `uv run python <smoke_script>` and implies a custom throughput probe, but the goal allows only `train.py` edits. Unless those are inline or outside the repo, helper files would violate the hard constraint and trip the plan’s own `git status --porcelain` integrity check.

4. [P3 | Smoke B/C] `SE_RATIO` and `SE_LAYERS` are import-time globals. A single smoke script that changes `os.environ` between `""`, `"23"`, and `"123"` after importing `train.py` will not test those configurations unless it reloads/mutates the module or uses subprocesses.

5. [P4 | Contention / Under-anneal] The plan says to re-run if GPU contention appears mid-session, but only logs `nvidia-smi` before each cell. A foreign job starting during an SE run can bias epochs/accuracy without direct occupancy evidence, and the final render path risks collapsing that execution failure into “no-improvement.”

---

## Resolutions (folded into 02-plan.md)

1. **[P0] cB eligible to win → placement-search on test metric.** RESOLVED: verdict is now keyed on **cA (layer2+3, the chosen hypothesis) vs c0 ONLY**. cB ("123") is explicitly DIAGNOSTIC/INFORMATIONAL — a cB-only win is logged as a next-loop hypothesis lead, never a credited EXP-019 improvement. (Milestone 3/4, Verification cond. 3 + render line.)
2. **[P1] Smoke A mis-specified (fresh `SE(c)` not zeroed).** RESOLVED: Smoke A is now MODEL-LEVEL — build `ResNet9(se_layers="123")` and assert each SE submodule returns its input bit-exactly AFTER the post-`apply` zero-init, which is the real clobber guard. Also added in-`__init__` zero-init as defense-in-depth (load-bearing re-zero stays post-`apply`).
3. **[P2] Helper smoke/probe files would violate train.py-only + trip integrity check.** RESOLVED: smoke (`/tmp/exp019_smoke.py`) and probe (`/tmp/exp019_probe.py`) live OUTSIDE the repo, imported via `PYTHONPATH=.`; `git status --porcelain` must show only `train.py`.
4. **[P3] SE_RATIO/SE_LAYERS import-time globals untestable in one process.** RESOLVED: `se_layers`/`se_ratio` are now **`ResNet9` constructor args** (defaulting to the env globals). `main()` reads env; in-process smokes pass explicit values — no env mutation / module reload / subprocess needed.
5. **[P4] Mid-run contention undetected; risk of mislabeling as no-improvement.** RESOLVED: a background `nvidia-smi -l 5` sampler runs for the whole session AND the in-log per-step `img/s` trace is cross-checked; if EITHER fires on any cell, that cell is classified infra-confounded (`crash`/re-run the full set), NEVER collapsed into `no-improvement`. (Milestone 3/4, Verification cond. 2 + render line.)
