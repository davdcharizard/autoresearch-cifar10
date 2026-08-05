1. **[Milestone 3 / “Preserve logs” vs Verification NC3]** Copying `run_c*.log` into `.autoresearch/.../experiments/012/` before cleanup can make `git status --porcelain` no longer be “only `M train.py`,” directly conflicting with NC3 and the goal’s “only `train.py` may be modified.” The plan needs a clean separation between metric verification and artifact preservation, or the verification can falsely fail.

2. **[Milestone 3 / “Bake-and-confirm”]** Removing env fallbacks after selecting a winner changes the code after the decisive 4-cell measurement, so the committed/no-env confirmation is not byte-identical to the winning env-toggled cell. This is probably behaviorally equivalent, but it creates a second implementation path and an avoidable source of post-selection bugs.

3. **[Milestone 1 / optimizer partition assert]** The runtime assert only checks `no_decay` is nonempty and `<5%` of learnable params. That does not prove exactness: a stray 1-D learnable parameter would silently enter no-decay, and a future conv/fc bias would also enter no-decay without being caught. The smoke test is stronger, but the in-run guard is too weak for the plan’s exactness claim.

4. **[Milestone 1 smoke command]** The smoke test is described as `uv run python -c "..."` but not concretely specified. If the implementer only replicates `p.ndim` grouping from `m.parameters()` without mapping params back to `named_modules()`/`named_parameters()`, it cannot actually assert “every no_decay param is BN γ/β or alpha” reliably.

5. **[Code Changes / WD_SHAPING=0 baseline-equivalence]** Adding `import os`, env parsing, new summary prints, and `LABEL_SMOOTHING=float(...)` means the no-env script is not byte-equivalent to EXP-008, only intended to be training-behavior-equivalent. That is likely fine, but the plan repeatedly says “byte-equivalent baseline,” which is too strong and can mask bookkeeping or parsing regressions.

6. **[Milestone 2 / same-session execution]** Running the cells sequentially as `c0 → cA → cB → cC` leaves the same-session baseline vulnerable to monotonic host/GPU drift, thermal effects, cache state, or background load changes. The plan says cross-cell ranking holds if all are “equally slowed,” but sequential cells are not guaranteed equally affected.

7. **[Verification NC2 / thin-winner confirmation]** The confirmation rule says a thin winner must clear “on both,” but does not require rerunning the same-session baseline alongside the confirmation. A second winner-only run can pass or fail due to the same epoch/throughput jitter the same-session design was meant to control.

8. **[Milestone 3 / mechanism check]** The plan says to check whether cell-A’s `rezero_alpha` is “measurably larger,” but gives no tolerance or acceptance criterion. This makes the mechanism read subjective, especially because raw `model` alpha and EMA alpha may diverge depending on EMA warmup/tail dynamics.

9. **[Verification NC2 / anti-bookkeeping grep]** `grep "eval ep" → max` is underspecified: the log includes both `test_acc` and `best`, and manual parsing can accidentally compare rounded summary `best_test_acc` against rounded per-epoch `best`. A one-line parser should be specified, or the anti-bookkeeping check is easy to perform inconsistently.

10. **[Milestone 2 / run commands]** `uv run train.py` relies on executable/script behavior. The goal definition uses `uv run train.py`, but if this environment requires `uv run python train.py`, all four planned runs can fail before training. The plan should verify the exact invocation in the smoke step.
