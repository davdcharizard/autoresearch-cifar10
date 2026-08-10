Below is my adversarial review of `experiments/021/02-plan.md` (with the proposal, brainstorm, idea-review, EXP-010/017 reports, learnings, insights, and infra-errors cross-checked). Concerns are ordered hardest/most-fundamental first. This is critique only.

---

### 1. The plan's own closest evidence points to a miss, and the accuracy verdict sits inside single-run noise (Goal/Hypothesis §, "Point prediction: 94.30%")

The single directly comparable prior is EXP-017: pool + learned 1×1 projection + BN scored **94.09% (−0.06 vs baseline)**. The plan bets that *removing* the learnable projection/BN — i.e., keeping only the pooling that EXP-017 also had — recovers **+0.21 to 94.30%**. That requires the pooling to be net-helpful *and* the learned projection to have been net-harmful, but the equally-live reading (flagged in `01-idea-review.md` #5) is that pooling is the diluting agent and the projection was partially compensating — in which case pool-only lands *below* 94.09. The plan states 94.30 as a point prediction without grounding it in the one data point it has, which argues the other way.

Compounding this: the target margin over threshold is 0.05 pp (5 test images) and the proposal itself concedes the effect is "near the one-seed resolution floor." A bare 94.25% crossing is statistically indistinguishable from trajectory noise, and the protocol forbids replication/seed selection. **Net:** the run cannot produce a trustworthy *accuracy* verdict; its defensible payload is the pre-registered mechanistic discriminator (switch/first-weak/NLL vs EXP-017), which survives even a miss. The plan should frame the expected outcome as "most likely no-improvement, high mechanistic value," not a confident 94.30.

### 2. A self-imposed `num_steps ≥ 26,360` hard gate can discard a genuine 94.25%+ improvement (Milestone 6; Verification step 5)

M6 lists `num_steps ≥ 26,360` as an *integrity* gate alongside exit-zero and param count, and step 5 treats sub-threshold exposure as a validity failure (only sub-94.25 is explicitly downgraded to "no-improvement, not retried"). This conflates attribution cleanliness with metric validity. The goal (`01-definition.md`) only requires `best_test_acc ≥ 94.25`. If the candidate hits, say, 94.28% with 26,300 steps (pool overhead + timing variance), rejecting it as invalid forfeits the goal's actual objective — and a *higher* accuracy achieved with *less* exposure is a stronger result, not a compromised one. Recommend demoting the production step-floor to a diagnostic caveat (record it, compare to EXP-010) rather than a hard reject. (Probability of binding is low — EXP-017 with far heavier shortcuts still hit 26,557 — but the logic is wrong if it ever binds.)

### 3. Evaluator-count parity is enforced as a ceiling only, not matched to EXP-010's 19 (Milestone 4 "projected evaluation count ≤19"; Milestone 6 "at most 19 evaluations")

`best_test_acc` is a max-over-looks statistic, and project-insights (EXP-013) records that look-count biases that max. The plan guards only the *inflation* direction (≤19, no 20th look). But pooling makes the candidate marginally slower, so it may complete *fewer* weak-tail epochs and get **17–18** best-of looks against EXP-010's 19 — a deflation asymmetry that can manufacture a false miss, and which the plan neither measures nor can fix (schedule is fixed). At minimum the plan should extract the candidate's actual look count and compare it to 19 as a verdict caveat, rather than only asserting `≤19`.

### 4. Two recurring infra failures for the disposable controller are not pre-empted (Milestones 2–3 controller + commands)

Both have already bitten preflights on this exact node and will hit the M3 `materialize` step (which spawns the production forkserver loader with 8 workers) and the root-module imports:
- **EXP-010:** Python 3.14 forkserver requires an `if __name__ == "__main__"` guard in the controller; without it the forkserver spawn fails.
- **EXP-016 (`infra-errors.md`):** a path-launched controller (`uv run python .../preflight_pool_option_a.py`) cannot import `train`/`prepare` implicitly — it must resolve project root from `__file__` and prepend to `sys.path`.

Neither is in the M1/M2 checklist. Also note importing `train.py` at module scope executes `evaluator = Eval()` (constructs the test DataLoader); this is allowed (no `evaluate()` call) but the controller inherits that side effect at import.

### 5. Timing arms run under deterministic cuBLAS while production does not — assert backend parity for the ratio gate (Milestone 4; `CUBLAS_WORKSPACE_CONFIG=:4096:8`)

The timing/replay commands set `CUBLAS_WORKSPACE_CONFIG=:4096:8` (per EXP-020), but the production launch (`uv run train.py`) sets nothing and `train.py` never calls `use_deterministic_algorithms`. If the controller enables deterministic algorithms/`cudnn.benchmark=False` for aligned replay, absolute per-step times are inflated vs production — which makes the projected-wall (<540s) and projected-steps gates *conservative* (production faster → more steps), so no false pass. But the load-bearing `candidate/control ≤1.02` gate is only meaningful if **both arms share identical determinism / `cudnn.benchmark` / TF32 flags**; the plan should explicitly assert that, otherwise the 2-launch-bound-kernel delta could be swamped or masked by asymmetric backend selection.

### 6. Shape equivalence of `::2` → `avg_pool2d(2,2)` silently relies on even spatial dims (Milestone 2; Code Changes)

`x[:, :, ::2, ::2]` returns `ceil(N/2)` while `avg_pool2d(k=2,s=2,ceil_mode=False)` returns `floor(N/2)`; these agree only because the transition inputs are even (32→16, 16→8). They are even here, and hardcoding `kernel_size=2, stride=2` (dropping the `self.stride` dependency) is correct because every `need_pad` block in this architecture is stride-2. But the M2 shape-parity assertion should explicitly confirm even input dims and that no `need_pad` block has `stride==1`, so the equivalence is checked rather than assumed — the semantics gate currently asserts the output *values* match a reconstructed reference but does not pin the even-dimension precondition that makes the shortcut output shape match `conv2`'s for the `out += shortcut` add.

---

Scope isolation (train.py-only, gitignored `.autoresearch/` artifacts, `data/` preserved), the Option-A channel-provenance/gradient-0.25 semantics, the replay collapse-veto design, and the anti-reward-hacking posture (no seed reroll, no fallback variant, exposure/eval caps, single non-retried run) are sound and I raise nothing on them.
