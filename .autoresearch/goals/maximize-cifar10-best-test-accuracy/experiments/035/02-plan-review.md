# Adversarial Plan Review — EXP-035

## Prioritized Concerns

1. **Trajectory veto authority may retire SiLU without a primary metric.** Three recent experiments were blocked by similar geometry gates. Because SiLU changes all 19 sites, false veto risk should be addressed; the controller uses one short replay and locally declared bounds.
2. **The ignored controller lacks explicit self-tests.** Most execution complexity lives in new hook/ratio/timing code. A math, site-count, or calibration bug could falsely veto the candidate, and one control/control calibration is weak under production-default CUDA nondeterminism.
3. **A fixed 18-19 evaluation-count range is stricter than the goal.** Evaluation count emerges from epoch timing; the user only explicitly requires no more than one validation per epoch. A hard count can invalidate an otherwise compliant run.
4. **Persisted-tensor timing does not include the live DataLoader pipeline.** Projecting absolute exposure from a historical 26,898-step anchor may not capture host scheduling or run-to-run drift; the scored run needs its own actual exposure check.
5. **Milestone timing wording is ambiguous.** “40/40/20 steps” can be read as counts, while the detailed protocol specifies 400/400/200 steps per arm.

## Resolution

- Retained the candidate-only trajectory gate because the corpora are registered real post-transform production batches, not synthetic inputs, and recent failures showed discrete candidate-specific collapse. Added explicit identity/known-array gate-math self-tests and two predeclared control/control repeats; thresholds remain frozen before candidate replay. A veto retires only this exact operating point and is reported as partial evidence, not an accuracy result.
- Added controller source/hash/schema tests, identity telemetry that must never veto, and two control/control calibrations to expose ordinary backend variation before interpreting one-sided candidate events.
- Removed the lower evaluation-count requirement but retained the accepted 19-look ceiling. The cross-goal project insight from EXP013 makes max-over-checkpoint opportunity count an integrity concern; once an extra look has been observed it cannot be unobserved. Once-per-epoch and terminal-look rules also remain.
- Clarified that the persisted timing interval exactly matches `train.py`'s counted `t0` interval: loader transforms and iterator wait occur before `t0` and are unchanged by the activation. The projection is preflight only; actual production exposure and wall time remain load-bearing.
- Reworded the milestone as 40%/40%/20% and retained the concrete 400/400/200 measurement counts.
