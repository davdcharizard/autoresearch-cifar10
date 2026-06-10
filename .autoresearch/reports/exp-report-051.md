# Report EXP-051: LayerScale — learnable per-channel residual-branch scaling
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-051.md
- **Plan**: plans/plan-051.md
- **Log**: logs/exp-log-051.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) by editing only `train.py` within the fixed 300s budget on a single H20. Baseline = **96.22%** (EXP-012, 6c417a4); bar = baseline + 0.1 = **96.32%**. With 43 prior no-improvements and both governing walls firm, this loop tested the cleanest remaining throughput-(near-)free sub-lever: a learnable residual-branch scale.

## Idea & Hypothesis
Chosen idea: add CaiT LayerScale — a learnable per-channel vector (init 0.1) multiplying each BasicBlock's residual branch before the shortcut add. Reasoning: gives the residual branches a learnable magnitude DOF (identity-dominated start, learns per-channel residual strength), the modern best-practice version of residual scaling; expected throughput-free. Hypothesis: dt 8ms / ~91 ep; IF a learnable residual-magnitude DOF helps generalization, best_test_acc ≥ 96.32; expected null per EXP-026 (zero-init-γ, "needs depth").

## Approach
Two edits to `BasicBlock` in `train.py`: (1) `__init__` adds `self.layer_scale = nn.Parameter(torch.full((out_channels,1,1), 0.1))`; (2) `forward` inserts `out = out * self.layer_scale` between the residual branch (`bn2(conv2)`) and `+= shortcut(x)`. Smoke test confirmed params 4,301,210 (baseline + 1,344 LayerScale scalars), 9 layer_scale params sized {64×3,128×3,256×3}, all init 0.1; `_weights_init` leaves layer_scale untouched (isinstance Conv2d/Linear only). Init 0.1 chosen over the tiny CaiT 1e-4 to avoid over-suppressing residuals on a shallow 9-block net.

## Execution
One clean run on idle GPU 1, 401.9s wall, exit 0, no retries, no NaN. Early monitoring caught dt at 9ms (not the hoped 8ms) — the LayerScale multiply only partially fused. Judged a mild epoch confound (not a graph break), so the run was allowed to complete rather than aborted (consistent with EXP-028's identical 8→9ms handling).

## Results
- **Primary metric**: best_test_acc **95.83%** (baseline 96.22, delta **−0.39pp**, −0.41%) @ ~ep (best); final 95.78% @ ep85 — 0.49pp below the 96.32 bar.
- **Observations**:
  - **NOT throughput-free**: the `(C,1,1)` pointwise multiply added ~1ms/step (dt 8→9ms, 314×8ms / 346×9ms), the same partial-fusion cost as SiLU (EXP-028). num_epochs fell to 85 (vs baseline 91), num_steps 33,140 — a mild ~6-epoch under-train confound (NOT a CUDA-graph break; that would be 14-16ms).
  - **final_test_loss 0.1987 ≈ baseline 0.195** — no loss gain either (not even a polish win).
  - peak_vram 522.1 MB (up from ~454 — LayerScale activations/grads).
- **Analysis**: The hypothesis is answered NEGATIVELY, with a clean mechanistic read. Two compounding causes for the −0.39pp: (1) **the residual-scaling DOF does not help on this shallow net** — directly confirming EXP-026 (zero-init-γ, "needs depth"); LayerScale's benefit, like zero-init-γ's, is depth-driven and a 9-block ResNet-20 already trains cleanly without it; (2) **the unfused multiply cost ~6 epochs** (the epoch wall, mild here). Even discounting the epoch cost, there is no loss or top-1 gain, so the DOF itself is inert. This fits both governing walls: a compute-(near-)neutral change that is generalization-bound (no top-1) AND a small compute add that costs epochs. The modern LayerScale formulation does not rescue residual scaling on this net.
- **Key Learning**: LayerScale (learnable per-channel residual scale, init 0.1) regressed −0.39pp (95.83): the residual-scaling DOF is inert on this shallow net (confirms EXP-026 "needs depth") AND the multiply unfused (+1ms → 85 ep); residual-scaling is closed in both its zero-init-γ and modern LayerScale forms.

## Verification
- **Conditions**: Condition 1 (`best_test_acc ≥ 96.32`) FAILED at 95.83; conditions 2 (clean run within budget — 401.9s < 600, params 4,301,210, no crash) and 3 (no hard-constraint violations — `train.py` only, eval untouched, once/epoch, no new deps, seed 42, no seed hacking) both PASSED.
- **Review Notes**: Results trustworthy — clean uncontended run, deterministic, scope-clean. The dt 8→9ms cost is a mild confound (it cost ~6 epochs), documented; but since loss also showed no gain, the conclusion (LayerScale inert here) holds regardless of the epoch cost. No integrity concerns.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid clean run; primary necessary condition failed (−0.39pp vs baseline, −0.49pp vs bar).

## Unexplored Avenues
- **LayerScale with smaller/larger init or no WD on the scale**: the lever showed no loss OR top-1 movement, so init/WD tuning would not change the inert conclusion. Not worth a loop.
- **Forcing the multiply to fuse (remove the ~1ms cost)**: even a perfectly free LayerScale would, per the loss being flat, still be a top-1 null — the DOF is inert on this shallow net, so the fusion fix is not worth chasing.
- Residual-scaling is best considered CLOSED in both forms (EXP-026 zero-init-γ + EXP-051 LayerScale).

## Next Steps
- **Residual-scaling axis closed** (EXP-026 + EXP-051). The two remaining brainstorm-051 alternates both carry the same ~1ms-unfused-multiply / epoch-wall risk just observed: PReLU (per-channel activation, EXP-028 SiLU already showed +1ms→null) and Weight Standardization (per-forward weight reduction, dt risk + BN-redundancy). Low confidence either helps; PReLU is the cleaner of the two to try once (low confidence).
- **The genuinely-new-mechanism space is effectively exhausted** (high-confidence assessment): after 44 no-improvements, every axis — capacity, augmentation, schedule, optimizer, normalization, batch, residual-scaling, activations, classifier-head, feature-routing, loss, weight-averaging — is closed, and the two walls (epoch wall for compute-adds; polish-vs-top1 for compute-neutral) bound everything that remains. 96.22 is the firmly-mapped practical ceiling for this k=4/300s setup.
- **Document-the-ceiling confirmation run** (high information, ~zero expected gain): a clean baseline replication characterizing the ±0.25pp noise band is the honest next move; per NEVER STOP, continue running the best-available fresh sub-lever (PReLU) and/or the confirmation run rather than stopping.

## Exit Action Results
<!-- No exit actions defined for this goal. -->
