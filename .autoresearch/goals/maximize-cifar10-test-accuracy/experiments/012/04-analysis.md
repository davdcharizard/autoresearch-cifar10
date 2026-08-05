# Report EXP-012: Exact 8x8 Bottleneck Residual Refinement
- **Created**: 2026-07-24

## Goal

Raise CIFAR-10 `best_test_acc` from accepted 94.07% to at least 94.17% within the fixed 300-second budget by testing a compute-efficient low-resolution transformation.

## Idea & Hypothesis

Append one fixed accepted-initialized `128->64->64->128` pre-activation identity bottleneck after stage 3. Two larger low-resolution capacity probes were positive near misses; this candidate sought a better generalization-per-MAC allocation while retaining at least 92% throughput and 120 realized passes.

## Approach

Added a three-convolution rank-64 bottleneck with no post-add activation, 53,760 parameters, and 3,407,872 MACs/image. Accepted WRN initialization ran first; bottleneck construction/initialization was isolated in a restoring CPU RNG fork, making accepted tensors and subsequent data RNG bitwise identical. All training/evaluation choices stayed accepted. A fail-closed matched preflight verified topology, MACs, semantics, RNG, and production timing.

## Execution

Preflight retained 96.04% throughput and projected 136.28 passes. One fixed-seed H20 run completed without retry or adjustment: one 65% mixup transition, 26,462 steps / 136 epochs, 300.0 counted / 339.8 total seconds, and 28 accepted-cadence evaluations. A nested-script import-path error was fixed before model construction and did not affect scoring.

## Results

- **Primary metric**: 93.74% (baseline: 94.07%, delta: -0.33 percentage points, -0.35%)
- **Observations**: Realized exposure was 135.48544 passes, above both 130.5 projection and 120 interpretation gates and 95.48% of accepted exposure. Best and final accuracy were 93.74% at terminal epoch 136; final loss was 0.2873 versus accepted 0.2432. Peak allocation remained 1,094.4 MiB.
- **Analysis**: The candidate achieved its efficiency mechanism but failed its accuracy mechanism decisively. Adequate exposure, exact RNG preservation, terminal best accuracy, and stable execution rule out throughput collapse, random-trajectory confounding, and premature stopping. The rank-64 correction either constrained useful feature transformation or the immediately active extra branch disrupted the accepted representation; its worse loss resembles EXP-011's generalization degradation without retaining EXP-011's positive accuracy delta. Exact post-stage-3 half-width bottleneck refinement is discredited, not bottlenecks universally.
- **Key Learning**: A rank-64 8x8 bottleneck preserves exposure but loses 0.33 points; efficient low-rank refinement does not retain dense capacity's benefit.

## Verification

- **Conditions**: Completion/process integrity passed; primary improvement failed.
- **Review Notes**: Trustworthy one-H20, one-run result with exact topology/count, frozen evaluator, fixed seed/RNG trajectory, 300.0 counted seconds, 135.49 passes, and `train.py`-only source diff.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid 93.74% is below both baseline and required 94.17%; no retry is permitted.

## Unexplored Avenues

- Endpoint-zeroed bottleneck initialization could reduce initial disruption, but it is a distinct joint architecture/initialization treatment and cannot rescue this result adaptively.
- Other bottleneck ratios or placements remain logically possible, but neighboring compression variants lack evidence after this clear regression.
- Whole-state late EMA remains an orthogonal accepted-model generalization treatment unaffected by this architectural negative.

## Next Steps

- **High confidence**: test the fully specified 65%-start, 0.999-decay whole-state EMA on the accepted model.
- **Medium confidence**: retain safe zero-initialized basic-block endpoints as a later optimization-geometry probe.
- **Low confidence**: defer more low-resolution architecture variants until a new mechanism justifies them.

## Exit Action Results

No exit actions were defined for this local-only goal.
