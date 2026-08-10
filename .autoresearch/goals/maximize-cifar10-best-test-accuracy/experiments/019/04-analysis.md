# Report EXP-019: Balanced Mixup and CutMix Geometry
- **Created**: 2026-08-06

## Goal

Increase CIFAR-10 `best_test_acc (%)` above the 94.15% moving baseline at `7c1e7d8`. A valid improvement required at least 94.25% under the fixed one-H20, seed-42, 300-counted-second training and ten-minute total-wall protocol.

## Idea & Hypothesis

Keep 50% of strong-phase batches hard, replace half of accepted alpha-1 CutMix events with alpha-0.4 Mixup, and preserve the complete N1/M7 plateau and hard weak tail. Mandatory external Claude idea review chose this over Nesterov and EMA because augmentation geometry directly addresses the diagnosed generalization bottleneck. The hypothesis was that complementary regional occlusion and whole-image interpolation would retain at least 99% exposure and raise best accuracy to at least 94.25%.

## Approach

Only tracked `train.py` changed. One forked CPU RNG draw selected 25% CutMix, 25% Mixup, or 50% hard batches; integer provenance replaced target-dimensionality counting, and the existing switch printed all geometry counts. Model, optimizer, schedule, loss, weak loader, timer, evaluator, and total mixed probability stayed fixed. External Claude approved the plan and required the method be described honestly as a compound geometry-and-strength bet, with alpha-0.4 underfit risk accepted rather than hidden.

## Execution

Static and direct semantic gates passed, including bitwise accepted hard/CutMix branches, collator CPU/CUDA RNG neutrality, exact Mixup pixel/target pairing, finite gradients and momentum, and 1,073,962 parameters. The eight-worker 20,000-collation gate also passed: 50.295% hard, 25.190% CutMix, 24.515% Mixup, 49.705% total mixed, all workers stopped, and a hard-label weak loader rebuilt in 2.953 seconds.

The first paired 200-real-batch safety attempt remained finite and ended with candidate/control loss-EMA ratio 0.981399, but hit the pre-registered candidate-only greater-than-95% class-concentration veto. Its controller asserted before printing the stored failure histogram. A serialization-only rerun passed with zero concentration events and final ratio 0.962090, but different loss trajectories showed that fresh forkserver scheduling had changed the augmented source sequence. Because the controller streamed rather than persisted post-transform batches, the rerun was not a replay and could not clear the original veto. Timing and production were skipped; no `run.log` or accuracy metric was produced.

## Results

- **Primary metric**: `NaN` (baseline: `94.15%`; no accuracy run)
- **Observations**: The categorical collator and lifecycle behaved exactly as intended at scale, and both paired attempts showed candidate loss EMA below control without non-finite state. Concentration behavior was inconsistent across fresh worker trajectories: attempt 1 crossed the conjunctive veto, attempt 2 did not.
- **Analysis**: The local data mechanism is feasible and does not show sustained gross loss failure, but the safety evidence is not replayable enough to authorize the sole production run. The first registered failure cannot be ignored after seeing a later pass, while the missing histogram and changed augmentation path prevent strong attribution to alpha-0.4 Mixup. This operating point is unproven, not conclusively unsafe; any reconsideration must start as a new reviewed experiment with exact post-transform batch persistence before paired training.
- **Key Learning**: Forkserver safety gates must persist post-transform batches before training; fresh seed-42 processes did not replay augmentation trajectories.

## Verification

- **Conditions**: Baseline/scope, static source, deterministic semantics, target integrity, 20,000-collation proportions, and worker lifecycle passed. The first paired 200-real-batch concentration gate failed; timing, exposure, production, and accuracy conditions were skipped.
- **Review Notes**: No hard project scope was violated and no test evaluator was used in preflight. The registered threshold was not relaxed and the non-identical passing rerun was not used as a fallback. However, incomplete failure serialization and cross-process augmentation drift make the partial result unsuitable for a production authorization or a strong causal conclusion.
- **Verdict**: invalid
- **Verdict Basis**: The safety gate produced non-replayable partial evidence and blocked production, leaving `best_test_acc` unavailable.

## Unexplored Avenues

- A new Mixup experiment could materialize and persist all 200 post-N1/M7 source batches before either paired arm, serialize every threshold event before assertion, and then test the same fixed intervention under genuinely replayable safety evidence.
- Alpha 0.2 remains a distinct milder geometry/regularization policy from the reviewed alpha-0.4 candidate, but testing it would require a new idea and plan review rather than an EXP019 rescue.
- Isolated Nesterov remains a clean low-cost attribution experiment that avoids worker-side data-policy complexity, though external review rated its likely impact below Mixup.

## Next Steps

- **High confidence**: harden future paired data-policy controllers by persisting exact post-transform source tensors and emitting results before veto assertions.
- **Medium confidence**: prioritize an independently justified candidate rather than immediately rerun alpha-0.4 Mixup; EXP019 did not earn production authorization.
- **Medium confidence**: retain Nesterov as a low-complexity option if the next brainstorm lacks a stronger generalization mechanism.

## Exit Action Results

- None defined.
