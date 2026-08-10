# Report EXP-032: Reset Momentum at the 80% Objective Boundary
- **Created**: 2026-08-06

## Goal

Raise CIFAR-10 `best_test_acc` from 94.15% at `7c1e7d8` to at least 94.25% under the fixed seed-42, one-H20, 300-second, `train.py`-only protocol.

## Idea & Hypothesis

Zero all 59 SGD momentum buffers exactly once after the accepted 80% switch evaluation and weak-loader construction, before the first LR-0.01 weak hard-label update. The hypothesis was that high-LR N1/M7+CutMix velocity was stale for the new objective; deleting it would improve refinement without EXP030's excess tail amplitude or the recurring global optimizer instability.

## Approach

Added one in-place reset helper and one transition call; no model, scalar, data, schedule, timer, or evaluator behavior changed. A single accepted source state was trained through 200 immutable strong records, cloned exactly, reset only in the candidate, and replayed through 64 immutable weak records. All 59 buffers went from aggregate norm 1.295493 to zero without changing parameters, BN, gradients, groups, RNG, or logits.

## Execution

One controller-only parameter-group comparison bug was corrected without changing candidate code or gates. Copied-state safety then passed: first/max update ratio 0.532279/1.030338, maximum relative update 0.000348951, own-median max 1.231899, loss-EMA ratio 1.000231, and no concentration. One scored H20 run exited zero without retry; it reset 59 buffers at 80.0%, stopped all eight workers, and realized 49.78% CutMix.

## Results

- **Primary metric**: 93.89% (baseline: 94.15%, delta: -0.26 percentage points, -0.28% relative)
- **Observations**: Switch accuracy was 89.15%, 0.58 below EXP010, while the first weak checkpoint reached 93.21%, 0.05 above EXP010's 93.16%, so reset did not stall immediate adaptation. The tail peaked at 93.89% in epoch 65, regressed 0.05 to 93.84% final, and ended at 0.2047 NLL versus EXP010's 0.1934. Exposure was healthy at 27,039 steps (100.52% of EXP010), with 598.7 MiB VRAM and exactly 19 looks.
- **Analysis**: The reset achieved its local effect safely and immediate weak recovery was at least as fast as the accepted trajectory, but deleting inherited velocity did not improve sustained generalization. The candidate retained more updates yet remained 0.26 points below baseline and had worse NLL. Historical strong-phase variation prevents isolating the exact causal delta, but clean scope, equal observation count, healthy exposure, and the copied-state result rule out instability or overhead. At this operating point inherited momentum is either useful or too transient to be the tail limiter; full deletion has no positive evidence.
- **Key Learning**: Boundary momentum reset improved immediate weak recovery but worsened NLL and peak accuracy; inherited strong velocity was not the limiting tail state.

## Verification

- **Conditions**: Safety, scope, completion, summary, timing, hardware, lifecycle, reset count, target provenance, CutMix fraction, parameters, exposure context, and exactly 19 evaluator epochs passed. Primary accuracy failed: 93.89% <94.25%.
- **Review Notes**: Trustworthy valid result: one idle H20, one seed-42 completion, exact reviewed diff, no reroll, and all preregistered integrity gates passed.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid result was 0.26 points below baseline and 0.36 below the required gate.

## Unexplored Avenues

- Partial/delayed/per-layer resets could retain useful velocity while reducing conflict, but would be post-result tuning without a directional measurement and are not justified.
- Direct boundary cosine measurements could motivate a conditional reset in another setting, but conditioning on training data adds complexity and a new hypothesis.

## Next Steps

- **High confidence**: preserve inherited ordinary momentum through the boundary; retire full reset.
- **Medium confidence**: seek a zero-overhead representation/data mechanism with intrinsic bounds rather than another state or fit-pressure adjustment.
- **Low confidence**: use a systems-only experiment only if the speed gain is independently valuable; extra exposure has not improved accuracy reliably.

## Exit Action Results

- None defined.
