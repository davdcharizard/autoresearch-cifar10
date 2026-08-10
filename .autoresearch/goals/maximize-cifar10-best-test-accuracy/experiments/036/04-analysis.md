# Report EXP-036: Reflection-Padded Strong and Weak Crops
- **Created**: 2026-08-06

## Goal

Raise seed-42 CIFAR-10 `best_test_acc` from the moving 94.15% baseline to at least 94.25% by changing only `train.py` under the fixed 300-second/one-H20 protocol.

## Idea & Hypothesis

Replace constant four-pixel RandomCrop borders with reflection in both training phases. The hypothesis was that removing a fixed negative-color crop-position cue while preserving edge texture would improve generalization without changing targets, model compute, or the validated curriculum.

## Approach

Exactly two `padding_mode="reflect"` keywords were added. An ignored controller exhaustively checked all 81 offsets, built aligned accepted/reflection views from identical source indices and RNG states, preserved CutMix decisions/targets, qualified two accepted control pairs first, then replayed identical models over 32 strong and 16 weak paired batches. Loader timing and production were conditional on global safety.

## Execution

Static, offset, RNG, target, corpus, and accepted-control gates passed. The candidate then exceeded the frozen 5x global logit/gradient bounds, so loader timing and production were skipped without retry. No primary metric was produced.

## Results

- **Primary metric**: NaN (baseline: 94.15%; delta: N/A)
- **Observations**: Expected/observed border-changed area was 13.4066%/13.3295%; downstream RandAugment/CutMix propagated differences to 23.6893% of tensor elements. Accepted calibrations stayed below 1.80x logits, 1.76x gradients, and 1.25x updates. Reflection reached 20.7200x logits and 9.8057x gradients on strong steps, while updates peaked at 2.9802x. There was no candidate-only concentration; whole update/parameter was 0.04719 and strong/weak loss EMA ratios were 1.0499/1.0957.
- **Analysis**: The intervention implemented its border prior exactly and the candidate-specific global divergence is trustworthy because controls passed before candidate authority. Mild average loss degradation and absence of class collapse cannot waive persistent output-scale excursions. This discredits the exact two-phase reflection policy under the registered safety standard, but supplies no accuracy claim.
- **Key Learning**: Reflection replaced the intended 13.33% border area yet amplified aligned strong-view logits 20.72x; the exact two-phase policy is unsafe to score.

## Verification

- **Conditions**: source/static/offset/RNG/target/corpus/control qualification passed; candidate global safety failed; loader timing/production/metric skipped.
- **Review Notes**: Report SHA `b89ccea220e7a6394337e51c6ae612086b1f2b2d7349a62fb0d39721698dd570`; controls passed prospectively, avoiding EXP035's non-specific-gate flaw.
- **Verdict**: invalid
- **Verdict Basis**: A preregistered candidate-specific safety veto blocked production, leaving partial evidence and no primary metric.

## Unexplored Avenues

- Strong-only, weak-only, narrower, edge, or symmetric padding are adjacent rescues without independent evidence and are retired for this line.
- Reflection loader cost remains unmeasured because model-trajectory safety failed first.

## Next Steps

- **Move away from crop-border tuning** (high confidence): preserve accepted constant padding and regional curriculum.
- **Reassess intrinsically small classifier/normalization levers or a new literature-backed representation** (low-medium confidence): require a mechanism plausibly above ten examples.
- **Keep channels-last deferred** (medium confidence): exposure still lacks a causal accuracy link.

## Exit Action Results

- No exit actions were configured.
