# Report EXP-031: End-to-End FP32 Channels-Last Training
- **Created**: 2026-07-26

## Goal

Raise fixed-seed CIFAR-10 `best_test_acc` above the accepted 94.32% baseline within the fixed 300-second training budget. This experiment asked whether an end-to-end FP32 channels-last path could preserve the accepted learner closely enough to use any H20 convolution speedup as additional full-model exposure.

## Idea & Hypothesis

Convert model weights and training inputs to channels-last, with an idempotent forward-entry conversion for the frozen evaluator, while preserving FP32 storage, accepted cuDNN/TF32 flags, full gradients, data decisions, hyperparameters, seed, and model logic. The hypothesis required fixed semantic bounds first, then at least 1.02x paired reciprocal-rate speedup and 135.667507 projected passes before a sole score targeting 94.42%.

## Approach

`train.py` converted the directly initialized model before optimizer construction, requested channels-last during pinned input H2D, and converted every forward input to channels-last. A step-zero log line would have audited actual conv-weight and post-mixup tensor layout/stride/dtype. The ignored preflight used an independent `git show 67c8e98:train.py` oracle and preregistered exact logical state/RNG checks, evaluator batches 256/16, per-tensor numerical bounds, post-warm replay, and paired timing gates.

## Execution

Static scope, compilation, local-data, baseline, and idle-H20 checks passed. The first semantic command reached the independent accepted/candidate batch-256 evaluator-logit comparison after construction, layout, input/mixup, and candidate-path checks. It then failed the fixed `rtol=2e-4, atol=2e-5` bound: 1,326 of 2,560 logits were outside tolerance, with maximum absolute difference 0.0008890629 and maximum relative difference 0.0555356. The plan prohibited loosening bounds, so no timing or score was run and no `run.log` was created.

## Results

- **Primary metric**: NaN (baseline: 94.32%; no scored result)
- **Observations**: Logical initialization values and construction RNG were preserved, but legal deterministic NHWC convolution accumulation changed logits much more than the predeclared “same learner” envelope before any training update.
- **Analysis**: The format change cannot be treated as a near-identity exposure optimization under the approved semantics. This does not show channels-last would necessarily train worse, nor whether it is faster; it shows that testing those claims would require accepting a materially different numerical learner after observing the failed bound, which the no-loosening/no-rescue rule correctly forbids.
- **Key Learning**: Channels-last is logically but not numerically neutral on this H20/cuDNN path; exact layout conversion exceeded conservative cross-kernel bounds before timing.

## Verification

- **Conditions**: Static scope/environment passed; the mandatory semantic cross-layout tolerance failed; replay, timing, and primary-metric verification were skipped.
- **Review Notes**: The failure is trustworthy and not an infrastructure issue. The independent accepted arm and fixed deterministic fixture produced a concrete out-of-bound result, and the bound was set before observation.
- **Verdict**: crash
- **Verdict Basis**: No scored metric was produced because a mandatory pre-score semantic gate failed.

## Unexplored Avenues

None within the exact channels-last treatment. The precommitted closure excludes relaxed bounds, alternate conversion placement, cuDNN/TF32 changes, reduced precision, compilation, fusion, batch changes, or LR changes as immediate rescues.

## Next Steps

- **Weaker alpha-0.1 batch-shared mixup (medium-low confidence)**: Run the already-developed one-line closure test of the only remaining mixup-strength bracket.
- **Reflection-padded crop geometry (medium-low confidence)**: Develop a zero-counted-cost input-boundary treatment that is distinct from feature masking and preserves the hard tail.
- **Batch 512 scaled-LR operating point (low confidence)**: Consider only with its strict 1.10 image-rate gate and explicit acceptance that Nesterov/BN/mixup decision dynamics change.
