# Report EXP-030: Early Drop-Path on the Added Stage-3 Block
- **Created**: 2026-07-26

## Goal

Maximize fixed-seed CIFAR-10 `best_test_acc` above the accepted 94.32% baseline under the fixed 300-second training budget. This experiment asked whether selectively reducing early reliance on the extra `layer3[2]` transformation could improve its generalization contribution without disturbing the accepted hard-label tail.

## Idea & Hypothesis

Apply inverted per-example whole-residual drop-path at p=0.05 only to `layer3[2]` during the accepted first 65% mixup/RandAugment window, using private seed 28028, then restore exact accepted computation. This was narrower than failed broad p=0.10 elementwise dropout and was expected to retain at least 130 passes, reach 94.42% best accuracy, and retain at least 94.32% final accuracy.

## Approach

`train.py` gained an identity-bypassing `apply_drop_path` helper, non-registered block attributes, and a `maybe_disable_drop_path` controller shared by production and preflight. Only `layer3[2]` was configured, its generator was created after model device transfer without changing global RNG, and the controller consumed the existing `use_mixup` predicate before forward rather than introducing a duplicate cutoff. Model parameters, state dict, data pipeline, optimizer, schedule, evaluator, and hard tail remained accepted.

## Execution

Two semantic preflight attempts stopped on harness-only limitations of the dynamic `git show` oracle: source inspection could not locate its class, then forkserver workers could not pickle it. The harness was corrected without changing production code; the third semantic attempt passed all model/RNG/controller/worker checks. Balanced timing retained 0.998064 of accepted fixed-time throughput and projected 132.7499 passes. The sole scored run then completed normally in 341.7 wall seconds with no retry.

## Results

- **Primary metric**: 93.91% (baseline: 94.32%, delta: -0.41 points, -0.43%)
- **Observations**: The treatment realized 132.72064 passes, essentially matching the 132.7499 projection. Mixup and drop-path disabled together at 195.0 seconds; RandAugment disabled 184 steps later after iterator exhaustion. Final accuracy was 93.86% versus accepted 94.22%, while final loss rose from 0.2523 to 0.2887.
- **Analysis**: The intervention achieved its intended isolation and operating regime, so inadequate exposure or contamination does not explain the miss. Both best accuracy and endpoint quality regressed, and loss worsened by 0.0364, consistent with early feature masking disrupting rather than regularizing the useful depth/RandAugment composition. Because the exact one-block p=0.05 treatment was preregistered as a closure run, adjacent probabilities, seeds, blocks, mask granularities, and cutoffs are not justified.
- **Key Learning**: The added stage-3 block needs its full residual contribution during early mixed/augmented training; even narrow p=0.05 masking harms generalization at normal exposure.

## Verification

- **Conditions**: Run validity, exposure, transition ordering, evaluation cadence, scope, and budget passed; primary accuracy and secondary endpoint corroboration failed.
- **Review Notes**: Results are trustworthy. The run used one H20, the frozen evaluator, exact fixed seed and budget, one score, 987,098 parameters, and only the planned `train.py` change. Preflight independently established accepted p=0/eval identity and global-RNG isolation.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid 93.91% score missed the 94.42% threshold by 0.51 points and the 94.32% baseline by 0.41 points.

## Unexplored Avenues

None within this masking family. The precommitted closure rule excludes probability, seed, placement, granularity, cutoff, and rescaling variants after a normal-exposure miss; the broader EXP006 failure reinforces that decision.

## Next Steps

- **Weaker alpha-0.1 batch-shared mixup (medium-low confidence)**: Close the only unmeasured one-constant mixup-strength bracket while preserving duration, coherence, and exposure.
- **Compute-efficient transformation of the added stage-3 block (medium confidence)**: Seek an architectural change that preserves full early participation rather than masking or freezing useful computation.
- **Predetermined checkpoint averaging (low confidence)**: Reassess only if a fixed low-overhead rule can be justified against EXP013's EMA miss and the accepted best-final gap.
