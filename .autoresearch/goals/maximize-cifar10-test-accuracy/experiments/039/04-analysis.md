# Report EXP-039: Rephase Cosine Across the Hard-Label Tail
- **Created**: 2026-07-27

## Goal

Maximize fixed-seed CIFAR-10 `best_test_acc` above the accepted 94.48% baseline under the frozen 300-second counted-training contract. EXP039 tested whether aligning a continuous cosine phase to the 65%-100% hard-label tail could improve the accepted learner by at least 0.10 points.

## Idea & Hypothesis

The accepted recipe changes from mixup and early RandAugment toward hard labels and clean crop/flip around 65%, but its global cosine continues without rephasing. The candidate preserved the accepted LR exactly through the boundary, then cosined from the accepted boundary LR to the accepted 0.002 endpoint over the remaining 35%. The hypothesis required best accuracy at least 94.58% and at least 127 passes; final accuracy at least 94.45% and loss at most 0.2456 were corroboration.

## Approach

Only `learning_rate()` changed. The curve retained accepted returned values before 65%, exact anchor `0.06123215295935604`, and exact endpoint `0.002`; candidate tail LR area rose from `0.00793445` to `0.01106563`, a 1.3946300912 ratio. Model, pooled head, initialization, optimizer state, momentum, coupled decay coefficient, data/RNG, augmentation transitions, evaluator, seed, and budget remained accepted. Because SGD decay is coupled, higher LR scaled both data-gradient motion and decay integration.

## Execution

Two disposable verifier assertions were corrected before timing or scoring: monotonicity was properly restricted to after the intentional warmup, and the analytic area-ratio reference was given enough digits for its declared tolerance. The production treatment never changed. Semantic qualification then proved schedule, state, RNG, and fresh/preseeded Nesterov oracles. Timing retained 0.999378 throughput, projected 130.223 passes, stayed below 0.29% CV, and peaked at 610.16 MiB.

The sole score completed without retry or runtime error. Mixup stopped at step 16,494 and 195.0 seconds; RandAugment stopped after the epoch-85 iterator exhausted at step 16,575. Logged LRs followed the rephased curve, and 27 unique evaluations followed the required cadence.

## Results

- **Primary metric**: 93.98% (baseline: 94.48%, delta: -0.50 points, -0.53%)
- **Observations**: Final accuracy equaled best at 93.98%, but final loss worsened from accepted 0.2456 to 0.2661. The run delivered 25,628 steps, 131.21536 passes, 132 epochs, 1,096.4 MiB peak VRAM, and 1,003,482 parameters in 300.0 counted / 341.9 wall seconds.
- **Analysis**: The treatment achieved its exact intended optimization effect at normal exposure, so compute loss or implementation contamination cannot explain the regression. A 39.46% larger tail LR area did not improve hard-label adaptation and instead worsened both boundary accuracy and evaluator CE. Together with EXP008's harmful zero endpoint, the accepted global cosine and nonzero floor are locally favored over the two tested lower/higher tail-motion changes. The evidence does not isolate data-gradient motion from coupled-decay integration and does not prove every tail schedule ineffective.
- **Key Learning**: The accepted tail schedule is locally bracketed: both reduced late motion and a 39.46% area increase worsen accuracy and loss at normal exposure.

## Verification

- **Conditions**: Completion/resource contract passed; primary metric improvement failed.
- **Review Notes**: Results are trustworthy. One H20, a single fixed-seed score, `train.py`-only scope, independent schedule/update/RNG checks, 131.215 passes, correct transitions, once-per-epoch cadence, and complete timing/summary evidence all passed.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid 93.98% score missed baseline by 0.50 points and the required 94.58% threshold by 0.60 points, with worse loss.

## Unexplored Avenues

- A one-time momentum-buffer reset at the hard-label boundary remains a distinct state intervention, but inherited momentum decays below 1% in about 44 steps and therefore has lower expected impact than the rejected sustained rephase.
- Other tail peaks, durations, and shapes remain formally untested, but the two-sided local evidence deprioritizes schedule tuning absent a new diagnosis.

## Next Steps

- **High confidence**: Preserve the accepted global cosine and 0.002 endpoint; seek a mechanism outside tail LR amplitude.
- **Medium confidence**: Investigate normalized classifier geometry only with a training-derived, prospectively fixed scale and explicit decay semantics.
- **Low confidence**: Test isolated transition momentum reset only if no stronger orthogonal mechanism survives review; do not combine it with this failed rephase as a rescue.
