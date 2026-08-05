# Report EXP-015: Per-Example Mixup Strengths
- **Created**: 2026-07-26

## Goal

Maximize fixed-seed CIFAR-10 `best_test_acc` within the 300-second counted training budget, starting from the accepted 94.07% WRN-16-2 recipe at commit `eb08811`. This experiment asked whether changing only mixup coefficient correlation could reach the required 94.17% threshold.

## Idea & Hypothesis

The accepted recipe draws one `Beta(0.2, 0.2)` coefficient for an entire batch. EXP-015 instead drew one independent coefficient per example while preserving the marginal distribution, alpha, pairing rule, 65% mixup window, and hard-label tail. The hypothesis was that greater within-update interpolation diversity would improve the top-1 boundary without adding regularization strength or compute cost.

## Approach

`train.py` sampled a length-`B` Beta vector, broadcast it only over image dimensions, and used a shared production `mixup_loss` helper to combine two unreduced cross-entropies per example before one mean. The ordinary random permutation was retained; derangement, coefficient symmetrization, alpha changes, and cutoff changes were excluded for clean attribution. Model, Kaiming initialization, FP32 SGD/Nesterov, continuous matrix decay, time-based cosine schedule with 0.002 floor, augmentation, loader, seed 42, and evaluation cadence remained accepted behavior.

An ignored evaluator-free harness imported the exact production mixing and loss helpers. It verified coefficient shape/distribution, pixel/label alignment, equivalence to accepted scalar math for a constant coefficient, hard-label state equality, and matched H20 throughput. The only plan clarification allowed the dummy evaluator's required module-level construction while making any evaluation call fail closed.

## Execution

The semantic harness initially failed its post-update equality check because its cuDNN settings did not match production. After locating the failure after, rather than before, the hard-label updates, the harness enabled the production deterministic settings and passed. This was a pre-score harness correction; no scored result or evaluator output was observed.

The sole scored command then ran once on one NVIDIA H20 and exited 0. Mixup disabled exactly once at epoch 92, step 17,745, and 195.0 counted seconds. The run completed 27,737 optimizer steps over 143 epochs in 300.0 counted seconds and 340.3 total seconds, with no traceback, non-finite value, OOM, duplicate evaluation epoch, or scope violation.

## Results

- **Primary metric**: 93.79% (baseline: 94.07%, delta: -0.28 points, -0.30%)
- **Observations**: Preflight established the intended local effect: 4,096 coefficients had mean 0.508997 and variance 0.178111, with 253 distinct values in the first batch. Timing was neutral (1.0040 measured retention), and the scored run realized 142.01344 passes, so the regression is not explained by lost exposure. Final test loss was 0.2628 and final accuracy was 93.62%, versus the accepted report's 0.2432 final loss and 94.07% best/final accuracy.
- **Analysis**: The expected per-example objective and marginal coefficient law were preserved, but destroying batch-level coefficient coherence reduced top-1 by 0.28 points at normal exposure. This rejects the hypothesis that batchwise coefficient correlation is merely harmful noise. The evidence is consistent with batch-shared strength variation providing useful update-level stochasticity or coherent activation statistics: each accepted batch is collectively near-clean or strongly interpolated, whereas the candidate mixes those regimes inside one BatchNorm batch. The run does not distinguish those mechanisms, but it cleanly rules out independent per-example coefficients as a sufficient standalone improvement.
- **Key Learning**: Per-example coefficient diversity preserves mixup strength and throughput but hurts this WRN; retain batch-shared alpha-0.2 coefficients.

## Verification

- **Conditions**: Completion/integrity passed; primary metric condition failed because 93.79% is below 94.17%.
- **Review Notes**: Results are trustworthy. The diff changed only `train.py`, all pre-score semantic and timing gates passed, one fixed seed was used, evaluator behavior was frozen, and the log contains a single complete summary from the scored run.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid complete result with no hard-constraint violation, but the primary metric fell 0.28 points below baseline and 0.38 below the required threshold.

## Unexplored Avenues

- **Structured or antithetic batch coefficients**: preserve a controlled batch-level distribution while reducing extreme update-to-update variance. This is materially different from fully independent coefficients, but it lacks current local evidence and should not be treated as an EXP-015 rescue.
- **Representation-level mixup**: interpolate at a selected hidden stage rather than varying pixel-space coefficients independently. This changes the regularization mechanism and may avoid heterogeneous input statistics, but it requires a fresh proposal and careful compute accounting.
- **Pairing changes**: deranged or class-aware pairing remains untested, but ordinary batch size 256 has only about one self-pair in expectation, so derangement alone has low expected impact.

## Next Steps

- **Medium confidence - objective/representation change**: evaluate a carefully isolated hidden-representation or auxiliary-supervision mechanism that preserves the accepted hard-label tail without merely increasing target softness.
- **Medium confidence - update geometry with measured justification**: measure gradient structure first, then test a projection or conditioning rule only if the diagnostic identifies a substantial nuisance component.
- **Low confidence - calibrated early flatness intervention**: consider a very short SAM-style window only after a training-only perturbation diagnostic validates its radius and exposure trade.
