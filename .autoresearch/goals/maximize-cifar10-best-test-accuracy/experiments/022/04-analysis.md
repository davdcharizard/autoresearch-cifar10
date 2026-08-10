# Report EXP-022: Lookahead-Wrapped Momentum SGD
- **Created**: 2026-08-06

## Goal

Maximize CIFAR-10 `best_test_acc` above the moving 94.15% baseline while changing only `train.py`, preserving the fixed evaluator and seed, using one 98-GB H20, and remaining within the 300-second counted training budget and 10-minute wall limit. A scored result needed at least 94.25% to advance the baseline.

## Idea & Hypothesis

Wrap accepted momentum SGD with Lookahead at `k=5`, `alpha=0.5`, retaining the inner momentum buffers. The idea was selected because its fast/slow interpolation acts throughout the noisy LR-0.1 RandAugment/CutMix trajectory and is materially different from EXP-018's failed post-hoc uniform SWA. The hypothesis predicted reduced trajectory variance, at least 26,629 fixed-budget steps, and `best_test_acc >=94.25%`.

## Approach

On experiment branch `autoresearch/maximize-cifar10-best-test-accuracy-022`, tracked `train.py` gained a detached slow copy of every optimizer parameter. After ordinary SGD steps 5, 10, 15, and so on, fused `torch._foreach_lerp_` moved slow weights halfway toward fast weights and `torch._foreach_copy_` reset the live parameters; momentum and BatchNorm buffers were untouched. Synchronization occurred before the existing CUDA synchronization so cost would be counted. Existing evaluations were not moved or added.

Adversarial plan review replaced a conceptual per-parameter loop with two fused foreach operations to avoid testing unnecessary tiny-kernel overhead. An experiment-local controller then materialized 200 exact post-N1/M7/CutMix batches and compared byte-aligned control/candidate models before authorizing timing or production.

## Execution

Exactly one H20 with 97,871 MiB was visible and idle. Syntax, Ruff, diff, and tracked-scope checks passed; only `train.py` was modified. The preflight corpus contained 94 hard and 106 probability-target batches. Control and candidate remained bitwise identical through steps 1-4, the first fused interpolation agreed with the algebraic reference within `2.98e-08`, momentum buffers persisted across synchronization, and step 6 updated them.

The mandatory safety gate failed before timing. Candidate predicted one class for 122/128 examples (95.3125%) at step 7 while control was 65/128 (50.78125%), and repeated 122/128 at step 13 while control was 110/128 (85.9375%). No timing or scored production run was launched, and no retry or parameter rescue was attempted.

## Results

- **Primary metric**: NaN / not measured (baseline: 94.15%, delta: N/A)
- **Observations**: The first five-step candidate displacement was exactly half the aligned SGD displacement (`2.5799` versus `5.1598` total L2 norm) while the momentum norm remained identical (`10.5149`). Two and eight steps after the first pullback, candidate-only concentration crossed the 95% veto. All state remained finite, and candidate terminal 200-step loss EMA was lower than control (`1.91333` versus `2.02415`, ratio `0.945253`).
- **Analysis**: The registered persistent-state point changes more than variance. The half pullback leaves the momentum vector from the fast trajectory attached to a different parameter location, and it halves the first committed displacement without halving velocity. The immediate candidate-only class transients support the pre-registered location/velocity mismatch risk. Lower aggregate loss does not establish safe multiclass learning, echoing EXP-020's Nesterov result. Because the screen used one immutable production-distribution corpus, the difference is attributable to the optimizer path rather than forkserver replay drift.
- **Key Learning**: Persistent-momentum Lookahead created candidate-only single-class transients within 13 exact-corpus steps, blocking timing and production.

## Verification

- **Conditions**: Scored verification not run; mandatory pre-production concentration condition failed.
- **Review Notes**: The result is trustworthy as a no-go for the exact operating point. Recurrence/state checks passed, all evidence was serialized before assertions, and control consumed the identical augmented tensors. The controller's first thrown exception cited a harmless FP32 reference difference because that assertion preceded the concentration assertion, but the saved report independently records both concentration failures.
- **Verdict**: invalid
- **Verdict Basis**: The pre-registered safety gate blocked timing and production, so no trustworthy primary metric exists and the index records `NaN`.

## Unexplored Avenues

- **Reset or interpolate momentum at synchronization**: This could remove the observed location/velocity mismatch, but it is a materially different optimizer with its own effective-step dynamics and must not be treated as a rescue of EXP-022.
- **Warm up Lookahead or use a weaker pullback**: Delaying synchronization or choosing alpha closer to one could preserve early progress, but adds schedule/hyperparameter questions and lacks local evidence after two optimizer-path concentration failures (EXP-020 and EXP-022).
- **Evaluation-only short EMA**: A weak-tail EMA avoids feeding a moved parameter location back into persistent momentum, though EXP-018 warns that averaging an annealed trajectory can bias away from the better online point.

## Next Steps

- **High confidence — retire exact persistent-momentum Lookahead**: Do not retry `k=5`, `alpha=0.5`, LR 0.1 or weaken the production-distribution concentration gate.
- **Medium confidence — prioritize a non-optimizer representation lever**: Revisit a low-overhead, identity-scale final-stage attention/readout idea while preserving accepted residual transitions and strong-phase fit.
- **Low confidence — use a cheap FP32 channels-last timing probe only**: It can answer the remaining systems question without committing a scored run if the tiny-kernel H20 workload does not clear a meaningful speed gate.
