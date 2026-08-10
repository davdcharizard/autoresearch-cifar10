# Report EXP-028: Signal-Scale-Matched Positive-Negative Momentum
- **Created**: 2026-08-06

## Goal

Maximize seed-42 CIFAR-10 `best_test_acc` above the moving 94.15% baseline, with at least 94.25% required for improvement, while modifying only `train.py` and retaining the fixed one-H20, 300-second counted-training, under-600-second total protocol. EXP028 tested whether an optimizer-shaped generalization mechanism could improve the accepted width-2/CutMix recipe without disturbing its protected strong phase.

## Idea & Hypothesis

The selected intervention was beta0=1 Positive-Negative Momentum with two `rho=0.81` alternating histories and paper `+2/-1` geometry. A closed-form per-step multiplier matched ordinary PyTorch momentum's response to a constant decay-augmented gradient at every step, removing the paper-default 22.36x coherent-scale mismatch while preserving its stochastic negative-history mechanism. The hypothesis predicted at least 26,091 production steps and `best_test_acc >=94.25%`; immutable-corpus concentration and update gates were explicitly allowed to invalidate the candidate before production.

## Approach

Tracked `train.py` alone received a local `ScaleMatchedPNM` optimizer. It stored a global parity step plus two zero FP32 buffers per parameter, applied unchanged all-parameter coupled `1e-4` decay, used fused foreach operations for the recurrence/update, and left the model, initialization, data curriculum, LR schedule, evaluator, timer, seed, precision, and worker lifecycle unchanged. Final scalar PNM diagnostics were added after the standard summary. An ignored controller independently checked first-step/constant/changing-gradient algebra, optimizer state serialization, inactive-buffer preservation, gradients/RNG, then compared byte-aligned SGD and PNM models on the registered 200-batch strong corpus and a once-persisted 64-batch weak corpus.

The reviewed plan clarified that production and timing would omit per-step synchronizing finite scans, that the fresh timing ratio rather than a historical step projection would govern cost, and that actual steps were hypothesis evidence rather than a way to discard a genuine fixed-budget win. None of these production refinements became active because safety preflight failed.

## Execution

One H20 preflight ran from 18:09:19 to 18:10:14 UTC; there was no retry. The EXP022 strong corpus matched SHA-256 `e04dc2fe...8946`, and the new immutable weak corpus was persisted at SHA-256 `ffefe980...5032`. All 264 paired steps completed with finite state, valid BN counters, unchanged corpora/gradients/RNG, and correct optimizer parity. The controller serialized `preflight-report.json` before raising on constant-direction relative tolerance, candidate-only class concentration, and the paired-update spike gate. Per the preregistration, timing and production were skipped.

## Results

- **Primary metric**: NaN (baseline: 94.15%, delta: N/A; scored production was not authorized)
- **Algebra**: First-step parameters matched SGD within `2.98e-08`; changing-gradient parameters matched the manual recurrence exactly (`0.0` max error), buffers within `7.45e-09`, and state roundtrips passed at steps 1, 2, 63, and 64. Constant-direction update absolute error passed at `9.54e-07`, while relative error `4.19e-06` narrowly exceeded `1e-06` because the compared FP32 delta was small.
- **Trajectory safety**: Candidate-only class concentration occurred on 157 of 264 steps. It began at step 3 with 100% candidate versus 86.72% control class share and remained present at step 264 with 97.66% candidate versus 15.62% control.
- **Update geometry**: Candidate/control total-update ratio had median `0.712803`, p95 `6.697578`, and maximum `12.345637`. At the step-5 maximum, candidate update norm was 12.9813 versus 1.05149 control, loss was 41.0337 versus 10.0437, and predicted-class share was 100% versus 55.47%.
- **Loss behavior**: Strong loss-EMA ratio was `1.138453` and weak ratio `1.347240`, both below the loose 1.5 gate. Their pass did not mitigate concentration/spike failure.
- **Analysis**: The implementation achieved the intended coherent-scale match and correct alternating recurrence, but that proof did not bound the changing-gradient response. PNM did not simply enlarge every update—the median was below control—rather, its parity-dependent history produced rare-to-frequent extreme updates and persistent class collapse. This directly falsifies the safe-generalization premise for beta0=1 scale-matched PNM at global LR 0.1. The 1.30 median gate alone would have missed the failure; paired spike and output-geometry gates were essential.
- **Key Learning**: Constant-gradient scale matching cannot make beta0=1 PNM safe here; alternating stochastic-history spikes reached 12.35x and collapsed classes by step 3.

## Verification

- **Conditions**: Scored verification was not run because mandatory immutable-corpus safety conditions failed before timing and production.
- **Review Notes**: The partial result is trustworthy as a safety failure: exact corpus and starting state were shared, recurrence and state behavior were independently verified, all evidence was serialized before assertions, and no seed/data reroll occurred. The narrow constant-direction relative-tolerance miss is not the cause of rejection; severe concentration and the 12.35x update spike independently reject the candidate.
- **Verdict**: invalid
- **Verdict Basis**: No trustworthy primary metric was produced because the preregistered safety gate blocked the scored run; the index therefore records NaN rather than treating the controller failure as an accuracy result.

## Unexplored Avenues

- A beta0 below 1 with a separately derived coherent-signal normalization could reduce the newest-gradient coefficient, but choosing that value would be a new optimizer hypothesis and should not be an EXP028 rescue. Three optimizer-path failures now make its expected value low without an intrinsic stability derivation.
- Activating PNM only in the low-LR weak tail could avoid the high-LR collapse, but it would have little horizon and would need newly initialized alternating state at a sensitive transition. It remains distinct but weakly supported.
- Conv2d-weight-only gradient centralization, the reviewed EXP028 fallback, changes gradient geometry without alternating history or parameter/state misalignment and preserves ordinary SGD; it remains the cleaner next optimizer-adjacent probe.

## Next Steps

- **Conv-only gradient centralization (medium-high confidence)**: test the reviewed forward-preserving projection with exact decay ordering and a strict kernel-overhead gate.
- **Profile batch-size scaling before another architecture change (medium confidence)**: measure whether H20 sublinear scaling at batch 256/512 can increase examples processed under a noise-aware LR rule.
- **Preserve ordinary momentum for global-LR experiments (high confidence)**: after three concentration failures, prioritize data/representation/system levers unless a new optimizer has an intrinsic update bound.

## Exit Action Results

No exit actions were configured for this goal.
