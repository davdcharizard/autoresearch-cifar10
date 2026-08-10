# Report EXP-007: Literature-Scale ASAM in the Validated Clean Tail
- **Created**: 2026-08-05

## Goal

Maximize CIFAR-10 `best_test_acc` under a fixed 300-second charged training budget by testing whether scale-aware perturbations improve the already validated late sharpness-aware phase. The parent and global best were EXP-004 at 95.40%; the frozen necessary threshold was 95.50% and the preregistered mechanism-sized target was 95.70%.

## Idea & Hypothesis

EXP-004 showed that a paid second pass for period-two Euclidean SAM in the clean final quarter improved this lineage. EXP-007 replaced that package with literature-scale p=2 ASAM (`rho=0.5`, `eta=0.01`) while retaining the complete data, model, schedule, optimizer, stochastic, and evaluation paths. The hypothesis was that this package would preserve at least 25,000 optimizer steps and reach at least 95.70%; 95.50-95.69 would be a formal improvement but below the preregistered evidentiary target.

## Approach

Only `train.py` changed. All 30 non-bias tensors used scales `abs(snapshot)+0.01`, while all 14 bias tensors used unit scale. The global denominator was `D=||s*g||_2` and the perturbation was `epsilon=0.5*s^2*g/(D+eps)`. The inherited two-pass path retained CUDA RNG replay, second-pass BatchNorm-stat suppression, exact parameter restore, one Nesterov update, the 0.75 start, and period-two cadence. First-pulse diagnostics measured actual adaptive radius, maximum normalized coordinate, Euclidean norm, maximum scale, and conv/BatchNorm/classifier/bias shares. No planned scalar, cadence, or tensor-coverage value changed.

This was intentionally a comparison between complete optimizer packages, not an equal-Euclidean-radius isolation: the parent used Euclidean `rho=0.05`, whereas the literature ASAM package used adaptive `rho=0.5`.

## Execution

Static, inventory, FP64 closed-form geometry, restoration, RNG, BatchNorm, optimizer, and actual-parent parity checks passed. A full-WRN GPU-0 smoke measured radius 0.500000 and Euclidean norm 0.390258. Warm pulse latency was 20.0858 ms median / 20.1514 ms p90 versus the parent's 20.0628 / 20.2317 ms, projecting about 25,557 steps.

Exactly one fixed-seed run was launched on physical GPU 0 with `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py`. It exited 0 after 463.6 total seconds and no metric-driven adjustment or retry. Three corrected assertions were verification-harness mistakes only: a unit-bias one-scale discriminator, FP64-versus-required-FP32 denominator expectation, and an over-broad nonfinite regex. The production implementation and run configuration did not change.

## Results

- **Primary metric**: 95.34% (parent/global best: 95.40%, delta: -0.06 points, -0.06%)
- **Final accuracy / loss**: 95.18% / 0.1550 (parent: 95.40% / 0.1654)
- **Runtime / memory**: 300.0 charged seconds, 463.6 total seconds, 1.1 startup seconds, 1,213.3 MiB peak VRAM
- **Exposure**: 132 epochs and evaluations, 25,575 optimizer steps, 2,748,890 trainable parameters
- **CutMix**: 10,237 / 20,634 selected, ratio 0.4961, last progress printed 0.7500
- **ASAM**: 2,470 / 4,941 selected, ratio 0.4999, first step 20,636 at progress 0.7501
- **ASAM denominator**: min / mean / max 0.005270 / 0.071997 / 0.332359
- **First ASAM pulse**: radius 0.500000, normalized maximum 0.061053, Euclidean norm 0.450053, maximum scale 1.976884
- **First denominator shares, conv / BN / fc / bias**: 0.141254 / 0.239253 / 0.011648 / 0.607845
- **First epsilon shares, conv / BN / fc / bias**: 0.137905 / 0.097499 / 0.014345 / 0.750251
- **Failures**: nonfinite 0, geometry 0, restoration 0, overlap 0

The ASAM path retained slightly more rather than less optimizer and second-pass exposure, so throughput does not explain the miss. The first production perturbation's Euclidean norm was about nine times the parent's scalar SAM radius and 75.0% of its squared energy was in bias tensors. These measurements characterize the complete fixed package but do not establish that bias concentration persisted or caused the result. Likewise, the 0.0104 lower final loss cannot compensate for the failed accuracy condition or establish improved generalization.

The last eight accuracies were 95.17, 95.26, 95.17, 95.34, 95.24, 95.29, 95.20, and 95.18. Their approximately 0.062 standard deviation places the -0.06 parent delta within observed tail noise, while the 0.36-point miss against the 95.70 mechanism target rules out the predicted large gain for this package.

Claude's corrected adversarial audit (`04-result-review.md`) agreed that the run is trustworthy and the threshold reject is correct. It emphasized that the evidence supports only a package-level negative result, not the claim that adaptive geometry itself is inferior to Euclidean geometry.

- **Key Learning**: Published-default late ASAM preserves compute but offers no detectable gain here; its adaptive radius produced a roughly ninefold larger Euclidean perturbation, preventing geometry-only attribution.

## Verification

- **Conditions**: Scope, hardware, runtime, completion, evaluation frequency, step exposure, parameter count, implementation geometry, restoration, and audit integrity passed; the necessary accuracy condition failed.
- **Review Notes**: The CutMix last progress is printed to four decimals as 0.7500 although the source branch is strictly `<0.75`; direct structural verification established the boundary, but the summary precision alone is insufficient. Both four-group share vectors sum to one. The result is a valid single run without retries or selection.
- **Verdict**: no-improvement
- **Verdict Basis**: 95.34% is 0.06 points below the 95.40% parent and 0.16 below the required 95.50%; the separate 95.70% mechanism-sized target also failed.
- **Tree placement**: failed leaf on `br-000`, parent EXP-004, commit `428cefd`; global best remains EXP-004 at 95.40%.

## Unexplored Avenues

- Match actual Euclidean perturbation magnitude to the parent's 0.05 while changing only adaptive geometry. This would answer a narrower causal question, but its expected effect is close to the protocol's noise floor.
- Collect category-share and perturbation-to-optimizer-step ratios across all pulses. This could test whether first-pulse concentration persists, but instrumentation alone does not improve the primary metric.
- Exclude normalization affine and bias tensors from adaptive scaling while keeping them under the parent perturbation. This may direct more treatment toward feature weights, but it adds another design choice and should not be selected from EXP-007's accuracy.

## Next Steps

- **High confidence**: Return to EXP-004 and prioritize an additive representation mechanism with a plausible effect near 0.3 points, retaining all validated CutMix and SAM exposure.
- **Medium confidence**: Test lightweight identity-preserving channel recalibration in the wider stages if static and GPU latency gates retain at least 25,000 steps.
- **Low confidence**: Use matched-radius adaptive geometry as a preregistered fallback, then retire this axis after one decisive test.
