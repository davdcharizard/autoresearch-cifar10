# Report EXP-031: Scale-Controlled Max-Residual Global Pooling
- **Created**: 2026-08-06

## Goal

Raise CIFAR-10 `best_test_acc` from the 94.15% frontier at `7c1e7d8` to at least 94.25% under the fixed seed-42, one-H20, 300-second, `train.py`-only protocol.

## Idea & Hypothesis

Replace pure global average pooling with `avg + 0.10*s*(max-avg)`, using a frozen training-only calibration `s<=1`. The bounded shared descriptor was intended to preserve area-sensitive CutMix evidence while retaining localized peaks, without EXP014's independent raw-max classifier. The hypothesis predicted at least 99% relative exposure and 94.25% accuracy.

## Approach

Training indices 0-1023 and exact seed-42 CPU initialization produced reproducible `rms_avg=4.7559553078`, `rms_residual=6.9018204627`, and frozen `s=0.68908708` (`c=0.068908708`), making initialization aggregate perturbation exactly 10%. Only `train.py` was modified with the convex descriptor and non-tuning diagnostics. Static checks passed. The registered 200 strong and 64 weak immutable corpora then compared copied accepted/candidate trajectories before timing or production.

## Execution

The first calibration controller hit the known path-import issue and was fixed with the standard root bootstrap. The first safety controller used an invalid `1e-12` FP64 distributivity tolerance; correcting only that oracle to FP32 tolerance allowed the unchanged candidate replay. The corrected replay failed genuine preregistered dynamic gates. Evidence was serialized before assertions. No paired timing or scored accuracy run occurred, and no coefficient/gate rescue was attempted.

## Results

- **Primary metric**: NaN (baseline: 94.15%; no production run)
- **Observations**: Candidate-only >95% class concentration occurred at steps 3, 18, and 19. Update p95/max ratios were 1.338413/1.583063; strong/weak terminal loss-EMA ratios were 1.130417/1.304067. The effective aggregate perturbation rose from its calibrated 0.10 to 0.409621, while maximum per-example perturbation reached 4.341249. Classifier-gradient p95 was a safe 1.126985, showing that aggregate classifier scale alone missed the sparse example-level failure.
- **Analysis**: The headline bound held only for aggregate eval-mode initialization statistics. Once training began, sparse max locations and small per-example average norms made the same fixed convex coefficient effectively dominant for some examples. The shared classifier avoided EXP014's independent-branch recruitment mechanism, but not max pooling's underlying area-insensitive, sparse geometry. The resulting trajectory created class concentration and excessive updates before timing. This exact initialization-calibrated hard-max residual is unsafe; lowering its coefficient after observing the veto would be unprincipled tuning.
- **Key Learning**: Initialization RMS scaling does not bound sparse max-pooling geometry during training; per-example and trajectory gates remain essential even with a shared classifier.

## Verification

- **Conditions**: Calibration, hashes, syntax, scope, parameter identity, and finite execution passed. Exact-corpus concentration, loss, update, aggregate-ratio, and per-example-ratio gates failed. Timing and production were skipped.
- **Review Notes**: The veto is trustworthy: registered corpora matched exact hashes, metrics were serialized before assertions, the only controller corrections addressed known import/tolerance bugs, and neither changed candidate code or thresholds.
- **Verdict**: invalid
- **Verdict Basis**: Immutable safety gates blocked the candidate before a valid metric could be produced; invalid experiments record NaN.

## Unexplored Avenues

- A smooth intrinsically bounded aggregation such as low-temperature log-sum-exp could avoid hard argmax sparsity, but needs a new scale theory and timing review.
- Per-example normalization could enforce a true bound, but introduces data-dependent reductions and changes representation semantics; it is not a rescue of this point.
- Boundary-only momentum reset remains a clean deferred candidate with lower expected impact but negligible runtime and no max-feature geometry.

## Next Steps

- **High confidence**: retire initialization-only hard-max residual scaling and preserve pure global average pooling.
- **Medium confidence**: return to the boundary-only momentum reset as a clean zero-overhead state-transition test.
- **Low confidence**: investigate smooth pooling only with an intrinsic per-example bound and a strong timing premise.

## Exit Action Results

- None defined.
