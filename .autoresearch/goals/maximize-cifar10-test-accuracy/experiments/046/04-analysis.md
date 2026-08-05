# Report EXP-046: CIFAR-Mean Crop Fill
- **Created**: 2026-07-27

## Goal

Increase fixed-seed CIFAR-10 `best_test_acc` above the accepted94.48% frontier under the unchanged300-second single-H20 budget. The experiment tested whether the frequent normalized black crop border harmed input invariance and boundary quality.

## Idea & Hypothesis

Replace only `RandomCrop`'s raw-black constant fill with fixed quantized CIFAR mean `(125,123,114)`, already used by accepted RandAugment. The hypothesis predicted at least94.58% best accuracy and127 realized passes if removing the artificial border improved train/test boundary-statistic alignment.

## Approach

Changed one `train.py` argument and retained constant PIL padding, crop geometry/order, all stochastic decisions, early RandAugment/mixup, model, optimizer, schedule, and evaluator. An ignored harness used independent NumPy bytes for all81 offsets/two flip states, checked sampled incidence and RandAugment RNG, replayed the real persistent-worker active-to-inactive transition, then separated trace-heavy semantics from production-payload loader timing. The plan prospectively required both delay-free service and11ms production-overlap stability before H20 timing or scoring.

## Execution

Semantic qualification passed on the first attempt. The treatment changed only intended pad pixels in160/162 forced cases, reached98.748% of sampled crops and13.367% of pixels on average, covered21 distinct RandAugment decisions with exact states, and preserved49,920-sample active/inactive worker traces plus all model state. Loader overlap timing was stable and projected344.169s wall, but three delay-free service CVs exceeded5% and one candidate inactive epoch exceeded its maximum ratio. The preregistered fail-closed gate stopped execution before H20 timing and the sole score; it was not retried or relaxed.

## Results

- **Primary metric**: unavailable (baseline:94.48%, delta: N/A, N/A%)
- **Observations**: Candidate/accepted weighted production-overlap epochs were nearly identical at2.70705/2.70499s with overlap CVs0.17-2.11%. Delay-free accepted-active variability was actually worse than candidate-active (`13.55%` versus`8.21%`), while candidate-inactive was`5.76%`; this prevents attributing the instability specifically to mean fill.
- **Analysis**: The local pixel/RNG mechanism and production-like delivery behavior were both supported, but the complete feasibility contract failed. Because no scored result exists, the accuracy hypothesis is unresolved rather than falsified. The crash verdict records that the exact experiment could not proceed under its prospective service-stability rule. Closure forbids rerunning or relaxing this treatment after seeing the windows, although the evidence does not imply mean fill would hurt accuracy.
- **Key Learning**: Delay-free loader service can be highly variable even when paired production-paced overlap is stable; absolute service-CV gates can block an otherwise wall-safe transform.

## Verification

- **Conditions**: Semantic/source/state passed; loader feasibility failed; H20 exposure and primary metric were skipped.
- **Review Notes**: The failure is trustworthy under the preregistered contract, but not candidate-attributable: accepted active service CV also failed while paired overlap was stable.
- **Verdict**: crash
- **Verdict Basis**: No accuracy result was produced because loader qualification failed before the sole scored run.

## Unexplored Avenues

- The exact mean-fill accuracy effect remains unknown, but rerunning it with a relaxed, overlap-only gate would be post-result protocol rescue and is closed.
- A future, fundamentally different CPU transform may prospectively gate wall feasibility on paired production-paced overlap while treating delay-free service as diagnostic; that protocol lesson does not reopen mean fill.
- Other boundary augmentations require independent evidence and must not inherit alternate colors, modes, widths, schedules, or fill tuning from this result.

## Next Steps

- **High confidence**: Return to a GPU-local or post-pooling mechanism that avoids worker-delivery feasibility entirely.
- **Medium confidence**: For any new CPU transform family, preregister paired production-paced overlap as the wall gate and report unconstrained service only diagnostically.
- **Low confidence**: Fixed pooled-feature mixup remains a distinct high-impact replacement idea, but its removal of validated input mixup makes the prior cautious.
