# Report EXP-014: Zero-Initialized Concatenated Average-Max Readout
- **Created**: 2026-08-06

## Goal

Maximize CIFAR-10 `best_test_acc` above the 94.15% frontier while changing only `train.py`, preserving the fixed seed-42 evaluator/hardware/time protocol, and requiring at least 94.25% for improvement. EXP-014 tested whether class-specific peak evidence could improve the accepted global-average readout without weakening its initial function.

## Idea & Hypothesis

Add a zero-initialized bias-free classifier over global-max final features and sum its logits with the accepted average classifier. Exact zero initialization preserved accepted state, RNG, initial logits, and mathematically the first-step backbone gradient, while allowing the new branch to learn class-specific localized evidence. The hypothesis predicted at least 97% update retention, healthy strong fit, and `best_test_acc >=94.25%`.

## Approach

Kept the accepted EXP-010 model, optimizer, schedule, data, CutMix, timing, evaluation, seed, and lifecycle unchanged. Constructed `max_fc: Linear(128,10,bias=False)` inside a CPU RNG fork after accepted initialization, zeroed its 1,280 weights, and returned `fc(avg) + max_fc(max)`. Production evaluations logged `||max_fc||/||fc||`. Claude's mandatory implementation addendum rejected a verification hook in production; deterministic first-step identity was instead proved with a disposable detached subclass, while the clean model separately proved second-step max-path gradients.

## Execution

Structural checks, pre-commit, hard/soft target checks, and causal preflight passed. Five alternating fresh-process pairs passed timing with a 1.001381x training ratio, 26,860 projected steps, 1.026495x inference ratio, and 598.686 MiB candidate allocation. The sole production run then completed normally: 300.0 counted seconds, 329.8 total, 26,803 steps, one 80.0% switch, eight stopped workers, 19 unique evaluations, and exit 0. A nested timing import and an analysis-only stdin/forkserver diagnostic issue were corrected outside tracked code; neither affected the run.

## Results

- **Primary metric**: 10.00% (baseline: 94.15%, delta: -84.15 points, -89.38%)
- **Observations**: Every evaluation from epoch 14 through 69 was 10.00%; final NLL was 2.3026. `max_readout_ratio` was already 3.963452 at the first evaluation and ended at 3.963398. The strong switch and first weak checkpoint were both 10.00%, so the hard tail could not recover. An exact first-production-batch diagnostic measured max/average classifier gradient norms of 54.39/13.27 (4.10x); after one update the max/average weight ratio was 1.221, same-batch loss jumped from 5.678 to 56.362, and all 128 predictions selected class 7.
- **Analysis**: The intervention strongly engaged but at an uncontrolled scale. Global maxima are much larger and more spatially concentrated than averages, so zero output initialization did not imply a small first update. The accepted global LR applied a fourfold larger classifier gradient to the empty branch, immediately overwhelming accepted logits and producing a self-sustaining chance solution. Equal initial logits and backbone gradients protected only the instant before the first optimizer step; they did not protect optimization continuity.
- **Key Learning**: Zero output is not a safe identity initialization when a new feature statistic has a much larger classifier-gradient scale; gate the first update, not only the initial function.

## Verification

- **Conditions**: Primary accuracy failed; completion, budget, scope, structure, exposure, lifecycle, evaluation-count, target-format, and provenance conditions passed.
- **Review Notes**: Results are trustworthy. Chance accuracy is corroborated by all 19 checkpoints, chance-level NLL, finite summary fields, and the first-step collapse diagnostic; it is not stale output or a parser failure.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid in-scope result completed normally but missed the 94.25% threshold by 84.25 points.

## Unexplored Avenues

- Normalize max features or logits to an accepted average-path scale before learning a bounded contribution; require a preflight first-update logit/weight gate, not only zero initial output.
- Give a max contribution a separately reviewed small LR or bounded scalar gate so the first step cannot dominate; this changes optimizer semantics and must be isolated.
- Test smooth GeM pooling rather than an independent raw-max classifier; distributed gradients and intermediate feature scale may avoid the observed extreme update.

## Next Steps

- **Endpoint weight averaging (high confidence)**: test the prior runner-up that targets weak-tail checkpoint noise without introducing a high-scale feature branch.
- **Activation-aware pooling gate (medium confidence)**: only revisit pooling with measured first-step contribution bounds and normalization fixed before the run.
- **Compute-neutral postactivation initialization (medium confidence)**: seek representation gains while preserving the accepted postactivation topology and strong-fit marker.
