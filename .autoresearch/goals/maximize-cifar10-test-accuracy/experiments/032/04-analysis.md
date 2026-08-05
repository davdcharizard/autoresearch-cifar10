# Report EXP-032: Reflection-Padded Random Crops
- **Created**: 2026-07-26

## Goal

Raise fixed-seed CIFAR-10 `best_test_acc` from the accepted 94.32% baseline to at least 94.42% within the fixed 300-second counted-training budget and 600-second wall limit. This experiment tested whether replacing artificial zero crop borders with reflected image content could improve boundary quality without changing the accepted learner or its GPU exposure.

## Idea & Hypothesis

Change only `RandomCrop(32, padding=4)` to reflection padding. Reflection was selected as an effectively zero-GPU-cost input intervention orthogonal to the failed feature-masking family. The hypothesis was that more natural boundary pixels would preserve at least 130 passes and improve `best_test_acc` by at least 0.10 points while retaining the accepted crop, flip, RandAugment, mixup, optimizer, and model decisions.

## Approach

The production diff added only `padding_mode="reflect"` to the existing crop. An ignored preflight independently compared the accepted and candidate sources, replayed crop/flip/private-RandAugment decisions, checked reflection pixels against a NumPy oracle before RandAugment, traced forkserver-worker decisions across active and inactive phases, and ran balanced real-loader timing. The plan required every loader-timing CV to be at most 5% before the sole fixed-seed score could launch.

## Execution

Static and semantic qualification completed without a scored run or retry. The semantic harness matched 49,920 active and 4,096 inactive per-sample worker traces, all 162 crop/flip oracle cases, 21 independently decoded RandAugment decisions, terminal main RNG, model construction, optimizer construction, and 987,098 parameters. Padding-derived pixels were contacted in 98.8020% of sampled windows.

Balanced loader timing then failed the precommitted stability bound. Candidate active epochs had 11.1988% CV, with raw times `[4.0061, 2.9626, 3.2606, 3.1525, 2.9708, 3.0357]` seconds and a 3.0941-second median. Accepted active epochs had 0.9735% CV and a 2.8304-second median. The score was therefore skipped, `run.log` was never created, and the timing was not repeated.

## Results

- **Primary metric**: NaN (baseline: 94.32%; delta: N/A)
- **Observations**: Weighted loader medians were 2.8155 seconds accepted and 2.9959 seconds candidate. Differential and absolute wall projections were 369.34 and 444.41 seconds, both below 500, and counted exposure remained source-fixed at 133.00736 projected passes. The disqualifier was variability, not median cost or semantic drift.
- **Analysis**: The implementation achieved its intended local pixel intervention while preserving every audited stochastic decision. However, the active reflection transform did not demonstrate repeatable worker overlap under the preregistered protocol. The single 4.0061-second active epoch makes its median cost insufficiently trustworthy for one-shot scoring. With no evaluator run, neither the accuracy hypothesis nor the broader crop-geometry family is discredited; only this exact reflection implementation is closed under the current feasibility protocol.
- **Key Learning**: CPU-side input changes must demonstrate stable active-worker delivery even when their median wall projection and counted GPU exposure appear acceptable.

## Verification

- **Conditions**: Static scope, semantic identity, pixel confinement, batch integrity, finite data, and wall projections passed; candidate active timing CV failed at 11.1988% versus the 5% limit; scoring was not run.
- **Review Notes**: The raw timings were emitted before assertion and align with the recorded failure. There was no stale log, evaluator access, source-scope violation, or indication of infrastructure failure that would authorize a retry.
- **Verdict**: crash
- **Verdict Basis**: No primary metric was produced because the preregistered pre-score loader-stability gate failed.

## Unexplored Avenues

- Symmetric or replicate padding could have different CPU cost and boundary statistics, but neither is justified as an immediate rescue without a distinct mechanism.
- A tensor-native or batched implementation might stabilize delivery, but that would be a materially different intervention and must independently preserve RNG and counted exposure.

## Next Steps

- **High confidence**: Test the remaining batch-shared mixup strength bracket at alpha 0.1 while preserving the accepted 65% cutoff and worker pipeline.
- **Medium confidence**: Revisit checkpoint-side variance control only with a predetermined low-cost averaging mechanism that does not repeat EXP013's lagging whole-state EMA.
- **Low confidence**: Explore a non-masking conditioning mechanism on the added stage-3 capacity after establishing a concrete near-zero-cost design.
