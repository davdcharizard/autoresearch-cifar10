# Report EXP-014: Safe Zero-Initialized Residual Endpoints
- **Created**: 2026-07-26

## Goal
Raise CIFAR-10 `best_test_acc` from accepted 94.07% to at least 94.17% within the fixed 300-second counted budget.

## Idea & Hypothesis
Zero all six final residual `conv2` weights after unchanged Kaiming initialization, making the initial network follow skip/projection paths while preserving RNG consumption, graph, parameters, and throughput. The hypothesis predicted >=135 passes and >=94.17%.

## Approach
Added a strict constructor switch and deterministic post-initialization overwrite of exactly six `PreActBlock.conv2.weight` tensors. BN scales, shortcuts, topology, optimizer, schedule, mixup, seed, data, and evaluator remained accepted. Fail-closed tests verified non-endpoint/RNG equality, initial residual semantics, positive endpoint gradients on backward one, upstream branch opening on backward two, and the fatal alternative of zeroing pre-ReLU `bn2`.

## Execution
Matched preflight retained 100.09% throughput and projected 142.03 passes with timing CVs below 0.001. One fixed-seed H20 run completed without retry: 27,892 steps / 144 epochs, 300.0 counted / 341.4 total seconds, one transition, and 29 accepted-cadence evaluations.

## Results
- **Primary metric**: 93.88% (baseline 94.07%, delta -0.19 points, -0.20%)
- **Observations**: Realized exposure was 142.80704 passes, slightly above accepted 141.9. Best and final accuracy were 93.88% at terminal epoch 144; final loss 0.2660 was worse than accepted 0.2432. Peak allocation stayed 1,094.0 MiB.
- **Analysis**: The intervention achieved the intended identity-biased startup and all branches opened normally, but both accuracy and loss regressed at normal exposure. Throughput, dead branches, RNG drift, and premature stopping are ruled out. Under the accepted warmup/LR schedule, random Kaiming residual outputs produce a better optimization/generalization trajectory than exact-zero endpoints. This closes exact all-six zero initialization, not all small residual scales or stage-selective policies; adjacent retries are not justified immediately.
- **Key Learning**: Exact zero residual endpoints preserve exposure but lose 0.19 points; accepted Kaiming residual startup is better calibrated to this schedule.

## Verification
- **Conditions**: Process integrity passed; primary metric threshold failed.
- **Review Notes**: Trustworthy one-H20, fixed-seed, one-run result with exact six endpoints, 142.81 passes, complete summary, frozen evaluator, and `train.py`-only source diff.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid 93.88% is below baseline and 0.29 below required 94.17%; no rerun.

## Unexplored Avenues
- Small nonzero endpoint scales could retain random-feature diversity, but selecting a scale after this result would be an ungrounded neighboring sweep.
- Stage-selective endpoint initialization remains distinct but lacks evidence about which stage caused the regression.

## Next Steps
- **High confidence**: return to a thorough brainstorm and seek a new training/objective mechanism rather than another capacity, averaging, or initialization neighbor.
- **Medium confidence**: examine optimizer update geometry or target formulation while preserving the validated mixup window and hard tail.
- **Low confidence**: retain SiLU and channel standardization only as exploratory fallbacks after exact design/cost work.

## Exit Action Results
No exit actions were defined.
