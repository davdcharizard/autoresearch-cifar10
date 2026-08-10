# Report EXP-013: Batch-256 Linear Scaling
- **Created**: 2026-08-06

## Goal

Increase CIFAR-10 `best_test_acc (%)` from the 94.15% moving baseline at `7c1e7d8`. A valid improvement required at least 94.25% under the fixed one-H20, seed-42, 300-second training protocol.

## Idea & Hypothesis

Use otherwise idle H20 parallelism by doubling training batch size to 256 and scaling the complete LR curve from `0.1/0.01/1e-4` to `0.2/0.02/2e-4`. Linear scaling approximately preserves update and coupled-decay displacement per dataset pass. The hypothesis required at least 20% more image slots in the fixed budget and predicted that extra exposure would raise accuracy to at least 94.25% despite fewer optimizer decisions.

## Approach

Implemented exactly four training constants on the accepted EXP-010 model and recipe. Mandatory Claude plan review found that 195-step candidate epochs would otherwise create 22-23 evaluations versus EXP-010's 19 and bias max-based `best_test_acc`; the plan therefore added an accuracy-independent fixed 19-point elapsed-progress schedule as measurement control. Functional checks preserved model/RNG state, parameters, optimizer semantics, target paths, LR boundaries, and unique evaluation epochs. A mandatory five-pair fresh-process H20 gate tested the throughput premise before production-loader work or an accuracy run.

## Execution

Mandatory external Claude idea and plan reviews completed successfully with no fallback reviewer. The compiler and zero-gamma seeds were retired during brainstorming for environment incompatibility and Option-A dead channels, respectively. EXP-013's functional gate passed. Five alternating fresh-process pairs each ran 100 warmups plus 500 synchronized hard/soft steps. The throughput gate failed stably, so the reviewed abort criterion prevented the full seed-42 run; no threshold, batch, LR, precision, or performance fallback was used.

## Results

- **Primary metric**: `NaN` (baseline: `94.15%`; no accuracy run)
- **Observations**: Control/candidate median trial means were 10.8437/18.2380 ms, a 1.68189 step ratio. Batch 256 delivered 1.18914x image throughput and projected 15,992 steps / 4,093,952 slots, below the 1.20x and 4,131,000 gates. P95 image throughput passed at 1.18851x against a 1.15x floor; control/candidate CVs were only 0.474%/0.197%; memory was 598.7/1,120.2 MB. The stable fresh-pair result superseded an earlier serial probe that had estimated 1.2844x exposure.
- **Analysis**: The candidate missed its mechanism threshold by 1.09 percentage points of image throughput. That is too small a difference to claim batch 256 is intrinsically incapable of improving accuracy, but the experiment pre-registered 20% because the method gives up roughly 40% of optimizer decisions and changes gradient/BN noise. Launching after the stable miss would have converted a failed systems premise into an unreviewed accuracy gamble. The result also validates two protocol lessons: fresh alternating processes are necessary for batch comparisons, and batch-dependent epochs must not silently increase the number of test-set looks for a max metric.
- **Key Learning**: Fresh pairs measured only 18.91% more image throughput, below the 20% floor; batch 256 did not justify its optimizer-update loss.

## Verification

- **Conditions**: Functional/scope checks passed. Mandatory paired throughput failed; production-loader, wall, full-run, and primary-accuracy conditions were not reached.
- **Review Notes**: Timing results are trustworthy: the H20 was uncontended, five alternating fresh pairs were highly stable, and all numerical/memory checks passed. No primary result exists, so no accuracy comparison or no-improvement claim is possible.
- **Verdict**: invalid
- **Verdict Basis**: Partial feasibility evidence only; the pre-registered mechanism gate correctly blocked the accuracy run, leaving `best_test_acc` unavailable.

## Unexplored Avenues

- Batch 192 may retain more optimizer updates while still improving image throughput, but it needs a new paired knee measurement and a separately reviewed 1.5x LR rule.
- Batch 512 processes slightly more images than 256 but retains only about half as many updates; current evidence does not justify that accuracy gamble.
- BF16/autocast or channels-last could attack convolution backward without changing batch noise, but each changes numerics/data layout and requires independent review and timing.
- Batch 256 could still improve accuracy at 18.91% extra exposure, but relaxing a stable pre-registered gate after measurement would be invalid; revisit only as a new hypothesis with independent justification.

## Next Steps

- **Medium confidence**: return to the fixed average-max pooling finalist, whose mechanism changes representation readout without sacrificing update count.
- **Medium confidence**: brainstorm a backward-performance mechanism compatible with Python 3.14 that preserves batch-128 optimization noise.
- **Low confidence**: measure batch 192 only if adversarial review finds its update/exposure balance stronger than the current representation candidates.

## Exit Action Results

- None defined.
