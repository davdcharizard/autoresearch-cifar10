# Report EXP-004: Clean-Finish Periodic SAM
- **Created**: 2026-08-05

## Goal

Increase CIFAR-10 `best_test_acc` under the frozen 300-second charged training budget by adding a qualitatively different generalization mechanism to EXP-002. The parent and prior global best were EXP-002 at 95.23%; improvement required at least 95.33%. EXP-004 becomes the new accepted global best at 95.40%.

## Idea & Hypothesis

Apply plain SAM only in the final clean quarter, after front-loaded CutMix ends, to improve generalization from an already well-fitted solution without paying the cost of full-run SAM. Claude's randomized idea review selected SAM over EMA and clean-only label smoothing because its demonstrated effect ceiling was large enough to exceed the 0.14-0.29-point selection variability observed in EXP-003. The preregistered hypothesis used rho 0.05 every second eligible step and required at least 95.33% with at least 24,000 total steps.

## Approach

Only `train.py` changed. The first 75% remains the exact EXP-002 WRN/CutMix path. At `progress >= 0.75`, every even upcoming one-based step performs a normal forward/backward, computes a global FP32 gradient norm, snapshots and perturbs parameters, replays CUDA RNG, disables BatchNorm running-stat tracking for a separately autocast second pass, restores BatchNorm flags and exact parameter snapshots, then performs the sole Nesterov update. All extra work is inside the charged timer. Preallocated snapshots avoid allocation churn, and final counters record eligibility, application ratio, first step, and first progress.

## Execution

One run launched on physical GPU 0 with a 600-second timeout and fixed seed 42. Before launch, tiny-model and full WRN BF16/channels-last smokes verified the 0.05 perturbation norm, distinct perturbed loss, CUDA RNG parity, one BatchNorm buffer update, exact parameter restoration, and one momentum update. Claude's plan review drove separate autocast contexts, overhead-inclusive exposure bounds, stage-aware restoration, and a complete audit contract. The run completed exit 0 without retry or metric-driven adjustment.

## Results

- **Primary metric**: 95.40% (parent: 95.23%, delta vs parent: +0.17 points, +0.18%; global best: 95.40%)
- **Observations**: Final accuracy equaled the best accuracy at epoch 132, so success did not depend on an isolated checkpoint. Final loss improved from the parent's 0.2044 to 0.1654 despite completing 25,560 steps versus 27,950 for EXP-002. SAM applied on exactly 2,449 of 4,898 eligible batches, first at step 20,664 and progress 0.7500. CutMix exposure remained 0.4962 and froze at the same transition. Peak allocation increased only 11.6 MiB, from 1,178.9 to 1,190.5 MiB.
- **Analysis**: The hypothesis was validated. Period-two late SAM supplied enough flatness-aware optimization to overcome an 8.6% reduction in optimizer steps and clear the parent-relative noise margin. The improved final loss and final-equals-best trajectory support a genuine better late solution rather than checkpoint selection. The result specifically validates a clean-tail periodic dose; it does not establish that full SAM, a different rho, or scalar cadence tuning would improve further.
- **Key Learning**: Period-two SAM in the clean final quarter adds 0.17 points while preserving 25,560 optimizer steps and exact parent RNG semantics.

## Verification

- **Conditions**: All passed. Accuracy reached 95.40% versus the 95.33% threshold; the run completed at 300.0 charged seconds and 457.3 total seconds with a complete summary.
- **Review Notes**: Physical GPU 0 was the 97,871 MiB H20; 132 evaluations occurred for 132 epochs; only `train.py` changed; model size stayed 2,748,890; fixed seed and evaluator were unchanged. Claude's post-run adversarial review independently approved log freshness, exact cadence arithmetic, scope, timing, RNG replay, BatchNorm handling, restoration, and live BF16 perturbation.
- **Verdict**: improvement
- **Verdict Basis**: Every hard constraint and necessary condition passed, and the 0.17-point gain exceeded the required 0.10-point parent-relative margin.

## Unexplored Avenues

- Reuse a sharpness-aware gradient component between occasional full SAM pulses, as in LookSAM-style methods, to increase flatness bias without another full second pass on every eligible pulse.
- Test a broader or earlier SAM phase only if its extra effect can justify the corresponding loss of optimizer exposure; full-run SAM remains infeasible under this budget.
- Combine the validated SAM tail with an orthogonal representation or augmentation change rather than tuning rho or cadence narrowly against the test metric.

## Next Steps

- **High confidence**: Grow from EXP-004 and add an orthogonal low-cost representation or augmentation mechanism while preserving the validated SAM tail.
- **Medium confidence**: Explore a compute-reuse sharpness method that strengthens SAM exposure without materially reducing the 25,560-step horizon.
- **Medium confidence**: Revisit channel allocation or lightweight attention using the remaining H20 memory headroom, leaving the validated timing and RNG structure intact.
