# Report EXP-080: Very Short Linear LR Warmup
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-080.md
- **Plan**: plans/plan-080.md
- **Log**: logs/exp-log-080.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed single-GPU, fixed-time harness by changing only `train.py`. The current experiment-index baseline is `94.11%` at commit `1119ff8`, and the goal's noise guard requires at least `94.21%` to count as an improvement.

## Idea & Hypothesis
The chosen idea was a very short linear LR warmup: keep the validated CutMix anchor and long-run `LR=0.1` schedule, but ramp from 0.02 to 0.1 over the first 500 optimizer steps. The hypothesis was that early mixed-label updates might be slightly too aggressive, and reducing the first few hundred updates could improve post-drop convergence without weakening CutMix exposure.

## Approach
`train.py` added `LR_WARMUP_START = 0.02`, `LR_WARMUP_STEPS = 500`, and a `current_warmup_lr(step)` helper. During warmup, each optimizer param group's LR is set before the batch update; after step 500, manual LR assignment stops and the existing `MultiStepLR` owns the schedule. CutMix alpha/probability/smoothing, clean label smoothing, architecture, optimizer, milestones, transforms, batch size, seed, compile/channels-last, and validation cadence were unchanged.

## Execution
One local foreground run was launched on GPU0 with stdout/stderr captured to `run.log`. Startup confirmed the unchanged CutMix settings, `822,790` parameters, and `LR warmup: 0.02 -> 0.1 over 500 steps`. The log confirmed LR ramp behavior at steps 50, 500, and 550, and the first LR drop occurred at step 21000 in epoch 54.

## Results
- **Primary metric**: 94.08% (baseline: 94.11%, delta: -0.03pp, -0.03%)
- **Observations**: Pre-drop best reached 87.96% at epoch 33. Post-drop accuracy rose to 94.07% by epoch 63 and only nudged to 94.08% at epoch 93, then finished at 93.37%.
- **Analysis**: The warmup did not improve the CutMix anchor. It preserved schedule integrity and completed cleanly, so the result is a valid negative signal against early LR warmup as the missing stability lever.
- **Key Learning**: A 500-step LR warmup keeps the run healthy but peaks at 94.08%, so early optimizer softening is not enough to beat the CutMix anchor.

## Verification
- **Conditions**: All process and hard-constraint checks passed; the improvement threshold check failed.
- **Review Notes**: Results are trustworthy: only `train.py` changed, the run completed, metric lines are present, CutMix and schedule anchors were verified in the log, and no error signatures appeared.
- **Verdict**: no-improvement
- **Verdict Basis**: `best_test_acc=94.08%` is below the 94.11% baseline and below the required 94.21% improvement threshold.

## Unexplored Avenues
- A shorter 100-200 step warmup could reduce undertraining risk, but the 500-step result plus prior LR scalar failures makes the expected gain low.
- A coupled LR/momentum startup schedule could test a distinct optimizer transient, but isolated momentum and LR deviations have been weak historically.

## Next Steps
Try slightly weaker reflection crop jitter with medium confidence; it addresses possible over-regularization while preserving the validated CutMix and schedule anchors. Consider a small final-stage width rebalance with low confidence if augmentation micro-tuning fails, but architecture changes remain a weak family.

## Exit Action Results
