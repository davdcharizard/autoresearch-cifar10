# Report EXP-026: Momentum 0.95 on Current Anchor
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-026.md
- **Plan**: plans/plan-026.md
- **Log**: logs/exp-log-026.md

## Goal
EXP-026 targeted higher CIFAR-10 `best_test_acc` under the fixed harness and fixed 300s training budget. The current experiment-index baseline was 93.23% from commit `f187edf`, so the goal's +0.10 percentage-point rule required at least 93.33% to count as an improvement.

## Idea & Hypothesis
The chosen idea was to raise the classical SGD momentum coefficient from `0.9` to `0.95` on the current 28/56/112 anchor. The hypothesis was that stronger velocity smoothing would improve post-drop refinement without reducing throughput, adding capacity, or adding explicit regularization.

## Approach
`train.py` changed only `MOMENTUM` from `0.9` to `0.95`. Architecture, batch size, learning rate, weight decay, LR milestones, optimizer class, augmentation, FP32 compile/channels-last path, fixed training budget, and once-per-epoch validation were preserved.

## Execution
One local single-GPU run was launched on GPU 0 with stdout/stderr captured to `run.log`. Startup was clean, CUDA saw one NVIDIA H20, the expected 822,790-parameter model was used, `Batches per epoch: 390` confirmed the batch size was preserved, and the first LR drop fired at step 21000 with `lr=0.0100`. The run completed normally with no traceback, OOM, NaN, or Inf patterns.

## Results
- **Primary metric**: 92.90% (baseline: 93.23%, delta: -0.33 points, -0.35%)
- **Observations**: Pre-drop accuracy was noisy and lower than the current anchor trajectory, with best only 87.44% before the first drop. After the step-21000 drop, accuracy rose quickly to 92.90% by epoch 72, then plateaued and drifted downward through the final epoch.
- **Analysis**: The hypothesis was rejected. Higher momentum did not improve late refinement; it produced a lower post-drop plateau and worse final accuracy than the current anchor. Since throughput and schedule were preserved, the result points to optimizer dynamics rather than a runtime or reachability failure.
- **Key Learning**: Momentum 0.95 made post-drop refinement worse, peaking at 92.90%, so the anchor should keep classical momentum 0.9.

## Verification
- **Conditions**: Process, schedule, and hard-constraint checks passed; the metric improvement condition failed.
- **Review Notes**: Results are trustworthy. The run completed successfully, reported numeric metrics, modified only `train.py`, preserved once-per-epoch validation, hit the step-21000 LR drop, and finished in 390.3 total seconds.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result, but `best_test_acc=92.90%` is below the 93.23% baseline and the required 93.33% improvement threshold.

## Unexplored Avenues
- A smaller momentum adjustment such as 0.92 could test whether 0.95 overshot, but the large drop to 92.90% makes this a low-priority follow-up.
- Coupled momentum and LR retuning might recover stability, but it would be a broader optimizer-search experiment rather than an isolated coefficient test.
- Standard CIFAR channel-std normalization remains a distinct input-conditioning lever, but it may require careful LR handling.

## Next Steps
Medium confidence: test standard CIFAR channel-std normalization with the current anchor, because it is a distinct conditioning change not covered by capacity, schedule, batch-size, or momentum failures.

Low confidence: test isolated mild label smoothing only if input-conditioning changes fail, since explicit regularization has repeatedly underperformed.

Low confidence: consider a conservative LR coefficient change only after exhausting cleaner one-factor changes.
