# Report EXP-028: Zero-Initialize Residual Branch Last BatchNorm
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-028.md
- **Plan**: plans/plan-028.md
- **Log**: logs/exp-log-028.md

## Goal
EXP-028 targeted higher CIFAR-10 `best_test_acc` under the fixed evaluation harness and fixed 300s training budget. The current experiment-index baseline was 93.23% from commit `f187edf`, so the goal's +0.10 percentage-point rule required at least 93.33% to count as an improvement.

## Idea & Hypothesis
The chosen idea was to zero-initialize each residual block's final BatchNorm scale (`BasicBlock.bn2.weight`) so residual branches initially behave closer to identity mappings. The hypothesis was that this identity-preserving initialization would improve optimization stability and post-drop refinement without changing throughput, parameter count, architecture, schedule, or evaluation.

## Approach
`train.py` added a post-initialization loop in `ResNet.__init__` after `self.apply(self._weights_init)`. For every `BasicBlock`, it applies `init.constant_(m.bn2.weight, 0)`. All anchor hyperparameters and implementation choices were preserved: `STAGE_WIDTHS=(28,56,112)`, batch size 128, LR 0.1, momentum 0.9, weight decay `1e-4`, milestones `[21000,64000]`, FP32 compile/channels-last, augmentation, optimizer, seed, and once-per-epoch validation.

## Execution
One local single-GPU run was launched on GPU 0 with stdout/stderr captured to `run.log`. Startup was clean, CUDA saw one NVIDIA H20, the expected 822,790-parameter model was used, `Batches per epoch: 390` confirmed the batch size was preserved, and the first LR drop fired at step 21000 with `lr=0.0100`. The run completed normally with no traceback, OOM, NaN, or Inf patterns.

## Results
- **Primary metric**: 91.74% (baseline: 93.23%, delta: -1.49 points, -1.60%)
- **Observations**: Pre-drop learning lagged the current anchor trajectory, reaching only 87.27% before the first drop. The step-21000 drop produced an immediate jump to 90.12%, but post-drop refinement plateaued around 91.7% and never approached the baseline.
- **Analysis**: The hypothesis was rejected for this fixed-budget recipe. Zeroing the residual branch terminal BatchNorm scale preserved throughput and constraints, but the identity-biased initialization appears to slow useful representation learning too much for the 300s window. This result does not discredit residual-identity initialization generally; it says the isolated zero-gamma variant is a poor fit for the current CIFAR-10 anchor and schedule.
- **Key Learning**: Zero-initializing residual branch BatchNorm severely slowed useful learning under the fixed budget, plateauing at 91.74%.

## Verification
- **Conditions**: Process, implementation, schedule, parameter-count, and hard-constraint checks passed; the metric improvement condition failed.
- **Review Notes**: Results are trustworthy. The run completed successfully, reported numeric metrics, modified only `train.py`, preserved once-per-epoch validation, hit the step-21000 LR drop, preserved `num_params=822,790`, and finished in 389.9 total seconds.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result, but `best_test_acc=91.74%` is below the 93.23% baseline and the required 93.33% improvement threshold.

## Unexplored Avenues
- A partial residual scale initialization such as `bn2.weight=0.1` might retain some identity bias without fully zeroing residual branches, but the full zero-gamma result makes this lower priority.
- A coupled schedule change with a later first LR drop might compensate for slower early learning, but schedule-only variants have already looked bounded.
- Reflection padding remains a separate augmentation-quality change that does not inherit this initialization failure mode.

## Next Steps
Medium confidence: test reflection padding for `RandomCrop`, because it is a small augmentation-quality change with no expected throughput or parameter-count cost.

Low confidence: test low-frequency late EMA only if implemented after the first LR drop with sparse updates to avoid EXP-004's per-step overhead and EXP-021's long-window averaging collapse.

Low confidence: revisit residual branch initialization only as partial scaling, not full zero initialization, and only if simpler no-overhead levers are exhausted.

## Exit Action Results
