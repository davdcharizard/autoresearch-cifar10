# Report EXP-042: Mild Mixup Alpha 0.1
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-042.md
- **Plan**: plans/plan-042.md
- **Log**: logs/exp-log-042.md

## Goal
Maximize CIFAR-10 `best_test_acc` under the fixed training and evaluation harness. The current experiment-index baseline is 93.97%, so EXP-042 needed a valid final `best_test_acc >= 94.07%` to count as an improvement under the +0.10 percentage-point rule.

## Idea & Hypothesis
EXP-042 tested mild batch-level mixup with `MIXUP_ALPHA = 0.1` on top of the current 2e-4 weight-decay, label-smoothed, reflection-padding anchor. The hypothesis was that light input/label interpolation could improve generalization beyond the current late plateau without changing architecture, schedule, dependencies, or evaluation.

## Approach
`train.py` was modified to add `MIXUP_ALPHA = 0.1`, print the active mixup alpha at startup, sample a beta-distributed batch mixing coefficient, permute each batch on-device, train on mixed inputs, and compute a weighted two-target cross entropy while preserving `label_smoothing=0.05`. The current anchor settings were otherwise preserved: `STAGE_WIDTHS = (28, 56, 112)`, `BATCH_SIZE = 128`, `LR = 0.1`, `LR_MILESTONES = [21000, 64000]`, `MOMENTUM = 0.9`, `WEIGHT_DECAY = 2e-4`, reflection crop padding, FP32 compile, and once-per-epoch validation.

## Execution
Two local attempts were made. Run 1 started cleanly on GPU 0 and reached a partial best of 83.49% at epoch 9, then stopped after step 4400 without a final summary or traceback. Run 2 used the same code unchanged on GPU 1, started cleanly, reached a partial best of 86.16% at epoch 14, then stopped after step 6100 without `best_test_acc`, traceback, `Killed`, `nan`, or `inf` in `run.log`.

Detached background relaunches were also tested, but this shell environment immediately reaped those child processes with empty logs. Because two real attempts failed before producing the required final metric, no third EXP-042 run was launched.

## Results
- **Primary metric**: NaN (baseline: 93.97, delta: N/A)
- **Observations**: The implementation path activated correctly (`Mixup alpha: 0.1`) and preserved the expected 822,790-parameter model and 390 batches per epoch, but neither attempt reached the first LR drop at step 21000.
- **Analysis**: EXP-042 does not provide evidence for or against mixup accuracy because the result is an execution failure, not a completed benchmark run. The partial curves show learning was finite, but they ended too early to judge the post-drop regime where the active anchor makes most of its gains.
- **Key Learning**: Mild mixup remains unproven here because local execution was interrupted twice before any final `best_test_acc` could be reported.

## Verification
- **Conditions**: failed. The run did not complete without interruption and did not report a numeric `best_test_acc`.
- **Review Notes**: Code-scope verification passed: only `train.py` changed, and the fixed harness and dependency files were untouched. Metric verification could not run because no final metric existed.
- **Verdict**: crash
- **Verdict Basis**: No valid result was produced due to interrupted local execution after two attempts; per protocol, the metric is `NaN` and the baseline remains unchanged.

## Unexplored Avenues
- Retry mixup only if the local launch failure is understood or avoided; the current result does not discredit mixup itself.
- If revisiting mixup, consider a lower-overhead variant that samples `lam` less frequently or uses an implementation with measured step overhead before committing to a full benchmark run.
- A lower initial LR remains an untested scalar alternative, but recent scalar brackets suggest it has lower upside than a distinct stable training mechanism.

## Next Steps
Use a lower-risk experiment next rather than retrying EXP-042 immediately. Medium confidence: test a simple LR 0.08 probe on the 2e-4 anchor because it is easy to execute and interpret. Medium confidence: revisit bounded late EMA only with a low-frequency implementation and explicit overhead checks. Low confidence: retry mixup after proving the run-control path can survive a full 300-second training window.

## Exit Action Results
