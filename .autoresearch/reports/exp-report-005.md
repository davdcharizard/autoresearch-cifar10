# Report EXP-005: Isolated Cutout
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-005.md
- **Plan**: plans/plan-005.md
- **Log**: logs/exp-log-005.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed `prepare.py` evaluation harness and 300 second training budget. The current baseline is EXP-002 at 91.95%, and the tightened goal requires at least +0.10 absolute percentage points, so this experiment needed `best_test_acc >= 92.05%`.

## Idea & Hypothesis
EXP-005 tested isolated 16x16 cutout on the validated FP32 compile/channels-last ResNet-20 recipe. The hypothesis was that cutout could improve generalization enough to clear 92.05% while avoiding EXP-000's confounds from label smoothing, Nesterov, and a slow cosine schedule.

## Approach
`train.py` added explicit cutout constants and inserted `transforms.RandomErasing` into the training transform after normalization. Model architecture, optimizer, LR milestones `[32000, 48000]`, batch size, seed, FP32 arithmetic, `torch.compile`, channels-last, and once-per-epoch evaluation were preserved.

## Execution
Run 1 was aborted before measurement because physical GPU 0 had unrelated activity. Run 2 relaunched on physical GPU 1 and completed cleanly with output captured to `run.log`; the temporary log was removed after metrics were copied into `logs/exp-log-005.md`.

## Results
- **Primary metric**: 91.72% (baseline: 91.95%, delta: -0.23 points, -0.25%)
- **Observations**: Pre-drop accuracy lagged badly, reaching only 87.37% by epoch 82. After the first LR drop at step 32000, accuracy recovered to a best of 91.72% at epoch 110, then plateaued below baseline.
- **Analysis**: The hypothesis failed. Cutout did not materially reduce optimizer steps (`num_steps=46238`), so the miss is not primarily throughput overhead. The issue is convergence/regularization strength under the fixed budget: 16x16 masking delays useful fitting and never repays that delay with better generalization.
- **Key Learning**: Isolated 16x16 cutout preserves throughput but regularizes too strongly for this fixed-budget ResNet-20 recipe.

## Verification
- **Conditions**: primary metric condition failed.
- **Review Notes**: Results are trustworthy: the run completed cleanly, produced numeric metrics, and the failed condition is an ordinary metric miss rather than infrastructure noise.
- **Verdict**: no-improvement
- **Verdict Basis**: `best_test_acc=91.72%` is below both the 91.95% baseline and the required 92.05% threshold.

## Unexplored Avenues
- Smaller or lower-probability cutout could retain some regularization while reducing the long pre-drop underfitting seen here.
- RandAugment-style light augmentation remains untested, but only if it can be implemented inside `train.py` without dependency or evaluation changes.

## Next Steps
- High confidence: try a higher-ceiling architecture change such as a modest ResNet depth/width increase with schedule calibration.
- Medium confidence: test a schedule calibrated to the observed 43k-46k step horizon without adding regularization.
- Low confidence: retry cutout only with much weaker parameters, since the isolated 16x16 variant already missed by a large margin.

## Exit Action Results
