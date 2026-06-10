# Report EXP-092: Fine Lower Weight Decay 1.75e-4 on Spatial Anchor
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-092.md
- **Plan**: plans/plan-092.md
- **Log**: logs/exp-log-092.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed benchmark harness, modifying only `train.py`. The active baseline before EXP-092 was 94.51% at commit `83d4e94`, and the goal's +0.10 percentage-point noise guard required `best_test_acc >= 94.61%` to count as an improvement.

## Idea & Hypothesis
EXP-092 tested a fine lower weight-decay bracket on the current padding-3 / flip-p=0.4 spatial anchor. The hypothesis was that `WEIGHT_DECAY=2e-4` might be slightly too strong after the spatial de-regularization improvements, so reducing it to `1.75e-4` could improve generalization without reopening closed CutMix or spatial brackets.

## Approach
The implementation changed only `train.py`: `WEIGHT_DECAY = 2e-4` became `WEIGHT_DECAY = 1.75e-4`. CutMix alpha/probability/label smoothing, clean label smoothing, reflection crop padding 3, flip p=0.4, unit-std normalization, architecture, optimizer type, LR milestones, batch size, seed, compile/channels-last, fixed budget, and validation cadence were preserved.

## Execution
One local attached single-GPU run was launched on GPU0. Startup confirmed CUDA, `RandomCrop padding: 3 reflect`, `RandomHorizontalFlip p: 0.4`, 822,790 parameters, unchanged CutMix settings, and the 300s budget. The first LR drop occurred at step 21000 with `lr: 0.0100`, and the run completed cleanly with no error signatures.

## Results
- **Primary metric**: 94.14% (baseline: 94.51%, delta: -0.37pp, -0.39%)
- **Observations**: The run reached 39,173 steps over 101 epochs with peak VRAM 660.4 MB. Pre-drop best was 88.77%, early post-drop accuracy reached 93.44% by epoch 57, and the late best reached only 94.14% at epoch 91.
- **Analysis**: The hypothesis was not supported. Lowering weight decay to 1.75e-4 worsened the spatial/CutMix anchor, consistent with EXP-041's lower-decay failure and the broader pattern that `2e-4` is the useful decay anchor.
- **Key Learning**: Lowering weight decay to 1.75e-4 peaked at 94.14%, so the spatial/CutMix recipe still wants `WEIGHT_DECAY=2e-4`.

## Verification
- **Conditions**: All integrity and execution conditions passed; the improvement-threshold condition failed for improvement classification.
- **Review Notes**: Results are trustworthy. Only `train.py` changed, syntax and ruff checks passed, startup markers matched the plan, the LR drop occurred at step 21000, metrics were numeric, and no error signatures appeared.
- **Verdict**: no-improvement
- **Verdict Basis**: EXP-092 produced a valid metric but did not exceed the 94.51% baseline or the 94.61% noise-guard threshold.

## Unexplored Avenues
- None identified for isolated lower weight decay. Both 1.5e-4 and 1.75e-4 have now failed, while stronger decay above 2e-4 also failed, so future decay work should require a distinct coupled mechanism rather than another scalar bracket.

## Next Steps
- Medium confidence: test higher BatchNorm momentum 0.2 on the spatial anchor, because it is distinct from the now-closed scalar decay, spatial, and CutMix brackets.
- Low confidence: test lower classical momentum 0.85 only if BatchNorm-state dynamics fail, because optimizer transient evidence is weaker.
- Low confidence: explore a coupled late-training mechanism only if it avoids schedule-only, averaging, label-smoothing, and batch-size retry patterns.

## Exit Action Results
