# Report EXP-076: Xavier Classifier Init With Zero Bias
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-076.md
- **Plan**: plans/plan-076.md
- **Log**: logs/exp-log-076.md

## Goal
Maximize CIFAR-10 `best_test_acc` under the fixed harness, with higher accuracy better. The active baseline before EXP-076 was 94.11% from commit `1119ff8`; the goal's noise guard required at least +0.10 percentage points, so an improvement needed `best_test_acc >= 94.21%`.

## Idea & Hypothesis
EXP-076 tested whether the final classifier head had a narrow initialization mismatch. The existing `_weights_init` applied Kaiming normal to both Conv2d and Linear weights, even though the final Linear layer emits logits and is not followed by a ReLU. The hypothesis was that Xavier classifier weights plus zero bias would improve initial logit calibration enough to push the CutMix anchor from 94.11% to at least 94.21%.

## Approach
Only `train.py` was modified. `_weights_init` was split so Conv2d kept the current default Kaiming normal initialization, while Linear used `init.xavier_uniform_(m.weight)` plus `init.zeros_(m.bias)` when a bias exists. A startup marker, `Classifier init: xavier_uniform weight, zero bias`, was added. CutMix alpha/probability, label smoothing, architecture, optimizer, schedule, transforms, seed, validation cadence, compile/channels-last, batch size, parameter count, and time budget were unchanged.

## Execution
One local foreground GPU0 run completed without crashes or retries. Startup markers confirmed the intended classifier initialization, unchanged parameter count 822,790, and CutMix `alpha=1.0`, `prob=0.5`, `label smoothing=0.05`. The first LR drop occurred at step 21000 with `lr: 0.0100`; post-drop accuracy climbed to 93.73% at epoch 83 and did not improve afterward.

## Results
- **Primary metric**: 93.73% (baseline: 94.11%, delta: -0.38pp, -0.40%)
- **Observations**: The run used the expected 300.0s training budget, completed 101 epochs / 39,345 steps, and matched the anchor's VRAM profile at 660.4 MB.
- **Analysis**: The classifier-calibration hypothesis failed. The run was clean and reached the LR drop, but Xavier/zero-bias initialization produced a lower plateau than both the 94.11% CutMix anchor and recent near-misses from fan-out Conv2d initialization, clean warmup, and hard CutMix endpoints.
- **Key Learning**: Classifier-specific Xavier/zero-bias initialization weakens the current CutMix anchor; final-head initialization is not the missing calibration lever.

## Verification
- **Conditions**: Scope, syntax, style, implementation markers, metric availability, and hard constraints passed; the improvement threshold condition failed.
- **Review Notes**: Results are trustworthy. The diff touched only `train.py`, startup markers match the plan, no error signatures appeared, and `run.log` contains a complete numeric metric block.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid completed run, but `best_test_acc=93.73%` is below both the 94.11% baseline and the required 94.21% improvement threshold.

## Unexplored Avenues
- Bias-free classifier remains technically untested, but EXP-061 and EXP-076 now make isolated classifier-head changes lower priority.
- A scheduled or coupled classifier-head change could still be explored only if a future experiment identifies late overfitting in the head; this run does not support isolated head calibration.

## Next Steps
Move away from final-head-only changes with medium confidence. Better candidates are distinct localized architecture changes that preserve the CutMix anchor, or a carefully justified optimizer-dynamics change that does not repeat the already-failed scalar LR and weight-decay brackets.

## Exit Action Results
