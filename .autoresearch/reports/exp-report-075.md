# Report EXP-075: Fan-Out Conv Init Plus Hard CutMix Endpoints
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-075.md
- **Plan**: plans/plan-075.md
- **Log**: logs/exp-log-075.md

## Goal
Maximize CIFAR-10 `best_test_acc` under the fixed harness, with higher accuracy better. The active baseline before EXP-075 was 94.11% from commit `1119ff8`; the goal's noise guard required at least +0.10 percentage points, so an improvement needed `best_test_acc >= 94.21%`.

## Idea & Hypothesis
EXP-075 tested whether the two strongest recent near-misses would compose: EXP-072's Conv2d fan-out ReLU Kaiming initialization and EXP-074's hard CutMix endpoint labels. The hypothesis was that improved residual signal scaling at initialization plus sharper mixed-batch supervision would push the CutMix anchor from 94.11% to at least 94.21%.

## Approach
Only `train.py` was modified. Conv2d weights now use `init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")`, while Linear weights keep the existing default Kaiming normal behavior. Clean batches keep label smoothing 0.05 through a new `CLEAN_LABEL_SMOOTHING` constant, while both CutMix endpoint losses use `CUTMIX_LABEL_SMOOTHING = 0.0`. CutMix alpha/probability, architecture, optimizer, schedule, transforms, seed, validation cadence, compile/channels-last, batch size, and time budget were unchanged.

## Execution
One local foreground GPU0 run completed without crashes or retries. Startup markers confirmed the intended initialization and smoothing split, unchanged parameter count 822,790, and CutMix `alpha=1.0`, `prob=0.5`. The first LR drop occurred at step 21000 with `lr: 0.0100`; post-drop accuracy climbed to 93.92% at epoch 81 and did not improve afterward.

## Results
- **Primary metric**: 93.92% (baseline: 94.11%, delta: -0.19pp, -0.20%)
- **Observations**: The run used the expected 300.0s training budget, completed 105 epochs / 40,676 steps, and matched the anchor's VRAM profile at 660.4 MB.
- **Analysis**: The coupled near-miss hypothesis failed. Instead of adding EXP-072's 94.16% and EXP-074's 94.17% signals, the combination regressed below the CutMix baseline and below both isolated variants.
- **Key Learning**: Fan-out Conv2d initialization and hard CutMix endpoint labels are not additive; combining them weakens the current CutMix anchor.

## Verification
- **Conditions**: Scope, syntax, style, implementation markers, metric availability, and hard constraints passed; the improvement threshold condition failed.
- **Review Notes**: Results are trustworthy. The diff touched only `train.py`, startup markers match the plan, no error signatures appeared, and `run.log` contains a complete numeric metric block.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid completed run, but `best_test_acc=93.92%` is below both the 94.11% baseline and the required 94.21% improvement threshold.

## Unexplored Avenues
- Classifier-specific initialization remains untested and is mechanically cleaner than more Conv2d fan-out retries, but its expected effect is small.
- A shorter CutMix warmup plus fan-out init remains possible, but EXP-073 and EXP-075 suggest temporal/target CutMix refinements are unlikely to clear 94.21% without a stronger independent lever.

## Next Steps
Try a narrow classifier initialization calibration with medium confidence; it tests a distinct unbracketed part of `_weights_init` without changing throughput. Alternatively, move away from near-miss composition and test a localized architecture or optimizer-dynamics lever with low-to-medium confidence.

## Exit Action Results
