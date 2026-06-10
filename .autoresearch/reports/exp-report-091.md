# Report EXP-091: Fine Lower CutMix Alpha 0.75 on Spatial Anchor
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-091.md
- **Plan**: plans/plan-091.md
- **Log**: logs/exp-log-091.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed benchmark harness, modifying only `train.py`. The active baseline before EXP-091 was 94.51% at commit `83d4e94`, and the goal's +0.10 percentage-point noise guard required `best_test_acc >= 94.61%` to count as an improvement.

## Idea & Hypothesis
EXP-091 tested a fine lower CutMix alpha bracket on the current spatial anchor. The hypothesis was that changing the CutMix patch-area distribution from alpha 1.0 to 0.75 might reduce excessive regional-mixing pressure while preserving the p=0.5 application frequency.

## Approach
The implementation changed only `train.py`: `CUTMIX_ALPHA = 1.0` became `CUTMIX_ALPHA = 0.75`. CutMix probability 0.5, CutMix endpoint smoothing 0.05, clean smoothing 0.05, padding 3, flip p=0.4, architecture, optimizer, schedule, seed, compile/channels-last, fixed budget, and validation cadence were preserved.

## Execution
One local attached single-GPU run was launched on GPU0. Startup confirmed CUDA, `RandomCrop padding: 3 reflect`, `RandomHorizontalFlip p: 0.4`, 822,790 parameters, `CutMix alpha: 0.75, prob: 0.5, label smoothing: 0.05`, and the 300s budget. The first LR drop occurred at step 21000 with `lr: 0.0100`, and the run completed cleanly.

## Results
- **Primary metric**: 94.34% (baseline: 94.51%, delta: -0.17pp, -0.18%)
- **Observations**: The run reached 39,540 steps over 102 epochs with peak VRAM 660.4 MB. It peaked at 94.34% at epoch 81 but did not approach the 94.61% threshold.
- **Analysis**: The hypothesis was not supported. Alpha 0.75 under the stronger spatial anchor still underperformed alpha 1.0, reinforcing the older alpha 0.5 and 2.0 failures.
- **Key Learning**: CutMix alpha 0.75 under the spatial anchor peaked at 94.34%, reinforcing alpha 1.0 as the best tested patch-area setting.

## Verification
- **Conditions**: All integrity and execution conditions passed; the improvement-threshold condition failed for improvement classification.
- **Review Notes**: Results are trustworthy. Only `train.py` changed, syntax and ruff checks passed, startup markers matched the plan, the LR drop occurred at step 21000, metrics were numeric, and no error signatures appeared.
- **Verdict**: no-improvement
- **Verdict Basis**: EXP-091 produced a valid metric but did not exceed the 94.51% baseline or the 94.61% noise-guard threshold.

## Unexplored Avenues
- None identified for static CutMix alpha. Broad and fine alpha brackets have now failed, so future CutMix work should require a distinct scheduled or coupled mechanism.

## Next Steps
- Medium confidence: test the remaining lower weight-decay bracket 1.75e-4 on the spatial anchor as a final scalar regularization diagnostic.
- Low confidence: test higher BatchNorm momentum 0.2 if scalar diagnostics are exhausted, because it is distinct but weaker-evidence.
- Low confidence: explore a late-training mechanism only if it avoids direct schedule-only, averaging, label-smoothing, and batch-size retries.

## Exit Action Results
