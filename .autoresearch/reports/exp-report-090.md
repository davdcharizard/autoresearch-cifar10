# Report EXP-090: Fine Lower CutMix Probability p=0.4 on Spatial Anchor
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-090.md
- **Plan**: plans/plan-090.md
- **Log**: logs/exp-log-090.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed benchmark harness, modifying only `train.py`. The active baseline before EXP-090 was 94.51% at commit `83d4e94`, and the goal's +0.10 percentage-point noise guard required `best_test_acc >= 94.61%` to count as an improvement.

## Idea & Hypothesis
EXP-090 tested a fine lower CutMix probability bracket on the current spatial anchor. The chosen idea was to keep reflection crop padding 3, `RandomHorizontalFlip(p=0.4)`, and the rest of the validated recipe intact while reducing `CUTMIX_PROB` from 0.5 to 0.4.

The hypothesis was that the padding-3 / flip-p=0.4 anchor might now be slightly over-regularized by applying CutMix to half of batches, so a smaller p=0.4 probability could retain regional mixing while reducing mixed-label pressure enough to reach at least 94.61%.

## Approach
The implementation changed only `train.py`: `CUTMIX_PROB = 0.5` became `CUTMIX_PROB = 0.4`.

All other anchor settings were preserved: `CUTMIX_ALPHA=1.0`, CutMix endpoint label smoothing 0.05, clean-batch label smoothing 0.05, reflection crop padding 3, `RandomHorizontalFlip(p=0.4)`, unit-std normalization, `STAGE_WIDTHS=(28, 56, 112)`, `WEIGHT_DECAY=2e-4`, `LR=0.1`, `LR_MILESTONES=[21000, 64000]`, batch size 128, seed 42, FP32 compile/channels-last, fixed 300s budget, and once-per-epoch validation.

## Execution
One local attached single-GPU run was launched on GPU0 with output captured to `run.log`. Startup confirmed CUDA execution, the intended CutMix probability 0.4 setting, reflection crop padding 3, flip p=0.4, 822,790 parameters, and the 300s training budget.

The first LR drop was reached at step 21000 with `lr: 0.0100`. Pre-drop best was 88.67% through epoch 53; post-drop convergence reached 93.35% by epoch 59 and peaked at 94.13% by epoch 87. The run completed cleanly with no error signatures.

## Results
- **Primary metric**: 94.13% (baseline: 94.51%, delta: -0.38pp, -0.40%)
- **Observations**: The run preserved schedule integrity and completed 40,415 steps over 104 epochs, with peak VRAM 660.4 MB and 822,790 parameters. Extra step coverage did not translate into a higher peak.
- **Analysis**: The hypothesis was not supported. Reducing CutMix frequency to p=0.4 under the stronger spatial anchor underperformed the p=0.5 baseline by a large margin. Combined with EXP-065 and EXP-066, this promotes CutMix probability moves away from p=0.5 to a high-importance failed direction.
- **Key Learning**: CutMix probability p=0.4 under the spatial anchor peaked at 94.13%, reinforcing p=0.5 as the best tested CutMix frequency.

## Verification
- **Conditions**: All integrity and execution conditions passed; the improvement-threshold condition failed for improvement classification.
- **Review Notes**: Results are trustworthy. Only `train.py` changed, syntax and ruff checks passed, startup markers matched the plan, the LR drop occurred at step 21000, the run produced numeric metrics, and no crash/error signatures appeared.
- **Verdict**: no-improvement
- **Verdict Basis**: EXP-090 produced a valid metric but did not exceed the 94.51% baseline or the 94.61% noise-guard threshold.

## Unexplored Avenues
- A fine CutMix alpha change under the spatial anchor remains technically untested, but alpha brackets are already a medium-importance failed family and should not be prioritized before more distinct mechanisms.
- A scheduled CutMix probability change is distinct from static p=0.4, but early and late CutMix schedules have already underperformed; any retry needs a stronger coupled rationale.

## Next Steps
- Medium confidence: test a fine lower CutMix alpha such as 0.75 only if the loop chooses to close the last remaining CutMix strength bracket on the spatial anchor.
- Medium confidence: test a lower weight-decay bracket such as 1.75e-4 only as a scalar diagnostic; prior evidence favors keeping `2e-4`.
- Low confidence: explore a distinct late-training mechanism that preserves validation cadence and the 21k LR drop, avoiding direct schedule-only, averaging, label-smoothing, and batch-size retries.

## Exit Action Results
