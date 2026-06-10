# Report EXP-016: 28/56/112 First LR Drop at 21k
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-016.md
- **Plan**: plans/plan-016.md
- **Log**: logs/exp-log-016.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed single-GPU, fixed-budget benchmark while modifying only `train.py`. The current baseline before this experiment was 93.09% from EXP-014, and the tightened success rule required at least +0.10 percentage points, so EXP-016 needed `best_test_acc >= 93.19%`.

## Idea & Hypothesis
EXP-016 tested whether the current best 28/56/112 ResNet-20 benefits from moving the first LR drop one thousand steps earlier, from 22k to 21k. EXP-015 showed 23k was too late for this width, so the hypothesis was that 21k would preserve enough LR 0.1 exploration while giving more LR 0.01 refinement time.

## Approach
The implementation changed only `LR_MILESTONES` in `train.py`, from `[22000, 64000]` to `[21000, 64000]`. `STAGE_WIDTHS = (28, 56, 112)`, depth, optimizer, augmentation, batch size, FP32 precision, compile/channels-last settings, seed, and once-per-epoch evaluation were all preserved.

## Execution
One local run was launched with `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` on a single NVIDIA H20. Startup was clean, the model reported 822,790 parameters, and the planned first drop was reached at step 21000 during epoch 54. No traceback, CUDA OOM, NaN/Inf, or compile-failure patterns were found.

## Results
- **Primary metric**: 93.23% (baseline: 93.09%, delta: +0.14 percentage points, +0.15%)
- **Observations**: The first post-drop surge reached 93.14% by epoch 60, then crossed the threshold at epoch 70 with 93.23%. Final accuracy was lower at 93.03%, so the best checkpoint still matters.
- **Analysis**: The hypothesis was supported. For the 28/56/112 width, the 21k first drop beat both the 22k baseline and the failed 23k retune, suggesting this model benefits from slightly more LR 0.01 refinement under the 300s budget.
- **Key Learning**: A 21k first drop reached 93.23%, showing the 28/56/112 model benefits from slightly more LR 0.01 refinement.

## Verification
- **Conditions**: all passed
- **Review Notes**: Results are trustworthy. The run completed normally, reported numeric final metrics, used one visible GPU, modified only `train.py`, preserved the fixed evaluator, and kept validation to one call per epoch.
- **Verdict**: improvement
- **Verdict Basis**: `best_test_acc=93.23%` is +0.14 points over the 93.09% baseline and clears the required 93.19% threshold.

## Unexplored Avenues
- Test 30/60/120 with an earlier first drop around 19k-20k to see whether width scaling still has headroom with schedule calibration.
- Bracket the new 21k anchor with a 20k drop only if future capacity changes suggest more LR 0.01 refinement is needed.
- Explore sparse late weight averaging after the 21k drop if the best/final gap persists without adding per-step overhead.

## Next Steps
Try a modest capacity increase to 30/60/120 with a first LR drop near 20k. Confidence: medium, because width scaling has been the strongest pattern but throughput will fall.

Alternatively test a lightweight late-training averaging scheme on 28/56/112 with the 21k schedule. Confidence: low-medium, because prior per-step EMA overhead was harmful, but sparse averaging may avoid that mechanism.

## Exit Action Results
- No exit actions defined.
