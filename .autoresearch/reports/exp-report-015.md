# Report EXP-015: 28/56/112 First LR Drop at 23k
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-015.md
- **Plan**: plans/plan-015.md
- **Log**: logs/exp-log-015.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed benchmark harness while modifying only `train.py`. The current experiment-index baseline before this run was EXP-014 at 93.09%, and the tightened goal required at least +0.10 absolute percentage points, so EXP-015 needed `best_test_acc >= 93.19%`.

## Idea & Hypothesis
The chosen idea was to keep the successful 28/56/112 ResNet-20 from EXP-014 and move only the first LR milestone from step 22000 to 23000. The hypothesis was that 1k more high-LR exploration would improve the current model while still leaving enough LR 0.01 refinement time.

## Approach
`train.py` changed only `LR_MILESTONES = [23000, 64000]`. Architecture, optimizer, augmentation, batch size, seed, FP32 precision, channels-last, `torch.compile`, fixed time budget, and once-per-epoch evaluation were preserved. This made the experiment an isolated schedule retune against the EXP-014 baseline.

## Execution
One local single-GPU run was launched on physical GPU 0 with `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. Startup was clean on NVIDIA H20, the model reported 822,790 parameters, and no traceback, CUDA OOM, NaN/Inf, or compile failure appeared. The planned first LR drop occurred at step 23000 during epoch 59 with about 132 seconds remaining.

## Results
- **Primary metric**: 92.88% (baseline: 93.09%, delta: -0.21 points, -0.23%)
- **Observations**: The run completed 38,274 steps and 99 epochs, more than EXP-014's 34,259 steps. Despite the larger step budget, post-drop accuracy plateaued below the prior best: 92.17% at epoch 60, 92.79% by epoch 66, and a peak of 92.88% at epoch 82.
- **Analysis**: The hypothesis was not supported. Moving the first drop later gave more high-LR exploration and a larger total step count, but it delayed low-LR refinement enough that the run never approached EXP-014's 93.09% peak. For the 28/56/112 model, the 22k drop appears better calibrated than 23k.
- **Key Learning**: The 28/56/112 ResNet-20 prefers the 22k first drop; delaying to 23k increases steps but lowers peak accuracy.

## Verification
- **Conditions**: primary metric condition failed; all process and scope checks passed
- **Review Notes**: Results are trustworthy. The log has numeric final metrics, the baseline check reported 93.09, CUDA visibility showed one NVIDIA H20, the diff only touched `LR_MILESTONES`, and validation cadence remained once per epoch.
- **Verdict**: no-improvement
- **Verdict Basis**: The run was valid but `best_test_acc=92.88%` did not meet the 93.19% threshold and did not exceed the 93.09% baseline.

## Unexplored Avenues
- Test the other side of the local schedule curve with a 21k first drop; EXP-014 succeeded at 22k and EXP-015 shows 23k is too late.
- Move to a cautious width step such as 30/60/120 with an earlier first drop if schedule-only tuning around 28/56/112 is exhausted.
- Try low-overhead late averaging on 28/56/112 only if implemented without per-step overhead, since both EXP-014 and EXP-015 show late low-LR plateaus below the best observed peak.

## Next Steps
High confidence: test 21k first drop on 28/56/112 to complete the local schedule bracket around the successful 22k setting.

Medium confidence: try a modest 30/60/120 width step with a first drop around 20k if the 21k schedule probe fails.

Medium confidence: plan sparse late averaging after the first drop, with strict overhead checks, if schedule and width probes plateau.

## Exit Action Results
