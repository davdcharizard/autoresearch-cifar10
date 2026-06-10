# Report EXP-014: ResNet-20 Width 1.75x with 22k First Drop
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-014.md
- **Plan**: plans/plan-014.md
- **Log**: logs/exp-log-014.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed benchmark harness while modifying only `train.py`. The current experiment-index baseline before this run was EXP-013 at 92.49%, and the tightened goal required at least +0.10 absolute percentage points, so EXP-014 needed `best_test_acc >= 92.59%`.

## Idea & Hypothesis
The chosen idea was to continue the successful width-scaling path by moving from 24/48/96 channels to 28/56/112 channels, while pulling the first LR drop earlier from step 24000 to 22000. The hypothesis was that the wider, slower model would need an earlier LR 0.01 phase to preserve enough refinement time and could clear the new 92.59% threshold.

## Approach
`train.py` changed only two planned constants: `STAGE_WIDTHS = (28, 56, 112)` and `LR_MILESTONES = [22000, 64000]`. Depth, batch size, optimizer, augmentation, seed, precision, compile/channels-last settings, fixed time budget, and once-per-epoch evaluation were preserved. The second LR milestone stayed unreachable at 64000 to avoid the harmful LR 0.001 phase observed earlier.

## Execution
One local single-GPU run was launched on physical GPU 0 with `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. Startup was clean on NVIDIA H20, the model reported 822,790 parameters, and no traceback, CUDA OOM, NaN/Inf, or compile failure appeared. The first LR drop occurred at step 22000 during epoch 57 with about 123 seconds remaining, after which test accuracy climbed rapidly past the threshold.

## Results
- **Primary metric**: 93.09% (baseline: 92.49%, delta: +0.60 points, +0.65%)
- **Observations**: The run reached 34,259 steps and 88 epochs in the fixed 300s training budget. Accuracy jumped from 91.81% at epoch 57 to 92.65% by epoch 59, then peaked at 93.09% at epoch 85. Final accuracy remained high at 92.92%, so late degradation was modest.
- **Analysis**: The result supports the hypothesis. Although the wider model reduced total steps versus EXP-013, the earlier 22k first drop left enough LR 0.01 refinement time to unlock a substantially higher peak. This also clarifies EXP-012: a 22k drop was too early for 20/40/80, but it is well matched to the slower 28/56/112 model.
- **Key Learning**: Further ResNet-20 width scaling can still pay off under 300s when the first LR drop is calibrated to the slower step budget.

## Verification
- **Conditions**: all passed
- **Review Notes**: Results are trustworthy. The log has numeric final metrics, the baseline check reported 92.49 before insertion, CUDA visibility showed one NVIDIA H20, the diff only touches planned `train.py` constants, and validation cadence remains once per epoch.
- **Verdict**: improvement
- **Verdict Basis**: All verification conditions passed and 93.09% exceeds the 92.59% improvement threshold by +0.50 points.

## Unexplored Avenues
- Test whether 28/56/112 prefers a slightly later first drop such as 23000; EXP-014 succeeded, but the peak occurred late enough that more high-LR exploration may or may not help.
- Try a smaller additional width step such as 30/60/120 with an even earlier first drop if the step budget remains sufficient.
- Revisit low-frequency late averaging on the widened model, since final accuracy stayed below the peak but the gap was not severe.

## Next Steps
High confidence: explore local schedule tuning around the successful 28/56/112 model, especially 23k or 21k first drops, because schedule calibration now appears width-dependent.

Medium confidence: attempt a cautious 30/60/120 or 32/64/128 width step with a preplanned earlier first drop and explicit step-budget threshold.

Medium confidence: add sparse late averaging after the first LR drop only if the implementation avoids per-step overhead and preserves the fixed validation cadence.

## Exit Action Results
