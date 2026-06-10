# Report EXP-050: Clean Mild ColorJitter Retry
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-050.md
- **Plan**: plans/plan-050.md
- **Log**: logs/exp-log-050.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed harness while modifying only `train.py`. The active baseline was `93.97%` at commit `755be2c`, and the goal requires at least +0.10 percentage points to count, so EXP-050 needed `best_test_acc >= 94.07%`.

## Idea & Hypothesis
EXP-050 retried EXP-047's mild ColorJitter under clean GPU conditions. The hypothesis was that conservative photometric augmentation would improve color and illumination robustness if the run reached the step-21000 first LR drop that EXP-047 missed under contention.

## Approach
`train.py` gained one transform: `transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)` after reflection crop and horizontal flip, before tensor conversion and normalization. All anchor settings were preserved: `STAGE_WIDTHS=(28, 56, 112)`, batch size 128, LR 0.1, momentum 0.9, weight decay 2e-4, `LR_MILESTONES=[21000, 64000]`, label smoothing 0.05, compile, channels-last, and once-per-epoch validation.

## Execution
One local foreground run was launched on GPU1 after `nvidia-smi` showed it idle. Startup, compile, and early training were clean. The run reached the first LR drop at step 21000 with `lr: 0.0100`, unlike EXP-047, and completed normally with 41,280 steps and 525.0 total seconds.

## Results
- **Primary metric**: 93.49% (baseline: 93.97%, delta: -0.48 pp, -0.51%)
- **Observations**: Accuracy jumped to 93.05% shortly after the LR drop and peaked at 93.49% by epoch 83, then plateaued below threshold. Final accuracy was 93.03% with final loss 0.2585.
- **Analysis**: The clean retry rejects the hypothesis. Mild ColorJitter is not merely a victim of EXP-047's missed LR drop; even with clean schedule behavior and a full fixed-budget run, it remains below the current anchor and far below the 94.07% improvement threshold.
- **Key Learning**: Clean mild ColorJitter reaches the schedule milestone but still underperforms, so isolated photometric jitter is not a useful current-anchor lever.

## Verification
- **Conditions**: all run-integrity conditions passed, but the primary metric did not clear the improvement threshold.
- **Review Notes**: Results are trustworthy: the tracked diff was limited to `train.py`, compile and ruff passed, GPU1 was selected, the first LR drop occurred, final metrics were present, and total runtime stayed under 10 minutes.
- **Verdict**: no-improvement
- **Verdict Basis**: valid run with `best_test_acc=93.49%`, below both the `93.97%` baseline and the required `94.07%` threshold.

## Unexplored Avenues
- A brightness/contrast-only variant could isolate whether saturation or hue jitter is the harmful part, but the clean 93.49% result makes this lower priority.
- A much weaker photometric augmentation might preserve more late accuracy, but it also has less chance of producing the +0.10pp threshold gain.
- Policy augmentation should only be revisited if paired with a clearly different mechanism, not as another isolated transform insertion.

## Next Steps
High confidence: move away from isolated augmentation insertions and test a distinct no-overhead initialization or schedule interaction. Medium confidence: try partial residual-branch BN scaling, acknowledging EXP-028's negative prior. Medium confidence: consider a targeted optimizer/schedule interaction that preserves coupled SGD L2 and the 21k first drop.

## Exit Action Results
