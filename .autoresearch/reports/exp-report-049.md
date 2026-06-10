# Report EXP-049: Decoupled SGD Weight Decay at 2e-4
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-049.md
- **Plan**: plans/plan-049.md
- **Log**: logs/exp-log-049.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed harness while modifying only `train.py`. The active baseline was `93.97%` at commit `755be2c`, and the goal requires at least +0.10 percentage points to count, so EXP-049 needed `best_test_acc >= 94.07%`.

## Idea & Hypothesis
The chosen idea was to keep the successful `WEIGHT_DECAY = 2e-4` magnitude from EXP-038 but change its semantics from optimizer-coupled SGD L2 decay to manual decoupled multiplicative shrinkage after each SGD step. The hypothesis was that preserving shrinkage while removing gradient and momentum-buffer coupling could improve late generalization enough to clear `94.07%`.

## Approach
`train.py` disabled SGD's built-in `weight_decay`, added `apply_decoupled_weight_decay(params, lr)`, captured all trainable parameters after optional compile, and applied multiplicative decay after `optimizer.step()` using the LR active for that update. All anchor settings were preserved: model widths, batch size, LR 0.1, momentum 0.9, LR milestones, reflection crop padding, label smoothing 0.05, compile, and channels-last.

## Execution
One local foreground run was launched on GPU1 because GPU0 had an unrelated active run. Preflight compile and ruff checks passed, startup was clean, and the run reached the first LR drop at step 21000 with `lr: 0.0100`. The process exited normally within budget after 404.0 total seconds.

## Results
- **Primary metric**: 93.06% (baseline: 93.97%, delta: -0.91 pp, -0.97%)
- **Observations**: The run completed 40,437 steps and 104 epochs with normal throughput and `peak_vram_mb=660.4`. Accuracy climbed after the first LR drop but plateaued around 92.8-93.06 rather than approaching the anchor.
- **Analysis**: The hypothesis was rejected. Decoupled decay preserved runtime behavior but weakened the recipe, which suggests the coupled SGD L2 interaction with gradients and momentum is part of why the current `2e-4` anchor works.
- **Key Learning**: Keep coupled SGD L2 decay for the current anchor; decoupling `2e-4` shrinkage under the same schedule loses nearly a point.

## Verification
- **Conditions**: all necessary validity conditions passed, but the primary metric did not clear the improvement threshold.
- **Review Notes**: Results are trustworthy: tracked diff was limited to `train.py`, compile and ruff passed, the first LR drop occurred, all final metrics were present, and the run completed under 10 minutes.
- **Verdict**: no-improvement
- **Verdict Basis**: valid run with `best_test_acc=93.06%`, below both the `93.97%` baseline and the required `94.07%` improvement threshold.

## Unexplored Avenues
- Decoupled decay could be retuned at a different scalar, but EXP-039, EXP-041, and EXP-049 now make isolated decay-axis retuning low priority.
- A future optimizer-dynamics experiment could change momentum scheduling or Nesterov-style behavior jointly with decay, but prior isolated momentum changes were also negative.

## Next Steps
Try a clean mild ColorJitter retry with high confidence for attribution value but medium-low expected improvement, because EXP-047 was confounded by missed LR drop. Partial residual-branch BN scaling is another low-overhead option with low-medium confidence, but EXP-028 makes the family risky. A more distinct late-stability lever may be preferable if it preserves the 21k first-drop budget.

## Exit Action Results
