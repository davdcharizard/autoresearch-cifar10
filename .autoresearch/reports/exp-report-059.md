# Report EXP-059: Average-Pool Option-A Downsample Shortcut
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-059.md
- **Plan**: plans/plan-059.md
- **Log**: logs/exp-log-059.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed harness, with higher accuracy better. The active baseline before this experiment was EXP-038 at `93.97%`, and the goal requires at least a `+0.10` percentage-point gain, so EXP-059 needed `best_test_acc >= 94.07%` to count as an improvement.

## Idea & Hypothesis
EXP-059 tested whether the stage-transition shortcut was losing useful spatial information through strided slicing. The chosen idea replaced option-A shortcut slicing with average pooling before zero-channel padding, preserving the no-parameter shortcut family while borrowing the downsampling-quality intuition from ResNet-D-style refinements.

Hypothesis: average pooling in the shortcut path would preserve local evidence at stride-2 transitions and lift the current anchor to at least `94.07%` without changing throughput enough to miss the first LR drop.

## Approach
`train.py` was the only tracked code file changed. The implementation added `SHORTCUT_DOWNSAMPLE = "avg_pool_option_a"` for startup logging and replaced:

```python
shortcut = shortcut[:, :, :: self.stride, :: self.stride]
```

with:

```python
shortcut = F.avg_pool2d(
    shortcut, kernel_size=self.stride, stride=self.stride
)
```

inside the existing `if self.need_pad:` branch. The existing zero-channel padding, model widths, optimizer, weight decay, LR milestones, augmentation, label smoothing, compile path, channels-last setting, time budget, and once-per-epoch validation were preserved.

## Execution
One local foreground run was launched on GPU0 with `env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. Startup confirmed `ResNet-20 | params: 822,790`, `Shortcut downsample: avg_pool_option_a`, and `Batches per epoch: 390`.

The run completed normally in 403.3 seconds total wall time. It reached the first LR drop at step 21000 in epoch 54 with `lr: 0.0100`, so the comparison was not confounded by missed schedule timing. No runtime errors, shape errors, OOMs, or scope violations occurred.

## Results
- **Primary metric**: 93.42% (baseline: 93.97%, delta: -0.55 pp, -0.59%)
- **Observations**: Accuracy rose quickly after the LR drop, reaching 93.42% by epoch 66, then stayed in the 93.0-93.4% band through epoch 102.
- **Analysis**: The hypothesis was not supported. Average-pool shortcut downsampling preserved the step budget and did not add parameters, but it lowered the best metric to the same 93.42% level as EXP-057 and far below the 94.07% threshold. This suggests the current strided option-A shortcut is not the limiting transition mechanism for the active recipe, or that shortcut averaging blurs features in a way the small CIFAR ResNet does not recover from under the fixed budget.
- **Key Learning**: Parameter-free average-pool shortcut downsampling preserves throughput and LR timing but weakens the current anchor, so transition smoothing is worse than strided option-A here.

Secondary metrics:
- `final_test_acc`: 93.20%
- `final_test_loss`: 0.2566
- `training_seconds`: 300.0
- `total_seconds`: 403.3
- `startup_seconds`: 2.4
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39,682
- `num_params`: 822,790

## Verification
- **Conditions**: All process and integrity conditions passed.
- **Review Notes**: Results are trustworthy. The run modified only `train.py`, completed within the 10-minute cap, reported numeric metrics, preserved the fixed evaluation harness, kept `num_params=822,790`, and reached the step-21000 LR drop.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result, but `93.42% < 94.07%`; the experiment failed the goal's required +0.10 percentage-point improvement threshold.

## Unexplored Avenues
- Stage-specific shortcut changes could test only the later transition, where features are more semantic, but the EXP-059 result makes this lower priority than more distinct mechanisms.
- A learned or anti-aliased transition coupled with schedule retuning remains possible, but EXP-018 projection shortcuts and EXP-059 average pooling both argue against isolated shortcut-transition changes.

## Next Steps
Prefer distinct regularization-balance or optimizer-dynamics probes over more isolated shortcut variants.

- Medium confidence: test mixup without label smoothing, because EXP-055 was a comparatively high no-improvement and may have been over-regularized by combining two target-softening methods.
- Medium confidence: try a narrowly scoped late-stage architecture variant only if it adds little overhead and avoids broad per-block costs seen in EXP-058.
- Low confidence: revisit transition shortcuts only with a stronger coupled rationale; isolated shortcut changes have now failed in both learned projection and average-pool forms.

## Exit Action Results
