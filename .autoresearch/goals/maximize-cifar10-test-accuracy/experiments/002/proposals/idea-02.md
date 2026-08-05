# Proposal: Early-Only Mild Mixup With a Hard-Label Cosine Tail

## Recommendation

Keep the successful EXP-001 WRN-16-2, optimizer, data pipeline, and time-aligned learning-rate schedule unchanged. Apply mild mixup with `alpha=0.2` only while counted training progress is below `0.65`, then switch completely to ordinary hard-label cross-entropy for the final 35% of the 300-second budget. Do not add Cutout or label smoothing in the same run.

This is a focused combination of one input/target regularizer and temporal removal, rather than a stack of overlapping regularizers. On the EXP-001 throughput profile, the switch occurs at about 195 counted seconds and leaves about 105 seconds, or roughly 50 epochs, for clean-target convergence while cosine LR falls from approximately `0.061` to `0.002`.

## Comparison

| Candidate | Exact candidate setting | Benefit under this budget | Main risk | Decision |
| --- | --- | --- | --- | --- |
| Early mixup | `Beta(0.2, 0.2)`, one scalar lambda and one permutation per batch, active for progress `<0.65` | Directly regularizes both samples and targets; alpha 0.2 is mild and the long hard-label tail addresses delayed convergence | Extra tensor mixing slightly reduces throughput; mixed targets can suppress late peak accuracy if left on too long | **Choose** |
| Mild Cutout | One random `8x8` zeroed square per image, active for progress `<0.65` | Preserves hard targets and may improve occlusion robustness | Per-sample masking/indexing adds implementation and input-pipeline overhead; less directly motivated by the observed near-zero-loss generalization gap | Reserve as a later isolated test |
| Early label smoothing | Cross-entropy with `label_smoothing=0.05` for progress `<0.65` | Essentially no data-movement overhead and directly limits overconfidence | Uniform target noise is less sample-aware than mixup; combining it with mixup double-softens targets and may underfit in 300 seconds | Do not stack; retain as the lowest-overhead fallback |

Mixup has the strongest mechanism match to the current limiter: EXP-001 converged stably to 93.38% and near-zero training loss, so the remaining gap is more plausibly generalization than optimization instability. The saved mixup summary reports CIFAR-10 generalization gains from convex sample/label interpolation. The time-dependent regularization evidence specifically supports applying mixup during an early critical period and removing it later. Label smoothing targets the same overconfidence problem, but its saved summary warns against stacking multiple soft-target methods without calibration. Mild Cutout is distinct, but it introduces an additional corruption and a second variable without better evidence for this first controlled refinement.

## Exact Experiment

- Preserve `NUM_BLOCKS=2`, `WIDEN_FACTOR=2`, batch size 256, SGD/Nesterov settings, weight decay, warmup, cosine schedule, seed 42, `MAX_STEPS`, and evaluation cadence.
- Add `MIXUP_ALPHA = 0.2` and `MIXUP_END_FRACTION = 0.65` in `train.py`.
- At the start of each training batch, compute `progress = min(total_training_time / TIME_BUDGET_S, 1.0)` using the same counted time that drives LR.
- When `progress < 0.65`, draw one lambda from `Beta(0.2, 0.2)`, create one device-local `torch.randperm` over the batch, and replace the model input with `lam * inputs + (1 - lam) * inputs[perm]`.
- Compute the mixed loss as `lam * CE(outputs, targets) + (1 - lam) * CE(outputs, targets[perm])`. Use one scalar lambda for the whole batch to minimize sampling and broadcasting overhead.
- When `progress >= 0.65`, skip all mixup sampling, permutation, and tensor interpolation, and use the existing `F.cross_entropy(outputs, targets)` exactly.
- Retain the existing random crop and horizontal flip throughout. Here, "clean tail" means no mixup, Cutout, or label smoothing and fully hard targets; it does not remove the proven baseline augmentation.
- Make no changes outside `train.py` and add no dependency.

The switch is deliberately hard rather than tapered. It creates a clear, long optimization tail, keeps implementation small, and tests the paper-backed claim that early regularization can be removed without losing its generalization benefit. A 65% cutoff is conservative for a short run: it gives mixup most of the high- and medium-LR representation-learning phase, then leaves over one third of the budget for hard-label margin fitting.

## Hypothesis

Early mild mixup will reduce memorization and improve the WRN representation during the high/medium-LR phase, while the final 105-second hard-label cosine tail will recover class margins and peak hard-label accuracy. The run should exceed the 93.38% baseline and must reach at least **93.48%** to satisfy the goal's 0.1-point improvement rule, without materially increasing VRAM or total runtime.

## Risks and Interpretation

- If epochs or steps fall materially below EXP-001's 147 epochs / 28,540 steps, input interpolation overhead may consume too much of the fixed budget. That would make a near-flat result ambiguous; early label smoothing at `0.05` is the cleaner follow-up.
- If the best score occurs soon after the switch but the final score falls, the hard tail may be too long or the LR floor may be too low; do not infer that stronger mixup is needed from that pattern.
- If both best and final accuracy trail the baseline while throughput is preserved, `alpha=0.2` or 65% duration is still too strong for this model/budget. The next controlled change should shorten mixup to 50%, not add label smoothing.
- A gain cannot be attributed to a seed change: retain seed 42 and run exactly once. The additional stochastic draws are part of the regularizer, not a seed reroll.

## Verification

Run once on the single H20 with `uv run train.py > run.log 2>&1`. Require a complete summary within 10 minutes, about 300 counted training seconds, validation no more than once per epoch, and `best_test_acc >= 93.48%`. Record `final_test_acc`, steps/epochs, and peak VRAM alongside the primary metric so any gain can be distinguished from a throughput or transient-evaluation artifact.

## Evidence

- `.autoresearch/goals/maximize-cifar10-test-accuracy/knowledge/papers/mixup.md`: convex sample/label interpolation improves CIFAR-10 generalization with little GPU overhead, but may delay hard-label convergence in a short budget.
- `.autoresearch/goals/maximize-cifar10-test-accuracy/knowledge/papers/time-matters-regularization.md`: mixup and other regularizers exert much of their benefit early and can be removed after the critical period.
- `.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/002/papers/label-smoothing.md`: mild smoothing can help overconfident classifiers, but stacking soft-target methods requires calibration.
- `.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/001/04-analysis.md`: the current WRN-16-2 completed 147 epochs in 300 counted seconds, peaked at 93.38%, and finished at 93.34%, establishing both the baseline and the available clean-tail convergence window.
