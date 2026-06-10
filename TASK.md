# Optimization task: CIFAR-10 (ResNet-20 baseline → modern)

Optimize `train.py` to maximize CIFAR-10 test accuracy under a fixed
training-time budget.

## Objective

Maximize `best_test_acc` — the best test accuracy (percent) reached within the
training-time budget. Higher is better.

## What you may change

Everything inside `train.py`: model architecture, optimizer, learning-rate
schedule, data augmentation, regularization, batch size, model size/type, and
the training loop — as long as the interface contract below stays intact.

## Hard constraints

- **Interface contract.** `train.py` MUST keep a top-level
  `train(seed, time_budget_s)` function that runs training and returns a dict
  containing at least:
  - `"model"`: an `nn.Module` with the **best** checkpoint weights loaded
    (not necessarily the final epoch),
  - `"device"`: the `torch.device` the model is on,
  - `"training_seconds"`: float, the measured training-step time.
  The score is recomputed by reloading this model and running the fixed
  evaluator on it — numbers printed by `train.py` are ignored, so the only way
  to score well is to return a genuinely good model.
- **Time budget.** The training loop must stop once `training_seconds` reaches
  `time_budget_s` (training-step time, excluding validation and startup), as the
  loop already does. Do not extend the budget. A run whose total wall-clock runs
  far past the budget is treated as a failed run.
- **Installed packages only.** `train.py` may import only packages that are
  already installed; do not introduce imports for packages that aren't present.
- **Single GPU.** Train on one GPU.
- **Seed.** Use the `seed` passed into `train(seed=...)`. Do not hardcode or
  search seeds to inflate the metric.

## Soft constraints

- **VRAM.** Increases are acceptable in exchange for a real accuracy gain;
  there is ample headroom on the target GPU.
