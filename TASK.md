# Optimization task for cifar-10

This is a task to improve a Resnet20 baseline from 2016 up to date.

## Git Rules

If the git remote is unavailable to push or make PRs, **this is intentional**. Keep all commits local for experimentation - the purpose is to isolate the experiments from any remote branches that might contain any information on how to improve the baseline, which would be a source of data leakage.

## Files

**Read the in-scope files**: The repo is small. Read the files for full context:
   - `prepare.py`: fixed constants, and evaluation function. Do not modify
   - `train.py`: the file you modify. Model architecture, optimizer, lr, augmentation, regularization, training loop etc...

## Experimentation

Each experiment must run on a single GPU. On the compute node you should have access to an **NVIDIA H20 with 98GB memory** (make sure to confirm this before starting). The training script runs for a **fixed time budget defined in prepare.py** (wall clock training time, excluding startup/compilation adn validation run). You launch it simply with: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` this the only file you edit. Everything is fair game: model architecture, optimizer, data augmentation, hyperparameters, training loop, batch size, model size, model type etc...

**What you CANNOT do**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation and time budget
- Install new packages or add dependencies. You can only use what is already in `pyproject.toml`
- Modify the evaluation harness. The `Eval.evaluate()` method in `prepare.py` is the ground truth metric.*

**The goal is simple: get the highest test accuracy (best_test_acc) possible.** Since the training time budget is fixed and the validation time is removed focus only on getting the best hyperparameters and training code setup. The first constraint is that the code runs without crashing and finishes within the time budget. The second is not to run the validation more than once per epoch

**VRAM** is a soft constraint. Some increase if acceptable for meaningful increase in best_test_acc. With this dataset you should have some leeway.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

**Timeout**: Each experiment should take a few minutes total. If a run exceeds 10 minutes, kill it and treat it as a failure.

**Pre-commit**: Ruff is used in pre-commit to have a consistent writing style. It will generally auto fix the issue but implement what you want do not feel limited by it. 

## Output format

Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)

Once the script finishes it prints a summary like this:

```
---
best_test_acc:    91.86%
final_test_acc:   91.86%
final_test_loss:  0.2543
training_seconds: 300.1
total_seconds:    325.9
startup_seconds:  3.2
peak_vram_mb:     1234.5
num_epochs:       164
num_steps:        64000
num_params:       272,474
```

You can extract the key metrics from the log file:

```
grep "^best_test_acc:\|^peak_vram_mb:" run.log
```

If the grep output is empty, the run crashed. You can run `tail -n 50 run.log` to read the Python stack trace and attempt a fix.

## Cleanup

When an experiment is finished and the log is no longer needed, make sure to remove the `run.log` before making any new experiments. This helps keep the working tree clean instead of letting logs accumulate from prior experiments.