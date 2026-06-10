# Report EXP-046: Time-Budget-Matched Cosine Schedule
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-046.md
- **Plan**: plans/plan-046.md
- **Log**: logs/exp-log-046.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed `prepare.py` evaluation harness and wall-clock budget. The active baseline is 93.97% from EXP-038 at commit `755be2c`; because the goal requires at least +0.10 percentage points, EXP-046 needed `best_test_acc >= 94.07%` to count as an improvement.

## Idea & Hypothesis
EXP-046 tested a time-budget-matched no-restart cosine schedule on the current EXP-038 anchor. The hypothesis was that driving LR by elapsed training-time fraction would make the schedule robust to variable realized steps and improve late optimization stability versus the abrupt 21k step drop.

## Approach
Only `train.py` changed. The experiment added a clamped `lr_at_fraction(frac)` helper and replaced the active `MultiStepLR([21000, 64000])` path with per-step LR assignment from `total_training_time / TIME_BUDGET_S`. It intentionally preserved model width, batch size, initial LR 0.1, momentum 0.9, weight decay 2e-4, reflection crop padding, label smoothing 0.05, FP32 compile, channels-last, and `MAX_STEPS`.

## Execution
One local foreground run was launched on GPU0 with output captured to `run.log`. Both H20 GPUs were busy from an external workload at launch, so realized throughput was lower than the clean anchor but the run still completed within the 10-minute cap. Preflight passed (`python3 -m py_compile train.py`, `uv run ruff check train.py`), the run completed cleanly, and final summary metrics were present.

## Results
- **Primary metric**: 93.01% (baseline: 93.97%, delta: -0.96 pp, -1.02%)
- **Observations**: The time-fraction LR schedule executed as intended, decaying from about 0.099 early to near zero by the final epochs. Best accuracy climbed to 93.01% at epoch 49 but never approached the 94.07% threshold.
- **Analysis**: The hypothesis was not supported. On this anchor, elapsed-time cosine removed the discrete 21k drop but also reduced LR much earlier than the proven step schedule, causing a lower plateau. GPU contention reduced steps to 19,691, but this was exactly the robustness case the schedule was intended to handle; the result suggests the current anchor still needs the sharper step-drop dynamic or a more carefully calibrated cosine variant.
- **Key Learning**: Time-fraction cosine on the current 2e-4 anchor decays too aggressively and underperforms the 21k step schedule.

## Verification
- **Conditions**: improvement condition failed.
- **Review Notes**: Results are trustworthy. The run completed, produced numeric metrics, modified only `train.py`, used one selected GPU, preserved the fixed harness, and stayed below the 10-minute cap.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result, but `best_test_acc=93.01%` is below both the 93.97% baseline and the 94.07% improvement threshold.

## Unexplored Avenues
- **Step-aware cosine with a held high-LR plateau**: A hybrid schedule could keep LR=0.1 until roughly the historical first-drop region, then cosine-decay through refinement. This may preserve the useful high-LR exploration that EXP-046 lost.
- **High-capacity budget-matched cosine bundle**: The sibling run suggests a broader retuned recipe can pair cosine with larger capacity and stronger regularization, but it should be treated as a bundled high-risk experiment rather than evidence for isolated cosine.
- **Targeted color-only augmentation**: EXP-044 showed broad mild RandAugment was insufficient, but a lower-overhead photometric-only perturbation remains a distinct augmentation test.

## Next Steps
- **High confidence**: Try a targeted color-only augmentation on the current step-schedule anchor, preserving the validated 21k drop and 2e-4 weight decay.
- **Medium confidence**: Test a hybrid schedule that keeps the 21k step anchor behavior but smooths only the late low-LR phase.
- **Medium confidence**: If incremental tests continue to fail, run the high-capacity budget-matched cosine bundle as an explicitly broad, attribution-poor search step.

## Exit Action Results
