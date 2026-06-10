# Report EXP-072: Fan-Out Kaiming Conv Initialization
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-072.md
- **Plan**: plans/plan-072.md
- **Log**: logs/exp-log-072.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed harness while modifying only `train.py`. The current experiment-index baseline was 94.11% from EXP-064 at commit `1119ff8`, and the goal requires at least +0.10 percentage points over baseline to count as an improvement, so EXP-072 needed `best_test_acc >= 94.21%`.

## Idea & Hypothesis
The chosen idea was a conv-only fan-out Kaiming initialization probe. The hypothesis was that the current default Kaiming call might be slightly miscalibrated for the residual convolution stack, and that explicit `mode="fan_out", nonlinearity="relu"` for Conv2d layers could improve signal scaling enough to lift the CutMix anchor past 94.21%.

## Approach
`train.py` was the only modified tracked file. `_weights_init` was split into explicit Conv2d and Linear branches: Conv2d now used fan-out ReLU Kaiming normal initialization, while Linear kept the existing default Kaiming normal call. A startup marker was added to verify the initialization variant in `run.log`. All EXP-064 anchor settings were preserved, including `STAGE_WIDTHS=(28, 56, 112)`, `LR=0.1`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, unit-std normalization, `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, and `CUTMIX_LABEL_SMOOTHING=0.05`.

## Execution
One local foreground run executed on GPU0 with output captured to `run.log`. Preflight checks passed: `git diff --name-only` listed only `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` passed. Startup markers confirmed CUDA, the CutMix anchor, and the fan-out conv initialization marker. The first LR drop fired cleanly at step 21000 in epoch 54; the run completed normally with no infrastructure or runtime errors.

## Results
- **Primary metric**: 94.16% (baseline: 94.11%, delta: +0.05pp, +0.05%)
- **Observations**: Pre-drop best reached 88.45% at epoch 46. After the step-21000 LR drop, best rose quickly to 93.29% by epoch 59, peaked at 94.16% at epoch 74, and ended at 94.04% after 102 epochs / 39,757 steps.
- **Analysis**: The intervention produced a small positive movement relative to the baseline but did not clear the +0.10pp noise guard. This suggests conv fan-out initialization is compatible with the current CutMix anchor and may slightly improve the plateau, but its isolated effect is too small to count as a reliable improvement.
- **Key Learning**: Conv fan-out initialization is a valid near-miss lever, but isolated initialization changes are likely too small to beat the CutMix anchor threshold.

## Verification
- **Conditions**: The run completed, produced numeric metrics, respected the `train.py`-only scope, reached the first LR drop, and stayed under the 10-minute cap; the improvement threshold failed.
- **Review Notes**: Results are trustworthy: the log contains the expected startup markers, final metric block, and no error signatures; persistent untracked `data/` was ignored.
- **Verdict**: no-improvement
- **Verdict Basis**: `best_test_acc=94.16%` is above the 94.11% baseline by only +0.05pp, below the required 94.21% threshold.

## Unexplored Avenues
- Early CutMix warmup could still test whether the validated static CutMix regularizer is slightly harmful during the earliest representation phase while preserving the post-drop CutMix anchor.
- A classifier-specific initialization probe remains untested, but expected effect size is probably even smaller than EXP-072 and may not clear the noise guard alone.
- Combining fan-out conv initialization with another independently motivated near-miss is possible, but should be deferred until there is a clearer paired mechanism to avoid additive-noise chasing.

## Next Steps
- **Early CutMix warmup** (medium confidence): run clean label-smoothed batches for a short initial phase, then restore static `CUTMIX_PROB=0.5`; this is distinct from the failed post-drop CutMix taper.
- **Low-cost classifier init/calibration probe** (low confidence): adjust only final-layer initialization if a precise rationale is selected, but treat it as likely sub-threshold.
- **Search for a distinct non-augmentation lever** (medium confidence): prioritize changes that preserve the CutMix anchor and 21k LR drop while avoiding recurring failed families.

## Exit Action Results
