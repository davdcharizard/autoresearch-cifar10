# Report EXP-003: Modest Label Smoothing
- **Created**: 2026-08-05

## Goal

Increase CIFAR-10 `best_test_acc (%)`, higher is better, from the moving baseline of `91.83%`. A valid improvement must reach at least `91.93%`, modify only `train.py`, use one idle H20 under the fixed 300-second counted training budget, and finish within 600 seconds total.

## Idea & Hypothesis

Add only `label_smoothing=0.05` to EXP-002's accepted hard-label cross-entropy while preserving its model, augmentation, standard momentum, 80% `lr=0.1` plateau, and low-LR cosine tail. The hypothesis was that mild target smoothing would reduce excessive confidence at negligible compute cost and raise best accuracy to at least `91.93%`. Mandatory external Claude adversarial review selected this idea but explicitly warned that smoothing might improve NLL/calibration without moving top-1.

## Approach

Changed only `train.py`: added `LABEL_SMOOTHING = 0.05` and passed it to the existing training-only `F.cross_entropy`. The fixed evaluator, seed, architecture, transforms, loader, optimizer, schedule, and evaluation cadence were untouched. Claude plan review added an idle-GPU preflight, a non-brittle counted-time sanity band, and a PyTorch API capability check.

## Execution

One fixed-seed local run executed on the only visible NVIDIA H20 after confirming 97,871 MiB total, zero used memory, zero utilization, and no compute process. Compilation, Ruff, pre-commit, API, diff, and scope checks passed. The process exited `0` with no retry, completing 300.0 counted seconds and 333.6 seconds total.

## Results

- **Primary metric**: `91.83%` (baseline: `91.83%`, delta: `0.00` percentage points, `0.00%` relative)
- **Observations**: Best accuracy matched the baseline at epoch 91 and ended at `91.80%` on epoch 93. Fixed-evaluator test loss fell from EXP-002's `0.2843` to `0.2740`, but top-1 did not improve. The run completed 36,039 steps versus EXP-002's 38,629, a loss of 2,590 steps or 6.7%, while architecture and counted time were identical. All 24 evaluation epochs were unique and terminal evaluation matched summary epoch 93.
- **Analysis**: The intervention achieved a plausible local confidence/NLL effect but failed the declared top-1 objective, exactly matching Claude's main risk. The built-in smoothing path was also not compute-free in this tiny-model regime: fewer updates and seven fewer epochs reduced low-level optimization exposure. Therefore the result rejects built-in `epsilon=0.05` under this fixed horizon, but it does not fully separate the regularizer's statistical effect from its 6.7% throughput penalty. A single-log-softmax equivalent implementation could test that narrower question; weaker smoothing with the same overhead is not well motivated.
- **Key Learning**: Label smoothing lowered test loss but not top-1 and reduced fixed-budget steps 6.7%; built-in smoothing was not compute-free.

## Verification

- **Conditions**: Primary accuracy failed (`91.83% < 91.93%`); remaining formal necessary conditions were skipped in declared order.
- **Review Notes**: The result is trustworthy: the current-run log was unique, the process exited cleanly, the evaluator and dataset preparation were untouched, only the planned `train.py` loss change existed, seed 42 remained fixed, and the selected H20 was idle. Post-verdict integrity inspection found a complete finite summary, 300.0 seconds counted training, 333.6 seconds total, and 24 unique evaluations ending at epoch 93.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid primary metric exactly matched the moving baseline and missed the required improvement by 0.10 percentage points.

## Unexplored Avenues

- Implement mathematically equivalent smoothing from one explicit `log_softmax` and target gather, then benchmark its step cost before a full run. Recovering the lost 6.7% optimization exposure could distinguish statistical failure from implementation overhead.
- Apply smoothing only during the high-LR plateau and return to hard labels for the refinement tail. This may retain representation regularization while aligning the final objective with top-1, but it changes the mechanism and requires a new predeclared experiment.
- A stronger `epsilon=0.1` could move top-1 more than `0.05`, but the flat accuracy and finite-horizon cost make this lower confidence than switching to an input-space regularizer.

## Next Steps

- **High confidence**: return to a fresh brainstorm with throughput cost treated as a first-class constraint; compare efficient smoothing, worker-side augmentation, and narrow schedule tuning.
- **Medium confidence**: consider one-operation RandAugment because host-side transforms may add generalization pressure without consuming synchronized GPU step time.
- **Medium confidence**: retain Claude's mixup-off-tail refinement as a higher-ceiling option, but measure its per-step cost before committing the fixed-budget run.

## Exit Action Results

- None defined.
