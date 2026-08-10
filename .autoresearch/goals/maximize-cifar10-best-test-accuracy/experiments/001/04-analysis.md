# Report EXP-001: Budget-Aligned Cosine SGD with Nesterov
- **Created**: 2026-08-05

## Goal

Increase `best_test_acc (%)`, higher is better, from the moving CIFAR-10 baseline of `91.67%`. A valid improvement must reach at least `91.77%`, complete successfully on one H20 under the fixed 300-second training budget, and finish within 600 seconds total.

## Idea & Hypothesis

The selected idea replaced unreachable fixed step milestones with a schedule tied to counted training time: hold `lr=0.1` for 15%, then cosine-decay to `1e-4`, with Nesterov momentum. It was chosen because the baseline reaches only 38,525 steps and never enters its planned `lr=0.001` phase. The hypothesis was that preserving a short exploration phase and guaranteeing thousands of low-LR refinement updates would raise accuracy to at least `91.77%`.

## Approach

Changed only `train.py`. The model, batch size 128, transforms, hard-label cross-entropy, seed 42, momentum coefficient, weight decay, maximum steps, fixed evaluator, and 300-second budget remained unchanged. Added elapsed-time LR computation, Nesterov SGD, and persistent training-loader workers; removed `MultiStepLR`. External plan review and diagnostics refined evaluation to 20/40/60/70/80/90% budget checkpoints plus termination because loader epochs measured 18.975s without persistence versus 1.025s with it, and fixed evaluator passes measured 17.271s.

## Execution

One local run executed on a single NVIDIA H20 with no retries or runtime errors. The process exited `0` before the 600-second supervisor timeout. Static compilation, Ruff, pre-commit, diff, and scope checks passed before launch. Verification stopped after the primary accuracy condition failed, as required.

## Results

- **Primary metric**: `91.57%` (baseline: `91.67%`, delta: `-0.10` percentage points, `-0.11%` relative)
- **Observations**: The schedule executed as designed: LR was `0.1000` at 15.7%, `0.0912` at 31.3%, `0.0406` at 62.6%, `0.0159` at 77.9%, `0.0015` at 93.7%, and `0.0001` at termination. Test accuracy was monotonic across all seven observations: `82.05`, `85.66`, `88.99`, `89.41`, `90.79`, `91.22`, and `91.57%`; final equaled best, so sparse evaluation did not create the failure. The run completed 38,434 steps over 99 epochs in `300.0s` counted training and `321.7s` total, with `330.1 MB` peak VRAM and 269,722 parameters.
- **Analysis**: The intervention achieved its local optimization objective but not the generalization objective. Smoothed train loss reached `0.0215` at 99.9% while test accuracy remained below baseline, so missing terminal decay was not the only limiter. The most plausible mechanism is that reducing the high-LR plateau from about 83% to 15% removed useful exploration/implicit regularization and allowed sharper fitting. Nesterov remains a confound, so this result discredits the combined operating point rather than time-aligned scheduling as a whole.
- **Key Learning**: A 15% LR hold with cosine/Nesterov converged monotonically but ended 0.10 points below baseline; preserve longer high-LR exploration.

## Verification

- **Conditions**: Primary accuracy condition failed (`91.57% < 91.77%`); remaining formal conditions were skipped per protocol.
- **Review Notes**: The metric is trustworthy: the command exited `0`, output came from the current run, only `train.py` changed, the fixed evaluator was untouched, and the full trajectory was monotonic with final equal to best. Post-verdict analysis found a complete numeric summary, `300.0s` training, `321.7s` total, and seven unique evaluated epochs, but these do not override the first-condition failure.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid primary metric failed the required improvement condition and was 0.10 percentage points below the moving baseline.

## Unexplored Avenues

- Keep standard momentum and the same time-based implementation but extend the `lr=0.1` hold to 60-75%, then anneal over the remaining budget. This tests whether terminal refinement helps when high-LR implicit regularization is preserved.
- Remove Nesterov while holding the rest of EXP-001 fixed. This is the pre-registered causal discriminator for whether Nesterov, rather than early annealing, caused the regression.
- Use time-indexed milestone fractions, such as decays near 70% and 90%, instead of a long cosine. This preserves a baseline-like plateau while guaranteeing the second refinement phase occurs.

## Next Steps

- **High confidence**: brainstorm a longer-hold, standard-momentum time-aligned schedule that isolates the dominant failure mechanism.
- **Medium confidence**: retain persistent workers and bounded validation as protocol improvements for all future runs.
- **Medium confidence**: after establishing a better optimizer horizon, test a de-bundled wider preactivation ResNet without the fragile compilation/augmentation stack.

## Exit Action Results

- None defined.
