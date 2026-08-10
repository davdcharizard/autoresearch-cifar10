# Report EXP-002: Long-Plateau Cosine Refinement
- **Created**: 2026-08-05

## Goal

Increase CIFAR-10 `best_test_acc (%)`, higher is better, from the moving baseline of `91.67%`. A valid improvement must reach at least `91.77%`, modify only `train.py`, use one H20 under the fixed 300-second training budget, and finish within 600 seconds total.

## Idea & Hypothesis

EXP-001 showed that annealing after only 15% of the budget overfit training while losing test accuracy. The selected follow-up retained standard momentum and held `lr=0.1` through 80% of counted training, then stepped to `0.01` and cosine-decayed to `1e-4`. The hypothesis was that baseline-like high-LR exploration supplies useful implicit regularization while guaranteed terminal refinement improves the final solution.

## Approach

Changed only `train.py`. Removed `MultiStepLR`, computed LR from elapsed counted training time, retained ordinary SGD momentum, and reused persistent DataLoader workers. Evaluation ran at four early checkpoints and every epoch during the final 20%, with an unconditional terminal evaluation. External Claude adversarial review specifically led to preserving the baseline's `0.01` regime and using dense late evaluation rather than a sparse schedule that could miss the best checkpoint.

## Execution

One local run executed on a single NVIDIA H20 with no retries or runtime errors. Static compilation, Ruff, pre-commit, diff, and scope checks passed before launch. The process exited `0` before the 600-second supervisor timeout and produced all required summary fields.

## Results

- **Primary metric**: `91.83%` (baseline: `91.67%`, delta: `+0.16` percentage points, `+0.17%` relative)
- **Observations**: Accuracy entered the dense tail at `89.82%`, reached `91.83%` at epoch 86, and ended at `91.82%` at epoch 100. The 26 unique evaluations therefore found only a `0.01`-point best-versus-final gap. The run completed 38,629 steps in `300.0s` counted training and `336.0s` total, with `330.1 MB` peak VRAM and 269,722 parameters.
- **Analysis**: The hypothesis is supported at this operating point. Restoring a long high-LR plateau reversed EXP-001's regression, while the explicit step to `0.01` and deep cosine tail supplied the refinement the unreachable baseline scheduler could not guarantee. Because EXP-002 also removed Nesterov, the gain cannot be attributed solely to plateau length, but the cross-experiment evidence strongly favors preserving high-LR exploration before terminal annealing.
- **Key Learning**: An 80% high-LR plateau followed by low-LR cosine refinement improved accuracy 0.16 points; preserve exploration before terminal annealing.

## Verification

- **Conditions**: All passed: `91.83% >= 91.77%`, all ten summary fields were finite and unique, counted training was `300.0s`, total time was `336.0s`, and 26 unique evaluations ended at summary epoch 100.
- **Review Notes**: Results confirmed trustworthy. The fixed evaluator and dataset preparation were untouched, only `train.py` changed, seed 42 remained fixed, output came from the current run, and the process used the required H20.
- **Verdict**: improvement
- **Verdict Basis**: All integrity and runtime conditions passed, and the primary metric exceeded the required moving-baseline margin by `0.06` percentage points.

## Unexplored Avenues

- Sweep the hold boundary narrowly around 75-85% while keeping standard momentum and the same low-LR tail; EXP-002 establishes that this regime is productive but not that 80% is optimal.
- Replace the discontinuous `0.1 -> 0.01` step with a short transition while preserving the same total low-LR exposure; this may reduce optimization shock without sacrificing exploration.
- Combine the validated schedule with one isolated regularization intervention such as label smoothing, leaving the optimizer horizon fixed to avoid the confounding seen in EXP-001.

## Next Steps

- **High confidence**: keep the validated optimizer and test one de-bundled generalization intervention, with label smoothing the lowest-risk candidate.
- **Medium confidence**: narrow-sweep the plateau/tail boundary if parameter-search overhead can remain within the fixed evaluation protocol.
- **Medium confidence**: explore a modest model-capacity change only after preserving the now-validated schedule and runtime controls.

## Exit Action Results

- None defined.
