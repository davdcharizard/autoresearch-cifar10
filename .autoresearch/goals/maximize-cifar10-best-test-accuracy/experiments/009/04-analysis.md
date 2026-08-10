# Report EXP-009: Exclude BN and Bias from Weight Decay
- **Created**: 2026-08-05

## Goal

Increase CIFAR-10 `best_test_acc (%)`, higher is better, from the width-2 moving baseline of 93.55% at `8faf0f3`. A valid improvement required at least 93.65% under the fixed one-H20, 300-second, `train.py`-only protocol.

## Idea & Hypothesis

Test parameter-selective decay while preserving the accepted kernel scalar: keep `1e-4` on tensors with `ndim > 1`, but remove decay from BN affine parameters and biases. Claude's idea critic selected parameter targeting over scalar interpolation; its plan critic then caught that the initial `2e-4` selective refinement both confounded two levers and pushed fit-limited kernels toward EXP-008's failure. The isolated hypothesis predicted that freeing functional scale/offset parameters would improve strong-view fit without surrendering kernel regularization, reaching at least 93.65%.

## Approach

Changed only optimizer parameter grouping in `train.py`. Materialized all trainable parameters once, partitioned 20 matrix/kernel tensors (1,071,200 elements) at `1e-4` decay and 39 one-dimensional tensors (2,762 elements) at zero, and added count, uniqueness, and element-coverage assertions. Width 2, total parameters, SGD LR/momentum, N1/M7 through 80%, deterministic weak tail, timer, seed, loss, and evaluator stayed fixed.

## Execution

Mandatory external Claude idea and plan reviews both completed with exit code 0; no fallback reviewer was used. Compilation, Ruff, formatting, pre-commit, exact diff scope, optimizer partition, and one-idle-H20 checks passed. One fixed-seed run exited 0 without retry after 300.0 counted seconds and 333.2 total seconds, completing 27,172 steps in 70 epochs with 598.7 MB peak allocation.

## Results

- **Primary metric**: `93.52%` (baseline: `93.55%`, delta: `-0.03` percentage points, `-0.03%` relative)
- **Observations**: Strong-view accuracy reached 89.43% at 60% but fell to 88.26% at the epoch-56 switch, 1.82 points below EXP-007 and 6.97 above EXP-008. Its strong-phase train-loss EMA was 0.2183, slightly lower than EXP-007's 0.2283. The first weak checkpoint reached 92.71% versus EXP-007's 92.96%; accuracy peaked at 93.52% on epoch 65 and finished at 93.50% with 0.2340 NLL. EXP-007 finished at 93.49% with materially better 0.2196 NLL. Final train-loss EMA was 0.0303, below EXP-007's roughly 0.04, while exposure was effectively identical (27,172 versus 27,143 steps).
- **Analysis**: The intervention achieved its intended local effect on fitting but not on generalization. With the same kernel scalar and exposure, lower strong and final train loss show that removing decay from BN affine/bias relaxed optimization pressure. However, strong-view test accuracy did not improve, final NLL worsened by 0.0144, and best top-1 remained 0.03 below baseline. The 0.03 top-1 difference alone is too small to claim a precise causal regression from one seed, but it decisively misses the pre-registered 0.10-point gain and the worse NLL supports retaining decay on these parameters. Combined with EXP-008, the evidence brackets the accepted all-parameter `1e-4` recipe: more decay underfits, while selectively less decay fits harder without better test quality.
- **Key Learning**: Removing BN/bias decay improved train fit but worsened NLL and missed baseline; all-parameter `1e-4` remains the better width-2 regularizer.

## Verification

- **Conditions**: Completion, numeric summary, timing, scope, hardware, parameter count, lifecycle, and evaluation uniqueness passed. Primary accuracy failed: 93.52% <93.65%.
- **Review Notes**: Results are trustworthy. Only the reviewed optimizer grouping changed; one idle H20 was used; evaluator, RNG seed, data lifecycle, schedule, and timer were preserved; 19 evaluations occurred on 19 unique epochs; no retry or seed selection occurred. The identical exposure and isolated scalar make the local comparison particularly clean, while ordinary single-run noise still limits interpretation of the tiny top-1 delta.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid run finished 0.03 points below the moving baseline and 0.13 below the required threshold.

## Unexplored Avenues

- The scalar interval between `1e-4` and `5e-4` remains technically unmeasured, but both external critics and the opposing EXP-008/009 failure mechanisms make its expected upside too small for the one-seed 0.10-point gate.
- Decoupled weight decay would change optimizer semantics beyond parameter targeting. It lacks local evidence and should not be prioritized without a separate mechanism that addresses the fixed-horizon fit limit.
- Trajectory or weight averaging could improve late generalization without increasing regularization during the strong-view fit phase, but BatchNorm state handling and per-step overhead must be reviewed and preflighted.
- A representation or data-target intervention such as preactivation or conservative mixing remains orthogonal to the now-narrowed decay lever and may offer a larger signal.

## Next Steps

- **High confidence**: preserve all-parameter `1e-4` decay and move to a distinct, higher-ceiling mechanism rather than testing scalar interpolation.
- **Medium confidence**: adversarially review tail trajectory averaging with an explicit BatchNorm-state and fixed-time cost plan.
- **Medium-low confidence**: revisit representation/data interactions on width 2, prioritizing candidates that preserve the validated RandAugment-to-weak lifecycle.

## Exit Action Results

- None defined.
