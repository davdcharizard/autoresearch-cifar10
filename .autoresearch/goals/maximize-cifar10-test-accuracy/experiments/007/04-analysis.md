# Report EXP-007: Disable Weight Decay for the Hard-Label Tail
- **Created**: 2026-07-24

## Goal

Maximize CIFAR-10 `best_test_acc` above the accepted 94.07% baseline, with at least 94.17% required for improvement, by testing whether matrix weight decay can be removed during the final hard-label refinement phase.

## Idea & Hypothesis

Keep selective `5e-4` matrix-parameter decay through 65% of counted training, then disable it exactly when mixup ends. The local temporal-regularization evidence made this a plausible way to retain early regularization while freeing late clean-label margin refinement. The hypothesis predicted at least 94.17% with unchanged exposure and final test loss no worse than 0.2432.

## Approach

Added a production helper that validates the two live SGD parameter groups, changes only matrix weight decay from `5e-4` to zero, and returns actual group values for logging. The existing mixup transition invokes this helper before the first hard-label update. A deterministic L2 helper records matrix-weight norms at the switch and endpoint; this telemetry is descriptive because the accepted run has no comparable norm measurements. Model, losses, LR schedule, seed, data, evaluator, and cadence were unchanged.

## Execution

One fixed-seed run completed without retry or adjustment. Preflight invoked the production switch on a real optimizer, proving live group values changed from `[0.0005, 0.0]` to `[0.0, 0.0]`, while parameter count, tensors, and CUDA RNG remained unchanged. The scored run switched mixup and decay exactly once at epoch 92, step 17,876, 195.0 seconds (65.0%), with LR 0.0612. It completed 27,835 steps and 143 epochs in 340.7 total seconds.

## Results

- **Primary metric**: 93.74% (baseline: 94.07%, delta: -0.33 percentage points, -0.35%)
- **Observations**: Best accuracy occurred at epoch 140 and final accuracy was 93.70%. The run realized 142.52 passes, slightly more than EXP-002's 141.9, while final test loss worsened sharply from 0.2432 to 0.3244. Matrix-weight L2 rose from 25.8668 at the switch to 37.5522 at the endpoint, a descriptive 45.2% increase without an accepted-run norm control.
- **Analysis**: The intervention executed exactly and had no meaningful throughput cost, so exposure cannot explain the regression. Near-zero training loss combined with substantially worse test loss and lower accuracy indicates that removing coupled L2 decay during the clean tail encourages a less generalizable, more overconfident solution. The norm telemetry is consistent with that interpretation but cannot establish causality by itself. Unlike mixup, matrix weight decay remains useful beyond the 65% boundary.
- **Key Learning**: Continuous matrix weight decay is beneficial throughout this recipe; removing it for the hard-label tail worsens loss and accuracy despite full exposure.

## Verification

- **Conditions**: Run completion and process conditions passed; the required 94.17% accuracy threshold failed.
- **Review Notes**: Results confirmed trustworthy. One H20, 300.0 counted seconds, 340.7 total seconds, 29 unique allowed evaluations, 142.52 passes, unchanged parameters, a single correct live-group switch, and a `train.py`-only diff were verified.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid completed run scored 0.33 percentage points below baseline and 0.43 below the required threshold, with normal exposure and markedly worse test loss.

## Unexplored Avenues

- A much later cutoff or gradual decay taper could reduce the abrupt no-decay tail, but the large test-loss regression makes more decay-removal tuning low priority.
- Decoupled weight decay through AdamW-style optimization would change both optimizer family and regularization geometry; it is materially different and lacks local evidence under this budget.

## Next Steps

- **High confidence**: retain continuous `5e-4` matrix decay and test the decoupled cosine-to-zero endpoint, which changes optimization amplitude without removing regularization.
- **Medium confidence**: investigate a throughput-preserving optimizer or precision change that can create more headroom than another small regularization adjustment.
- **Low confidence**: test evaluator-consistent channel scaling only with a stem-specific hypothesis, since immediate BatchNorm cancels much of uniform input scaling.

## Exit Action Results

No exit actions were defined for this local-only goal.
