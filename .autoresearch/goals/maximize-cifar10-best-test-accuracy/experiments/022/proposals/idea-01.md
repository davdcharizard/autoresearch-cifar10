# Proposal: Lookahead-Wrapped Momentum SGD

## Intervention and hypothesis

Wrap the accepted momentum-SGD optimizer with the fixed Lookahead rule from Zhang et al. Use `k=5`, slow step `alpha=0.5`, and retain the inner SGD momentum buffers across synchronizations. Every fifth ordinary optimizer update, interpolate a detached slow parameter copy halfway toward the fast parameters and copy the new slow values back into the live model. Keep the accepted LR schedule, momentum `0.9`, coupled weight decay `1e-4`, model, augmentation, loaders, timer, and evaluator unchanged.

The paper reports that Lookahead reduces trajectory variance and improves convergence with momentum SGD, including CIFAR experiments at batch 128, and uses `k=5, alpha=0.5` as a robust default. The local accepted recipe spends 80% of its short horizon at LR 0.1 on stochastic RandAugment/CutMix batches; smoothing every five updates could preserve useful high-LR exploration while returning the optimizer to a more generalizable local trajectory. The testable prediction is `best_test_acc >= 94.25%` without materially reducing EXP-010's 26,898 updates.

## Why this is distinct from failed averaging

EXP-018 uniformly averaged eight late weak-tail snapshots and evaluated the averaged endpoint after BN recalibration. That arithmetic window biased backward from a better online iterate and reduced 94.02% online to 93.85% SWA. Lookahead is not post-hoc endpoint averaging: its exponential slow trajectory is updated throughout training, copied into the live fast weights, and directly changes every later gradient. It keeps recent proposals dominant and cannot be evaluated as a separate stale endpoint. This distinction gives it a different mechanism, but also means a loss would be an optimizer result rather than an averaging-only result.

## Exact implementation

After constructing ordinary SGD, make one detached clone per parameter as the slow state and initialize a synchronization counter. Run the existing `optimizer.step()` every batch. On steps divisible by five, under `torch.no_grad()`, apply `slow.add_(fast - slow, alpha=0.5)` and `fast.copy_(slow)`. Do not reset or interpolate momentum buffers; the source paper found retaining inner state viable and used it generally. Do not average BN running buffers, alter parameter groups, change LR, add warmup, switch the evaluated weights, or synchronize on phase/epoch boundaries. Parameter order and shapes must remain exactly accepted.

The slow tensors should be allocated on the H20 in each parameter's dtype and layout after model construction. They add about one model copy, trivial against 98 GB. Synchronization arithmetic must stay inside the counted timed region because it is real training cost. Log only aggregate synchronization count and the final mean slow-fast norm; do not add validation.

## Verification and gates

Before production, use disposable aligned control/candidate checks on persisted hard and CutMix batches:

- At initialization, fast and slow tensors are bitwise equal; the candidate live model, RNG state, optimizer group, and momentum state match control.
- For steps 1-4, the candidate matches ordinary SGD bitwise. At step 5, manually recompute the interpolation and require exact agreement; require all slow/live tensors, gradients, BN buffers, and momentum buffers finite.
- Verify momentum buffers persist across synchronization and receive the sixth ordinary SGD update; no optimizer state is silently reset or duplicated.
- Over 200 identical production-distribution batches, require no candidate-only prediction concentration above 95%, terminal loss EMA at most 1.5x control, and finite parameters/state.
- Confirm evaluation reads the live synchronized model and does not swap to a shadow copy. A checkpoint just before a synchronization may legitimately evaluate a fast point; do not move evaluations to manufacture slow-point looks.

Run five alternating fresh-process H20 timing pairs over persisted strong and weak batches, with warmup and at least 1,000 measured synchronized steps. Require the weighted candidate/control step-time ratio at most 1.01, projected updates at least 26,629 (99% of EXP-010), peak allocation below 650 MiB, and projected total runtime below 540 seconds. Validate exactly one H20 near 97,871 MiB before the full run.

## Risks and falsification

The largest risk is effective-step shrinkage: resetting fast weights halfway to the slow point every five steps can suppress the healthy 89.73% strong-phase fit, especially because the accepted LR schedule was tuned for ordinary momentum. Retaining momentum after copying parameters also creates a temporary mismatch between velocity and location. Conversely, smoothing may improve NLL but not top-1, as label smoothing did. The paper's CIFAR result used ResNet-18 for 200 epochs and tuned comparison hyperparameters, so its small final gain is directional rather than a transferable effect size. A 0.10-point pass is only ten test images at one fixed seed.

If preflight or timing gates fail, classify the experiment invalid/no-go and do not tune `k`, `alpha`, LR, or momentum in this experiment. If the full seed-42 run completes below 94.25%, reject this exact Lookahead point. A switch checkpoint below 87.08% supports excessive trajectory damping; a healthy switch with worse final NLL/top-1 means variance reduction did not improve the relevant basin. Only a valid run at or above 94.25% is an improvement.

## Sources

- `experiments/022/papers/lookahead-optimizer.md` and the NeurIPS 2019 paper: fast/slow mechanism, variance rationale, CIFAR evidence, retained state, and robust `k=5, alpha=0.5` defaults.
- EXP-018 analysis: uniform weak-tail SWA harmed the accepted online iterate.
- `02-system-understanding.md`: accuracy is limited by generalization under a short strong-view phase, with ample memory and backward-dominated step cost.
