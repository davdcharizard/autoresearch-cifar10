# Report EXP-016: BF16-Funded Width-3 Postactivation ResNet-20
- **Created**: 2026-08-06

## Goal

Increase CIFAR-10 `best_test_acc (%)` above the 94.15% moving baseline at `7c1e7d8`. A valid improvement required at least 94.25% under the fixed one-H20, seed-42, 300-counted-second training and ten-minute total-wall protocol.

## Idea & Hypothesis

Increase the accepted postactivation ResNet-20 width from 2 to 3 and use CUDA BF16 autocast for training forward plus cross-entropy while retaining FP32 master parameters, gradients, optimizer state, BatchNorm persistent state, and evaluation. Width was the only local multi-point architecture lever, and forward/backward account for 97.57% of measured GPU-stage time. The hypothesis required BF16 to beat the actual default-TF32 width-3 path by at least 1.15x, retain at least 22,863 updates, remain numerically aligned on real data, and raise FP32-evaluated best accuracy to at least 94.25%.

## Approach

The tracked diff changed only `train.py`: `WIDTH_MULTIPLIER = 3`, a fail-closed CUDA BF16 capability/provenance check, and one autocast context around training forward/loss. It preserved the accepted graph, FP32 backward/SGD/evaluation, seed, data, optimizer, schedule, timer, evaluator, and worker lifecycle. Mandatory external Claude idea and plan reviews selected and hardened the method with no fallback reviewer. Because the disposable controllers decided whether production could run, their complete source and production diff also received adversarial Claude review; all identified TF32, BatchNorm-dtype, timing-fidelity, exposure, and wall-projection issues were corrected and re-reviewed to `APPROVED` before use.

## Execution

Static checks passed with exactly 2,412,730 FP32 parameters on the sole idle 97,871 MiB H20 and only `train.py` tracked. The first numerical-controller launch failed before candidate execution because a path-launched ignored script could not import root-level `train.py`; both controllers explicitly prepended the resolved project root, and Claude approved that narrow fix before the single code-error retry. The retry ran reset one-step comparisons and entered the 200-distinct-production-batch paired trajectory. It tripped the predeclared candidate-only concentration veto: BF16 predictions exceeded 95% in one class while the same-step width-3 FP32 control remained at or below 95%. The assertion preceded result serialization and worker cleanup, so the failed process was terminated and no step-specific JSON exists. No orphan process remained. Timing, loader, wall, and production runs were correctly skipped; no `run.log` was created.

## Results

- **Primary metric**: `NaN` (baseline: `94.15%`; no accuracy run)
- **Observations**: Scope, structure, BF16 capability, FP32 persistent state, and external reviews passed. The numerical trajectory alone failed on candidate-only near-single-class concentration before the method could establish its TF32-relative speed or exposure. No non-finite value was reported before the concentration assertion, but the controller intentionally stopped at the first failed conjunctive condition.
- **Analysis**: The paired control used the same width-3 initialization and exact real N1/M7+CutMix batches, so the divergent concentration localizes the veto to the BF16 training path in this operating point rather than width alone or different data. It does not prove BF16 training is universally unstable: stochastic high-LR trajectories can diverge, the failing step was not serialized, and a transient concentration might later recover. However, relaxing the registered 95% gate after observing it, adding a scaler/warmup, narrowing autocast, or switching width would be a new unreviewed method. The exact full-forward BF16-width3 resource exchange therefore remains unsafe and unproven, and the production accuracy budget was correctly preserved.
- **Key Learning**: Width-3 BF16 entered candidate-only near-single-class concentration within 200 real batches; this exact full-forward autocast operating point is unsafe.

## Verification

- **Conditions**: Baseline/scope/GPU, static structure, and external controller review passed. The paired production-distribution numerical gate failed; timing/funding, loader/lifecycle, wall, production completion, and accuracy conditions were not reached.
- **Review Notes**: The failure is trustworthy as a pre-registered veto, not as an accuracy measurement. Claude approved the exact controller and import repair, the H20 was uncontended, and the paired FP32 control did not cross the same concentration threshold. The missing serialized step limits mechanism depth but cannot convert the failed gate into a pass.
- **Verdict**: invalid
- **Verdict Basis**: Partial safety evidence only; the reviewed mechanism gate correctly blocked the accuracy run, leaving `best_test_acc` unavailable.

## Unexplored Avenues

- A new experiment could keep BatchNorm execution explicitly FP32 while autocasting only convolutions/linear work, but that is a distinct precision policy requiring fresh numerical, timing, and adversarial review rather than an EXP016 rescue.
- Width-2 BF16 could isolate precision behavior and throughput, but exposure alone has no locally demonstrated accuracy mechanism and the EXP016 paired failure lowers its priority.
- Width-3 FP32 could isolate capacity accuracy, but it lacks the resource funding premise and would require a separately justified exposure floor before consuming a production run.

## Next Steps

- **Medium confidence**: reconsider isolated Nesterov on the complete accepted recipe; it has clean attribution and near-zero compute cost, though its ceiling is close to noise.
- **Medium confidence**: reconsider full-path anti-aliased transitions with strict exposure/alignment gates; it is literature-grounded but risks detail suppression and strong underfit.
- **Medium confidence**: generate fresh candidates that preserve FP32 accepted numerics while changing representation or regularization, rather than immediately rescuing mixed precision.

## Exit Action Results

- None defined.
