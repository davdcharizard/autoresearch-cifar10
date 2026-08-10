# External Claude Plan Review - EXP-016

The mandatory external Claude plan review completed successfully. No fallback reviewer was used.

## Prioritized Concerns

1. **Disposable controller fidelity is load-bearing.** The timing and numerical scripts are ignored artifacts created during execution, but their outputs authorize the only production run. They must not silently enable cuDNN benchmark, channels-last, omit pinned H2D/schedule work, alter backend flags, or use a different synchronization/warmup regime. The ratio projection anchored to EXP010 can also be optimistic because clean warm timing differs from realized production averages.
2. **Accumulated BF16 BatchNorm/evaluation drift is under-tested.** FP32 evaluation consumes running statistics accumulated from BF16-rounded training activations. A one-step running-stat comparison cannot detect drift across the training trajectory or the resulting FP32-eval mismatch near a narrow metric threshold.
3. **The 300-second timing-controller timeout is fragile.** Fifteen fresh CUDA processes plus conditioning, warmup, and measured steps can exceed it for harness reasons even when the candidate is viable.
4. **Wall projection must use fresh width-3 evaluation measurements.** The historical 17.3-second evaluator note is contradicted by EXP010's 330.7-second total with 19 evaluations and is not a valid projection input here.
5. **The 300-second training summary is protocol evidence only.** It verifies the fixed timer, not candidate throughput or quality; actual steps carry exposure evidence.
6. **Conditioning allowances are ambiguous.** Timing and inference benchmarking need separately named, bounded conditioning processes rather than a single global "one" allowance.

## Adopted Corrections

- Require a second external Claude read-only review of the completed controller sources and production diff before any controller is trusted; no fallback reviewer.
- Pin and report backend/layout/autotune state, exact production interval, H2D, LR calculation, synchronization, workload reuse, warmup, and child-process behavior.
- Require both the EXP010 ratio projection and an absolute candidate mean-step projection with a 2.5% safety haircut to retain at least 22,863 steps.
- Extend the 200-step paired gate to compare accumulated BN state and FP32-evaluation logits/loss on held-out real batches; do not recalibrate BN.
- Raise the timing-controller timeout to 480 seconds while retaining hard no-go gates for actual timing instability.
- Base wall projection only on freshly measured width-3 FP32 evaluator/startup costs and current expected evaluation count.
- Clarify that one timing-conditioning process and one separate inference-conditioning process are allowed, with neither scored.
