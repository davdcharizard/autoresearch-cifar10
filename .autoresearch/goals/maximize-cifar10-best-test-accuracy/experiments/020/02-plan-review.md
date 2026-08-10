Claude verified the plan against code, proposal, goal constraints, and installed PyTorch 2.9.1 behavior: SGD defaults are correct; first-step buffer `b1=d1=grad+1e-4*theta0` is identical; Nesterov first update is `1.9*d1`; and both steady directions are `10d`. The one-keyword production change is genuine.

# Prioritized Concerns - EXP-020 Plan

## 1. Evaluation-count fairness gap
Project insights record that batch-dependent epochs can bias a max-over-checkpoints metric. Nesterov may shift step time enough to change weak-tail epoch/evaluation count, but the plan only required at most once per epoch. With a +0.10-point gate, extra looks are material. Record evaluation count and require equality to EXP-010 or treat a higher count as confounded.

## 2. First-direction controller trap
The 1.9x ratio holds on the pre-storage update vector `u1=(1+mu)d1`. Deriving it from stored FP32 parameter differences can suffer catastrophic cancellation and spuriously fail. Require the ratio directly on the update tensor before parameter storage.

## 3. Immutable-corpus CutMix window
The 45-55% CutMix gate plus no-reroll digest rule can deadlock on an unchecked deterministic corpus. Confirm seed 42 is in range before making this a hard gate or adjust the rule before execution.

## 4. Safety apparatus is over-scoped
Class concentration, update spikes, and loss-EMA gates were designed for pathological candidates. Nesterov has an analytically identical steady scale and only a bounded transient, so new controller complexity can create more risk than it removes. Recurrence equality is the load-bearing check.

## 5. Corpus assurance wording
The serialized controller corpus is production-distribution data, not the batches the scored process will see. It rules out gross instability on a production-like corpus and should not be described as production-batch identity.

## 6. Confidence remains fragile
The 94.25 gate is approximately ten test images and cannot be confirmed by reroll. Controller ceremony does not reduce final CUDA/trajectory variance; a bare pass remains weak causal evidence.

# Verdict: APPROVE (conditional)
The `nesterov=True` change is in scope, mechanically correct, and free of PNM's scale pathology. Approval requires closing the evaluation-count gap, measuring the first ratio pre-storage, and resolving the corpus-window deadlock. Other concerns are advisory.
