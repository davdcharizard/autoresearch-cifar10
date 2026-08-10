# Report EXP-029: Conv2d-Weight-Only Data-Gradient Centralization
- **Created**: 2026-08-06

## Goal

Increase seed-42 CIFAR-10 `best_test_acc` above the 94.15% moving baseline, with at least 94.25% required for improvement, while modifying only `train.py` under the fixed one-H20, 300-second counted-training, under-600-second total protocol. EXP029 tested a forward-preserving generalization projection while requiring at least 99% of accepted update exposure in pre-production timing.

## Idea & Hypothesis

The selected intervention subtracted each Conv2d output filter's data-gradient mean over input/spatial dimensions after backward and before ordinary momentum SGD. It left FC/BN gradients and the accepted all-parameter coupled decay path unchanged, yielding `P(g)+lambda*w` rather than the paper's `P(g+lambda*w)` theorem-aligned operator. The hypothesis predicted a nontrivial removed common-mode component, safe strong-phase behavior, at most 1% step overhead, and `best_test_acc >=94.25%`.

## Approach

Tracked `train.py` received an eleven-line change: a no-grad helper iterated the 19 Conv2d modules, required their weight gradients, applied the `(1,2,3)` projection, and was called exactly once between backward and unchanged SGD. No model, initialization, parameter/buffer, optimizer group, RNG, data, schedule, evaluation, precision, timer, or lifecycle setting changed.

An ignored preflight proved eligibility, FP64 projection agreement, idempotence, untouched non-Conv gradients, RNG neutrality, and exact installed/manual `P(g)+lambda*w` momentum recurrence. It then compared byte-aligned models over the registered 200 strong plus 64 weak batches. A fresh-process timing controller used real production loaders and the exact production helper for one conditioner plus five counterbalanced control/candidate pairs, each with 1,000 measured steps after warmup.

## Execution

The first preflight attempt exposed a controller-only fixture defect: artificially large gradient offsets left approximately `1.9e-06` FP32 mean residual against an absolute `1e-07` assertion, although the production trajectory's maximum residual was `1.49e-08` and the recurrence was exact. One allowed code retry reduced only the deterministic fixture magnitude; candidate code, corpora, gates, and trajectory were unchanged. The second preflight passed and reproduced all substantive metrics exactly.

Paired timing then ran to completion and serialized all five trials. The aggregate and maximum-pair gates failed, so the predeclared protocol blocked production. There was no candidate/timing threshold rescue and no scored run.

## Results

- **Primary metric**: NaN (baseline: 94.15%, delta: N/A; production was not authorized)
- **Projection correctness**: FP64 reference error `1.18e-08`, idempotence residual `1.12e-08`, exact five-step parameter/momentum recurrence, and maximum production-gradient filter mean `1.49e-08`.
- **Mechanism strength**: Strong removed/raw gradient-norm fractions were 0.8727 stem, 0.3669 stage1, 0.4449 stage2, and 0.5865 stage3; weak fractions were 0.9150, 0.4725, 0.5711, and 0.4509. BN did not make the intervention near-inert.
- **Trajectory safety**: Zero candidate-only concentration events over 264 steps; median/p95/max candidate-control update ratios were `0.880251/1.087029/1.274075`; strong/weak loss-EMA ratios were `0.999624/1.003094`; maximum Conv update/parameter ratio was 0.03718. All BN, RNG, state, and corpus integrity checks passed.
- **Timing**: Pair ratios were `1.014498, 1.042778, 1.027132, 1.005090, 1.009153`; aggregate ratio was `1.019749` against the 1.01 gate and maximum was `1.042778` against 1.04. Historical exposure projected 26,377 rather than 26,898 steps.
- **Path timing**: Mean ratios were 1.017750 strong-hard, 1.018790 strong-soft, and 1.025693 weak-hard, showing broad projection overhead rather than a CutMix-specific artifact.
- **Secondary timing integrity**: Control/candidate CVs were 0.770%/1.830%, peak allocation stayed 598.68 MiB, loader delivery was 112.70x consumption, maximum weak rebuild was 2.903s, lifecycle wall/count was 1.00968, and projected total runtime was 333.60s.
- **Analysis**: GC achieved its intended local mechanism and was much safer than recent global optimizer-state interventions, but the literal 19-reduction/19-subtraction implementation lost 1.97% fixed-budget exposure. This does not discredit GC's accuracy mechanism; it shows this implementation cannot test it under the preregistered 99%-exposure attribution standard. Because no primary metric exists, neither benefit nor harm to full-run accuracy can be claimed.
- **Key Learning**: All-Conv GC was safe and removed 37-87% of gradient norm, but 38 small GPU operations cost 1.97% and failed the exposure gate.

## Verification

- **Conditions**: Scored verification was not run; mandatory paired timing failed before production.
- **Review Notes**: The feasibility failure is trustworthy. Five counterbalanced fresh pairs used real loaders and the production helper; CV, memory, loader, lifecycle, and wall-time gates passed; overhead appeared across every target path. The one preflight retry corrected only an independently demonstrated synthetic-fixture scale bug and reproduced the immutable candidate trajectory.
- **Verdict**: invalid
- **Verdict Basis**: Only safety/mechanism/timing evidence exists. The registered fixed-budget feasibility gate blocked the scored run, so `best_test_acc` is NaN.

## Unexplored Avenues

- A genuinely fused all-Conv projection could retain the same semantics with fewer launches, but no supported fused reduction exists in the current code path and TorchInductor is unavailable on Python 3.14. Any custom/fused design is a new systems experiment, not a retry.
- Centralizing only the stem and stage3 would target the largest measured removed fractions with fewer operations, but changes the scientific intervention and layer coverage; it needs independent evidence and review.
- Tail-only GC would amortize overhead over 20% of training and avoid changing strong-phase drift, but its short horizon and phase-specific optimizer rule make it a distinct, lower-confidence hypothesis.

## Next Steps

- **Return to zero/near-zero-overhead data or scalar schedule levers (medium-high confidence)**: the 0.10-point gate is close, while per-tensor GPU regularizers consume scarce exposure.
- **Measure FP32 channels-last only as a pure feasibility probe (medium confidence)**: it targets backward cost directly, but require a real speedup before making any accuracy claim.
- **Consider a predeclared CutMix-alpha or tail-LR operating point (medium confidence)**: both preserve validated architecture/optimizer state and add no counted GPU kernels.

## Exit Action Results

No exit actions were configured for this goal.
