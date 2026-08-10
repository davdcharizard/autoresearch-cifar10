# Proposal: CutMix Alpha 0.5 at Fixed Probability

## Exact Intervention

Change one production literal:

```python
CUTMIX_ALPHA = 0.5  # accepted: 1.0
```

Keep `CUTMIX_PROBABILITY=0.5` and every other line of accepted `train.py` unchanged. The candidate remains width-2 postactivation ResNet-20, N1/M7 through 80%, 50% CutMix-gated strong batches, hard weak tail, all-parameter decay `1e-4`, standard momentum, accepted LR schedule, seed 42, worker lifecycle, timer, evaluator, and 1,073,962 parameters.

Do not change probability, phase timing, alpha after preflight, target loss, collator, logging, seed, or any second hyperparameter. No alpha interpolation or rerun follows a miss.

## Accuracy Mechanism

EXP-010 established alpha-1 CutMix at p=0.5 as a genuine +0.60-point improvement: 94.15% best, 89.73% switch accuracy, and 99.10% exposure retention. EXP-011 showed that raising p to 0.75 preserved throughput but lowered switch fit by 2.91 points and best accuracy by 0.15. Alpha tuning asks a different question: preserve the successful number of regional events while reducing how ambiguous each event is.

For continuous `lambda ~ Beta(alpha, alpha)`, alpha 1 is uniform and alpha 0.5 is U-shaped. Both have mean 0.5, but variance increases from `1/12` to `1/8`. The expected two-class ambiguity `E[2*lambda*(1-lambda)] = alpha/(2*alpha+1)` falls from `1/3` to `1/4`, a 25% reduction. Alpha 0.5 therefore produces more near-original and near-donor examples while retaining regional class-bearing replacement.

This may recover part of the small EXP-010 strong-fit deficit versus EXP-007 (89.73% versus 90.08%) without discarding CutMix's lower NLL and weak-tail benefit. It does not claim that p=0.5 alpha 1 is clearly too strong; the deficit is only 0.35 points, so alpha 0.5 may instead dilute the winning localization/occlusion signal.

Torchvision adjusts target lambda to the clipped integer rectangle area, so continuous Beta formulas are rationale, not assumed production statistics. The empirical gate below must prove the intended shift.

## RNG and Isolation

Retain the existing top-level `v2.CutMix`, forkserver loader, and `cutmix_collate` exactly. The worker draws the same p=0.5 gate inside `torch.random.fork_rng(devices=[])`; alpha changes only the selected CutMix transform's Beta distribution. The fork restores CPU RNG after either branch, so later crop/flip/RandAugment streams and subsequent gate states remain isolated from internal CutMix draws.

Preflight must show, from identical worker-entry RNG states, that hard decisions/outputs are bitwise accepted and that the mixed-versus-hard decision sequence is identical between alpha 1 and 0.5. Mixed rectangles/targets intentionally differ. Model construction, optimizer state, main CPU/CUDA RNG, weak loader, and all non-CutMix batches remain bitwise aligned before training.

## Stochastic-Distribution Gate

Use at least 10,000 independent controlled CutMix calls for each alpha on batch-128 FP32 32x32 inputs whose labels and pixel signatures are arranged so every rolled source/donor pair differs. Torchvision samples one lambda/rectangle per batch, so count calls rather than examples. Record adjusted retained-area lambda from probability targets and independently recover pasted rectangle area from pixels. Save raw statistics before assertions.

Require:

- every target is finite/nonnegative, has shape `[128,10]`, sums to one, and matches actual rectangle area;
- alpha-0.5 and alpha-1 adjusted-lambda means differ by at most 0.05; boundary clipping may bias both away from the continuous Beta mean;
- alpha-0.5 adjusted-lambda variance is at least 1.25x alpha-1 variance;
- alpha-0.5 mean `2*lambda*(1-lambda)` is at most 0.85x alpha-1, proving materially lower ambiguity after clipping/rounding;
- at least 70% of alpha-0.5 events paste a nonzero, non-full rectangle, so the candidate has not collapsed into effectively hard replacement;
- 20,000 full eight-worker collations realize 48.5-51.5% mixed batches, preserve CPU/CUDA RNG around collate, stop all workers, and rebuild a hard int64 weak loader.

If discrete behavior fails these mechanism gates, do not run production or choose another alpha.

## Replayable Production-Batch Safety

EXP-019 was invalid because fresh forkserver attempts did not replay post-transform batches. For this experiment, first materialize and persist at least 200 exact post-N1/M7 source batches, integer labels, p=0.5 gate decisions, and pre-CutMix CPU RNG states. Generate alpha-1 and alpha-0.5 final inputs/targets once from those artifacts, serialize both arms and all geometry metadata, then train paired accepted/candidate models only from the persisted tensors. Emit diagnostics before veto assertions.

Require:

- hard events are bitwise identical across arms; every mixed target/pixel pair passes area consistency;
- exactly the same batches are hard/mixed in both arms, with at least 80 hard and 80 mixed events;
- no non-finite loss, gradient, parameter, BN buffer, or momentum state;
- candidate terminal loss EMA is no more than 1.5x control and no candidate-only one-class concentration exceeds 95%;
- all parameters receive finite gradients under hard and alpha-0.5 probability targets;
- candidate/control shared starting state is bitwise equal and no augmented tensor is regenerated between arms.

These are catastrophic safety gates only. A favorable 200-batch loss does not override the full-phase underfit risk.

## Throughput and Fairness Gates

The transform and dense-target GPU path are structurally unchanged, so alpha should be compute-neutral. Verify with five alternating fresh-process accepted/candidate trials over persisted hard/mixed batches and the real forkserver loader. After warmup, measure at least 1,000 synchronized training steps and iterator waits.

Require candidate/control median GPU-step ratio at most 1.01, at least **26,629 projected steps** (99% of EXP-010), strong-loader delivery at least 1.20x GPU consumption, iterator-wait median below 10% and p95 below 20% of step time, integrated wall/count ratio no more than 1.07, peak allocation below 650 MiB, and projected total runtime below 540 seconds. The exact eight-worker switch and first hard weak batch must pass.

Evaluation remains untouched and at most once per epoch. No alpha-specific validation, extra terminal look, or control evaluation is allowed. Any small epoch/test-look difference is part of fixed-time execution.

## Hypothesis, Risks, and Verdict

**Hypothesis:** p=0.5 alpha-0.5 CutMix reduces per-event target/area ambiguity enough to preserve stronger plateau learning while retaining regional invariance, reaching `best_test_acc >=94.25%` with at least 99% accepted exposure.

Pre-register 87.08% as the recurring underfit marker and 89.0% as a healthier switch expectation. They are diagnostic, not mid-run vetoes or tuning signals. Compare switch accuracy, first weak checkpoint, final/best gap, and NLL against EXP-010/011.

Primary risks are lower effective regularization, many near-no-op/full-donor events, loss of moderate-size occlusion/localization, and a low effect ceiling around a successful alpha-1 point. Same-class pairings and 32x32 rectangle quantization can further weaken the intended distribution shift.

If every preflight gate passes, run seed 42 once on the sole idle H20 with output only in `run.log`. Require exit zero, approximately 300 counted seconds, total below 600, finite summary, at least 26,629 steps, one 80% switch, eight stopped workers, 48.5-51.5% realized mixed batches, hard weak targets, and unique at-most-once-per-epoch evaluations.

Accept only at `best_test_acc >=94.25%`. A valid miss rejects alpha 0.5 at p=0.5; a distribution, safety, throughput, scope, or runtime failure is a no-go/invalid result. Do not tune alpha, probability, rectangle policy, or seed, and do not rerun a valid experiment.
