# Proposal: Exact-Corpus Retry of Balanced Mixup/CutMix

## Decision and one-run hypothesis

Run the unresolved EXP-019 intervention unchanged: during the N1/M7 strong phase, use one worker-local categorical draw to select 50% hard batches, 25% alpha-1 CutMix, or 25% alpha-0.4 Mixup. At the existing 80% switch, remove RandAugment and all target mixing and retain the accepted hard crop/flip weak tail. Model, seed, optimizer, decay, LR schedule, timer, evaluator, and all other accepted semantics remain fixed.

The new work is the evidence protocol, not a new alpha or ratio. Before production, freeze one natural 200-batch stream of post-N1/M7 but **pre-policy** source tensors together with the exact worker CPU RNG state at the policy boundary. Both safety arms must clone each immutable source batch and independently apply their own policy from that same recorded state: the control applies the accepted 50% hard / 50% alpha-1 CutMix policy, while the candidate applies the 50/25/25 hard/CutMix/Mixup policy. This makes every control/candidate difference attributable to the policy on the registered corpus and fixes EXP-019's non-replayable fresh-forkserver comparison.

**Hypothesis:** complementary whole-image interpolation on half of the accepted mixed events will improve generalization without increasing the soft-target rate or materially reducing exposure, raising `best_test_acc` from 94.15% to at least 94.25%. Point prediction: **94.30%**, with at least 26,629 optimizer steps. The switch marker, NLL, and tail trajectory are diagnostic only and cannot authorize a rerun or policy adjustment.

## Why this is not an unchanged failed retry

EXP-019 did not produce a valid accuracy result. Its code semantics and 20,000-collation lifecycle gate passed, but the first safety attempt asserted on a candidate-only class-concentration event before serializing its histogram. A serialization-only rerun then used a fresh forkserver stream whose transformed tensors differed, so its pass could neither reproduce nor clear the first veto. The report therefore classified the experiment as `invalid`, explicitly described the alpha-0.4 policy as unproven rather than conclusively unsafe, and named persisted post-transform batches as the condition for reconsideration.

EXP-026 is a new pre-registered experiment that corrects that causal-measurement defect:

- it persists source tensors *before* either mixing policy rather than collecting already-hard and already-CutMix buckets;
- it records the exact worker RNG state at the policy boundary, so both arms share the same natural categorical draw and transform RNG origin;
- it applies the accepted policy to the control and the alternate policy to the candidate from cloned copies of every same source batch;
- it hashes the corpus and per-batch source/state records before model training and serializes all step evidence before evaluating vetoes;
- it does not use a fresh process, passing rerun, alternate seed, alpha, probability, or threshold to erase an observed failure.

This is a rigorous rerun of an unresolved hypothesis, not a rescue of a valid no-improvement and not a relaxation of EXP-019's gate.

## Literature grounding and evidence limits

The mixup paper defines virtual examples by applying the same `lambda ~ Beta(alpha, alpha)` convex combination to two inputs and their one-hot targets, with the intended inductive bias of linear behavior between training examples. It reports improved CIFAR-10 generalization, but its CIFAR experiments use `alpha=1`; its `alpha in [0.1, 0.4]` underfitting discussion is from ImageNet, where larger alpha and short/small-model training were less favorable. Thus alpha 0.4 is a locally pre-registered compromise from EXP-019's adversarial review, not a paper-validated optimum for this exact ResNet-20 horizon. Source: `knowledge/papers/mixup.md`; [Zhang et al., ICLR 2018](https://arxiv.org/abs/1710.09412).

The CutMix paper samples `lambda ~ Beta(1,1)`, replaces a rectangular area, and adjusts target mass to the realized visible area. It argues that regional replacement retains usable pixels and encourages spatially distributed evidence; its CIFAR ablation favored alpha 1 and reported CutMix above Mixup in its own long PyramidNet setting. This supports preserving the accepted alpha-1 CutMix branch, not the claim that a Mixup/CutMix mixture must be synergistic. Source: `knowledge/papers/cutmix.md`; [Yun et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.html).

The strongest local evidence is compositional: EXP-010 gained 0.60 points from 50% alpha-1 CutMix during the strong phase, while EXP-011 lost 0.15 points when total CutMix probability rose to 75%. The candidate therefore holds the total mixed rate at the accepted 50% and changes only half of its geometry. The main risk is still underfit: alpha-0.4 Mixup changes target/pixel strength as well as geometry, and neither paper tests this 25/25 composition on N1/M7 under a 300-second budget. A miss cannot isolate geometry, alpha, or split.

## Exact production implementation

Modify only tracked `train.py`. Add module-level torchvision v2 MixUp and explicit provenance constants:

```python
MIXUP_ALPHA = 0.4
CUTMIX_PROBABILITY = 0.25
MIXUP_PROBABILITY = 0.25

cutmix = v2.CutMix(alpha=CUTMIX_ALPHA, num_classes=NUM_CLASSES)
mixup = v2.MixUp(alpha=MIXUP_ALPHA, num_classes=NUM_CLASSES)

HARD, CUTMIX, MIXUP = 0, 1, 2

def apply_strong_policy(inputs, targets):
    u = torch.rand(()).item()
    if u < CUTMIX_PROBABILITY:
        inputs, targets = cutmix(inputs, targets)
        kind = CUTMIX
    elif u < CUTMIX_PROBABILITY + MIXUP_PROBABILITY:
        inputs, targets = mixup(inputs, targets)
        kind = MIXUP
    else:
        kind = HARD
    return inputs, targets, kind

def mixed_collate(batch):
    inputs, targets = default_collate(batch)
    with torch.random.fork_rng(devices=[]):
        return apply_strong_policy(inputs, targets)
```

Use exactly one categorical draw. Do not use independent Bernoulli draws, Python/NumPy/CUDA RNG, manual lambdas or permutations, rejection sampling, shared worker counters, or worker reseeding. Keeping the gate and chosen v2 transform inside `fork_rng(devices=[])` preserves the accepted property that policy RNG consumption cannot perturb subsequent RandomCrop/flip/RandAugment samples. The draw is nevertheless natural: it begins from the worker's real post-transform CPU state on that batch.

The strong loop conditionally unpacks `(inputs, targets, kind)`, validates provenance against target form, and increments explicit hard/CutMix/Mixup counters. `targets.ndim` may validate targets but must not infer geometry because both mixed branches return floating `[B,10]` targets. At the existing switch, print all counts and denominator once. The weak loader retains default two-item collation and must yield FP32 images plus one-dimensional int64 targets.

Everything else remains byte-for-byte or semantically identical to the accepted source: seed 42; width-2 postactivation ResNet-20 with 1,073,962 parameters; batch 128; ordinary SGD momentum 0.9; all-parameter weight decay `1e-4`; `lr=0.1` through 80%, then the existing abrupt `0.01` cosine tail to `1e-4`; 300 counted seconds; FP32; Option-A shortcuts; evaluator cadence; and worker lifecycle.

## Natural immutable source corpus

Create a new experiment-scoped controller and corpus; EXP-024's stored corpus is unsuitable because its strong buckets already contain hard or CutMix policy outputs. The new corpus must be generated once, before either model arm, as follows:

1. In a fresh importable process, set seed 42 exactly as production. Recreate the source-matched N1/M7 transform declaration, call imported production `make_train_loader`, and assert batch size 128, shuffle, eight persistent forkserver workers, pinned memory, drop-last, and dataset root. Its temporary top-level `source_collate` does only `default_collate(batch)` and then records `torch.get_rng_state()`, `get_worker_info().id`, and `torch.initial_seed()`; it performs no policy draw.
2. Mirror production RNG ordering before `iter(loader)`: create the loader, construct the accepted `ResNet(3,10,2)` and SGD objects in the same order as `main`, then create the iterator. This matters because model initialization and DataLoader iterator seeding share parent CPU RNG.
3. Take the first 200 sequential source batches with no filtering, quota filling, branch forcing, rejection, or rematerialization. For each ordinal store contiguous cloned FP32 inputs, int64 hard targets, the post-collate worker CPU RNG state that production policy would see, worker provenance, and SHA-256 digests of all three tensors. Shut down all eight workers explicitly.
4. Write the complete corpus atomically, compute its file SHA-256, reload it read-only, validate 200 unique records and tensor/state digests, and record PyTorch/torchvision versions, transform parameters, loader settings, seed/order facts, worker IDs/seeds, shapes, dtypes, label ranges, and corpus path. Do this before constructing either safety arm.

The controller predicts each natural gate `u` by restoring a record's policy state inside `fork_rng(devices=[])`. The fixed corpus must naturally contain at least 35 candidate instances of each geometry; accepted and candidate total-mixed decisions must be exactly equal because both use `u < 0.5`. If the fixed 200 batches miss a count floor, mark the experiment invalid and do not generate a friendlier corpus.

## Source-batch semantic comparison

For every corpus record, independently clone the immutable source `(inputs, hard_targets)` for each arm. Inside a CPU-only `fork_rng(devices=[])`, restore that record's policy RNG state and apply:

- **control:** draw `u` once; for `u < 0.5`, invoke the imported alpha-1 `cutmix`; otherwise return hard data;
- **candidate:** invoke imported production `apply_strong_policy`, which maps `u < 0.25` to CutMix, `0.25 <= u < 0.5` to Mixup, and the rest to hard.

Before training, require the following across all 200 records:

- the predicted `u` and mixed/not-mixed decision match exactly between arms;
- for `u < 0.25`, control and candidate post-policy inputs and targets are bitwise equal because both consume the same gate draw and CutMix implementation from the same state;
- for `u >= 0.5`, both outputs are bitwise equal to the source hard batch;
- only `0.25 <= u < 0.5` differs, with control producing valid area-adjusted CutMix targets and candidate producing valid alpha-0.4 Mixup inputs/targets using torchvision's roll pairing;
- hard targets are int64 `[128]`; mixed targets are finite floating `[128,10]`, nonnegative, and row-normalized; controlled unique-label fixtures separately prove pixel/target pairing and nondegenerate Mixup/CutMix behavior;
- both applications leave surrounding CPU RNG unchanged, consume no CUDA RNG, and leave every stored source/state tensor and digest unchanged.

Persist per-step ordinal, worker provenance, `u`, control/candidate kind, pre/post hashes, target checks, and aggregate counts. Serialize this semantic report before any assertion. This comparison is the core correction: the control must not replay a stored already-CutMix batch while the candidate remixes it, and the two arms must never obtain source tensors from different fresh loaders.

## Paired optimization safety gate

Create one seed-42 initial model state and hash it, then launch explicit fresh control and candidate processes that load the same initial state, empty ordinary-SGD state, and immutable corpus SHA. Use deterministic cuBLAS configuration and apply each arm's registered post-policy sequence as above. Train for all 200 steps at fixed `lr=0.1` and require:

- exact initial parameters, buffers, optimizer structure, CPU/CUDA state declarations, and 1,073,962 parameters;
- no non-finite source, policy output, target, logit, loss, gradient, parameter, BN buffer, or momentum state;
- exact BN batch counters of 200 and complete momentum state;
- candidate terminal debiased loss EMA no greater than 1.5 times control;
- no step where candidate maximum predicted-class share is greater than 95% while control is at or below 95%;
- at least 35 natural candidate hard, CutMix, and Mixup batches, identical total mixed decisions, and no provenance/target mismatch.

Record both full loss and class-share arrays plus every veto event, set report status, write and fsync the report, and only then raise on failure. A passing fresh rematerialization is forbidden. The safety probe calls no evaluator and its loss/concentration signals are integrity gates, not accuracy surrogates.

Separately rerun the unchanged 20,000-collation production-loader gate to verify 48.5-51.5% hard and 23.5-26.5% each CutMix/Mixup, valid provenance/targets, no worker growth, exact shutdown of eight workers, weak-loader rebuild below five seconds, and a hard two-item first weak batch.

## Timing and exposure gates

Only after semantic and safety reports pass, confirm exactly one idle NVIDIA H20 near 97,871 MiB. Run one unscored conditioner followed by five alternating fresh-process accepted/candidate timing pairs. Each arm uses the real eight-worker strong loader and its explicit natural production collator, at least 100 warmups and 1,000 synchronized training steps, then a measured weak segment and explicit worker shutdown. No evaluator is called.

Require:

- aggregate candidate/control synchronized counted-step ratio at most 1.01, every-pair ratio at most 1.04, and per-arm CV below 3%;
- conservative projected exposure `floor(26898 * control_mean / candidate_mean) >= 26629`;
- candidate warmed loader delivery at least 1.20 times GPU consumption, median iterator wait below 10% and p95 below 20% of GPU-step time;
- integrated wall/count ratio at most 1.07 and no more than 0.02 above paired control;
- natural geometry proportions within the registered intervals, peak allocation below 650 MiB, no worker or allocation growth, weak rebuild below five seconds, and projected total runtime below 540 seconds.

The exact corpus proves semantics; the alternating real-loader pairs measure production throughput. Fresh timing streams are acceptable because timing is aggregated and is not used to clear a trajectory-dependent safety veto. Do not move mixing to GPU, change workers/prefetch, force a backend, or relax exposure after observing a miss.

## Production and verification

After all gates pass:

1. Confirm baseline 94.15 at `7c1e7d8`, only `train.py` tracked-modified, no stale `run.log` variant, one idle H20, and no competing compute process. Pass `py_compile`, Ruff, format/diff, AST/source invariants, exact model/optimizer/schedule checks, and `git diff --check`.
2. Run seed 42 exactly once as `timeout -k 5s 595s uv run train.py > run.log 2>&1`. Poll only bounded status/tail lines. Never rerun a completed valid result.
3. Require exit zero, ten finite summary fields, `training_seconds` approximately 300, `total_seconds < 600`, 1,073,962 parameters, at least 26,629 steps, one switch near 80%, eight strong workers stopped, hard weak targets, and no repeated evaluation epoch.
4. Require production strong proportions of 48.5-51.5% hard and 23.5-26.5% each CutMix/Mixup. Report switch and first-weak accuracy, best/final epoch and accuracy, final NLL, exposure, VRAM, all counts, and timing/corpus hashes.
5. Accept only if `best_test_acc >= 94.25%`. A finite lower result is `no-improvement`; a protocol/gate failure is `invalid`. Do not retry with alpha 0.2/1.0, another ratio or seed, a forced corpus, altered threshold, or pure CutMix fallback.

The pre-registered 87.08% strong-underfit marker and accepted 89.73% switch value help interpret a valid result but do not affect its formal verdict. A bare ten-example pass is weak causal evidence; trajectory and NLL should be reported honestly without changing the metric rule.

## Abort criteria

Abort before production for any tracked edit outside `train.py`; evaluator, budget, seed, model, optimizer, schedule, or weak-tail change; more than one gate draw; non-natural/forced corpus selection; corpus/digest mutation; source mismatch between arms; failure to apply the accepted control policy from the same pre-policy source; lost shared-branch bitwise equivalence; target/provenance/RNG failure; report written after rather than before assertion; any registered safety, timing, exposure, memory, lifecycle, or wall gate miss; or GPU contention. An abort retires only this execution under its stated protocol. It does not permit an in-place rescue.

## Risk assessment

- **Scientific risk — medium-high:** the two papers establish distinct regularizers, not synergy, and alpha 0.4/split 25% is not validated in this short recipe. Replacing proven CutMix exposure may lower localization-oriented generalization.
- **Optimization risk — medium:** total soft frequency is unchanged, but global interpolation can deepen the strong-phase underfit that EXP-011 exposed. Exact-corpus concentration and loss gates catch gross transients, not subtle accuracy harm.
- **Implementation risk — medium:** capturing worker state at precisely the post-transform/pre-policy boundary and mirroring parent RNG ordering are essential. Corpus hashes and shared-branch bitwise checks make mistakes observable.
- **Runtime risk — low:** Mixup's full-image CPU interpolation is heavier than rectangular replacement, but eight-worker prefetch should hide it; the real-loader timing gate protects exposure and wall time.
- **Estimated effort:** high for controllers and evidence, low for production code. This cost is justified because it converts EXP-019's non-replayable invalid result into a genuinely attributable one-run test.
