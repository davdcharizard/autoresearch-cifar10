# Proposal: Original Option-B Strided Projection Shortcuts

## Decision and falsifiable hypothesis

Replace exactly the two accepted CIFAR Option-A dimension-changing shortcuts with the original ResNet Option-B form: a bias-free stride-2 `1x1` convolution followed by `BatchNorm2d`. Keep the seven same-shape shortcuts as exact identities and leave both stride-2 residual branches, every same-shape residual branch, widths, depth, classifier, data policy, optimizer, schedule, timer, evaluator, and worker lifecycle unchanged.

This is explicitly **not** another pool-first ResNet-D experiment. A stride-2 `1x1` projection reads `x[..., 2i, 2j]`, the same spatial lattice selected by the accepted `x[:, :, ::2, ::2]` shortcut. The intervention changes only channel transport at that lattice: Option A copies the old channels and zero-fills the newly introduced channels, whereas Option B learns a dense map from every input channel to every output channel and normalizes that map. There is no average pool, blur, larger kernel, residual-path change, or altered sampling footprint.

**Hypothesis:** learned full-channel transport at the two stage boundaries will remove Option A's zero-filled shortcut subspace and give newly introduced stage channels a direct normalized signal, improving late representation quality enough to raise `best_test_acc` from 94.15% to at least **94.25%**, while retaining at least **26,000** optimizer steps. The point prediction is **94.28%** with roughly 26,500 steps. This is falsified by any valid run below 94.25%; safety or timing vetoes make the operating point invalid rather than an accuracy failure.

## Evidence, novelty, and limits

The original ResNet paper defines zero-padded identity as Option A, projection only when dimensions increase as Option B, and all projections as Option C. It used economical Option A for its reported CIFAR models and Option-B stage transitions for ImageNet models. Thus the paper establishes that this is a canonical dimension-matching construction, but does **not** establish a CIFAR-10 advantage, much less an advantage under this seed-42, width-2, N1/M7 plus CutMix, 300-second recipe. Literature status is architectural legitimacy, not local efficacy evidence.

The accepted model has two dimension-changing blocks, `32 -> 64` at `layer2[0]` and `64 -> 128` at `layer3[0]`. Its Option-A paths preserve only 32/64 old channels respectively and insert zeros into the other 32/64 shortcut channels. Option B supplies all 64/128 output channels from learned combinations of all 32/64 inputs. Because the residual paths are untouched, any difference is attributable to the shortcut channel basis and its BN state, not added residual capacity.

Prior local evidence makes the proposal distinct but high-risk:

- EXP-017 tested `AvgPool2d(2,2) -> stride-1 1x1 Conv -> BN`, not Option B. It retained 26,557 steps and improved switch accuracy from the accepted 89.73% to 90.20%, but peaked at 94.09% and worsened final NLL to 0.2024. Pooling and learned normalized channel transport changed together, so that result cannot identify which caused the late deficit.
- EXP-021 isolated pool-first filtering with deterministic Option-A channel semantics. It produced candidate-only one-class transients at steps 17-18 on an immutable corpus. Together with EXP-017, this retires pool-first shortcuts; it does not test the same-lattice stride-2 projection proposed here.
- EXP-024's 32/64/160 model altered final width, classifier width, residual convolutions, Option-A identity fraction, parameter count, and RNG trajectory together, then hit a step-2 concentration event. Option B preserves all accepted widths and residual branches, so it is not a width rescue.
- EXP-025 showed that exact-function initialization does not guarantee stable recruitment for globally trained ECA logits. Option B is different: it is deliberately active from the first forward pass, uses ordinary Kaiming convolution weights and BN, and has no sigmoid gate that can saturate. Its active random normalized basis is nevertheless an early-trajectory risk that must be measured rather than assumed safe.

The most informative comparison is therefore Option B versus both the accepted Option A and EXP-017. If Option B avoids EXP-021's early transient but repeats EXP-017's higher switch fit and worse late NLL, the learned projection/BN basis—not average pooling—is the likely harmful component. If it improves late NLL and top-1 while EXP-017 did not, preserving the accepted sampling lattice is the supported mechanism.

## Exact `train.py` implementation

Modify only tracked `train.py`. Add a marker subclass for the two projection convolutions so their initialization can be isolated from the accepted model's global initialization stream:

```python
class ShortcutConv(nn.Conv2d):
    pass
```

Give `BasicBlock` access to the one model-owned shortcut generator. For a block with `stride == 2 and in_channels != out_channels`, construct exactly:

```python
nn.Sequential(
    ShortcutConv(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=2,
        padding=0,
        bias=False,
    ),
    nn.BatchNorm2d(out_channels),
)
```

For every other block, use `nn.Identity()`. In `forward`, compute the unchanged residual branch, add `self.shortcut(x)`, and apply the unchanged post-add ReLU. Delete or bypass only the transition slice-and-zero-pad code. Do not retain `need_pad`, use a conditional fallback at runtime, or apply a projection to same-shape blocks.

Mirror EXP-017's already reviewed initialization-isolation protocol so this experiment differs from EXP-017 only in pool removal and projection stride:

1. Create one CPU `torch.Generator` deterministically from the active `torch.initial_seed()` in `ResNet` and consume it sequentially for the `layer2[0]` then `layer3[0]` projections.
2. Construct each `ShortcutConv` inside `torch.random.fork_rng(devices=[])`, preventing its default constructor initialization from advancing the accepted global CPU RNG.
3. Kaiming-normal initialize each projection weight with that dedicated generator.
4. Make the model-wide `_weights_init` return without reinitializing `ShortcutConv`; retain its accepted behavior for every ordinary Conv/Linear tensor.
5. Leave shortcut BN at PyTorch defaults: gamma one, beta zero, running mean zero, running variance one, default momentum/epsilon, and tracked statistics.

This produces ordinary Kaiming Option-B weights without shifting any accepted shared parameter draw or post-construction CPU RNG state. It also permits a clean pool-first comparison because the projection weights follow the same sequential generator protocol used by EXP-017. Do not use zero BN gamma, partial-identity weights, copied Option-A rows, projection-specific LR/decay, a secondary tunable seed, or an identity warm-up; each would be a different mechanism.

Expected parameter count is exactly **1,084,586**, up from 1,073,962 by 10,624 trainable parameters:

- `32 -> 64` projection: 2,048 convolution weights and 128 BN affine values;
- `64 -> 128` projection: 8,192 convolution weights and 256 BN affine values.

The two shortcut BNs also add their ordinary running-stat buffers and counters. All parameters, including the new convolution and BN affine tensors, remain in the single accepted SGD group with LR 0.1, momentum 0.9, and weight decay `1e-4`.

Everything else must remain semantically identical to accepted `train.py`: width-2 postactivation ResNet-20; nine residual blocks and 19 residual/stem convolutions; batch 128; FP32; seed 42; standard SGD; alpha-1 CutMix on 50% of N1/M7 strong batches; hard crop/flip weak tail at 80%; the elapsed-time LR transition and cosine tail; 300 counted seconds; `MAX_STEPS=64000`; evaluation cadence; one H20; and explicit strong-worker shutdown before weak-loader construction.

## Initialization and recruitment gates

Option B cannot preserve the accepted initial function: BN standardizes a random dense projection and makes it fully active at step zero. That is the intended hypothesis, not an implementation defect. The relevant safety question is whether the active shortcut has a controlled scale, finite gradients, and stable early updates.

Before any trajectory test, an explicit seed-42 accepted/candidate constructor comparison must prove:

- exactly two stride-2 `1x1` `ShortcutConv + BN` paths at `layer2[0]` and `layer3[0]`, and exactly seven identity shortcuts;
- no average-pooling module or operation, no shortcut padding, no projection elsewhere, and unchanged residual branch source/shape/stride/order;
- bitwise equality of every shared parameter and buffer at construction, plus bitwise equality of post-construction CPU and CUDA RNG states;
- projection tensors equal the expected sequential draws of the dedicated generator, nonidentical to each other, finite, and Kaiming-scaled;
- exact parameter count, optimizer membership once per tensor, default BN state, and unchanged logits for an independently constructed accepted control oracle.

Use coordinate-ramp and channel-basis fixtures to prove exact sampling semantics. For a transition input, changing any pixel outside the `[..., ::2, ::2]` lattice must not change the Option-B shortcut output in eval mode. A basis impulse on a sampled coordinate must affect all output channels according to the corresponding projection column; an unsampled-coordinate impulse must have no effect. This distinguishes true Option B from pool-first ResNet-D and catches accidental stride/padding errors.

On immutable production-distribution hard and CutMix batches, record pre-update shortcut and residual RMS for both transitions, losses, logits, class histograms, gradient norms, parameter/update norms, BN batch counters, running means/variances, and replay loss. Both projections and both shortcut BN affine tensors must receive finite nonzero gradients and finite nonzero updates on hard and soft first-update probes. Use the same catastrophic tripwires as EXP-017: shortcut RMS must remain within `[0.25, 4.0]` times its paired residual RMS; an individual projection update may not exceed 25% of its pre-update norm; and replay loss may not exceed 2x its own pre-update loss or the paired-control replay loss. RMS above 2x is a recorded warning, not a tunable post-hoc veto. The load-bearing gates are update size, finite state, loss, and candidate-only concentration.

Recruitment is supported only if both projection weights and BN affine tensors update, both shortcut BN counters advance exactly once per training forward, running variances stay finite and positive, and ordinary SGD momentum exists for every new trainable tensor after the probe. There is no requirement for gradual emergence or near-identity behavior. Conversely, rapid scale growth, dead projection gradients, BN variance collapse, or a candidate-only class transient rejects this active-initialization point; do not repair it in-place with zero gamma or a smaller shortcut LR.

## Immutable-corpus paired safety

Reuse the read-only EXP-024 production corpus rather than generating a friendlier stream. It contains 100 N1/M7 hard batches, 100 N1/M7 alpha-1 CutMix batches, and 100 weak hard batches, with file SHA-256 `d4294f5adb2e58e0847366231458b21901c6f01f270d4cd1c9eae14a05b64565`. Recompute and require that digest before loading, validate every tensor/target digest and dtype/shape/label invariant, and never overwrite, filter, rebucket, or regenerate it. The architecture-only candidate and control must consume cloned byte-identical inputs and targets in the same registered order. The weak set is structural/diagnostic only; no evaluator is called.

Create one accepted seed-42 initial state and then construct explicit fresh Option-A control and Option-B candidate processes with deterministic backend declarations. Restore the common tensors exactly, retain the candidate's independently registered projection tensors, start both with empty ordinary-SGD state, and verify all shared state/RNG hashes before step 1. Replay all 200 strong records at fixed LR 0.1, collecting every loss, prediction histogram, branch scale, update norm, BN statistic, and state hash needed to diagnose a failure. Serialize and fsync the full report before evaluating veto assertions.

Require:

- all 200 steps complete in each arm with finite inputs, targets, logits, losses, gradients, parameters, buffers, and momentum;
- exact source/target/corpus identity and deterministic-backend parity;
- candidate BN counters equal 200 for every ordinary and shortcut BN, with complete momentum state for every trainable parameter;
- no step at which candidate maximum predicted-class share exceeds 95% while control is at or below 95%; report all shares even when both arms are concentrated;
- candidate terminal debiased loss EMA no greater than 1.5x control;
- no projection update above 25% of its immediately pre-update norm, no nonpositive/nonfinite running variance, and no shortcut/residual RMS outside `[0.25,4.0]` at registered checkpoints;
- immutable corpus and accepted shared-state hashes remain unchanged after both arms.

The safety gate is not an accuracy surrogate. Lower candidate loss cannot clear a concentration or scale veto, as EXP-021 and EXP-024 demonstrated. A passing trajectory also cannot erase EXP-017's valid 94.09% result; it only establishes that Option B is safe enough for timing and one scored run.

## Compute cost, exposure, and timing

The new projections perform about 1.05 million multiply-accumulates per image in total: `32*64*16*16` at the first transition plus `64*128*8*8` at the second. At batch 128 that is about 134 million forward MACs, plus backward and two BN paths. The accepted slice/pad path is nearly free, so some slowdown is unavoidable. However, EXP-017's pool-first projections have the same projection output shapes and weight count, plus an average-pool operation, and still completed 26,557 steps (98.73% of accepted EXP-010's 26,898). Option B should be no slower than that nearby implementation absent kernel-selection noise. Memory headroom is ample, but fixed-time exposure and evaluation wall cost remain binding.

After safety passes and exactly one idle 97,871 MiB H20 is confirmed, run one unscored conditioning process followed by five alternating fresh-process Option-A/Option-B timing pairs. Each arm must use explicit constructors, identical seed/backend settings, real eight-worker production strong loading, at least 100 warm-up steps and 1,000 synchronized measured training steps including H2D, forward, loss, backward, zero-grad, SGD, and synchronize. Then measure a real weak segment, inference/evaluation-region forward cost, loader rebuild, shutdown, and allocation behavior. Do not use synthetic-only timing, omit BN backward, compile, enable autocast/channels-last/autotune only for one arm, or share a model/process between paired trials.

Pre-register these timing gates:

- aggregate candidate/control counted-step mean ratio `<=1.03`, every pair `<=1.06`, and per-arm trial-mean CV `<3%`;
- conservative projected exposure `floor(26898 * control_mean / candidate_mean) >=26,000`;
- candidate median/p95 synchronized step time and iterator wait fully reported, with warmed loader delivery at least 1.2x GPU consumption, median wait below 10%, and p95 wait below 20% of candidate GPU-step time;
- peak allocation `<700 MiB` and no more than 96 MiB above control, with no monotonic allocation or worker growth;
- inference ratio `<=1.06`, projected production evaluations `<=19`, weak-loader rebuild `<5s`, integrated wall/count `<=1.07`, and projected total runtime `<540s`;
- exactly eight strong workers stop before one valid hard weak batch, and no live child remains after weak shutdown.

The 26,000-step floor is the mechanism-attribution floor, not a substitute for the metric. A valid accuracy pass below it would be formally metric-positive but exposure-confounded and must not be rerun. A timing miss retires this exact operating point; it does not authorize removal of BN, projection of only one stage, or a looser exposure floor.

## Expected mechanism and failure discriminators

The hypothesized mechanism is direct learned transport into the new half of each stage's channels. With Option A, those shortcut coordinates start at zero and every newly introduced channel depends entirely on the residual branch. With Option B, all output channels receive a normalized linear combination of the preceding representation, potentially improving gradient transport and preserving useful semantic combinations across stage changes without deleting spatial samples beyond the accepted decimation.

The following outcomes distinguish competing explanations:

- **Early scale/concentration veto:** the random normalized Option-B basis perturbs logit geometry too aggressively under global LR 0.1. This rejects active Kaiming Conv+BN recruitment, even if loss is lower.
- **Healthy safety, higher switch fit, worse final NLL/top-1:** reproduces EXP-017 without pooling and implicates learned projection/shortcut BN channel transport as the late-generalization liability.
- **Healthy safety, accepted-like switch fit, no accuracy gain:** Option A's zero-filled channel subspace is not a limiting representation bottleneck in this short recipe.
- **Lower step exposure and lower switch fit:** added projection backward cost, rather than representation quality alone, starves the fixed-time strong phase; actual steps and paired timing quantify the confound.
- **Better NLL but peak below 94.25:** Option B changes confidence/calibration without a sufficient classification-boundary gain; this is still no-improvement.
- **At least 94.25 with healthy NLL, switch/tail trajectory, and at least 26,000 steps:** supports learned same-lattice channel transport as the causal mechanism. A marginal 94.25-94.35 result remains single-seed noise-consistent weak evidence and must be reported as such.
- **Option B succeeds where EXP-017 failed:** preserving the accepted `::2` sampling lattice, rather than averaging all four pixels, is the leading discriminator. It does not prove all projection shortcuts superior.

Record switch accuracy, first-weak accuracy, best/final accuracy and epochs, final NLL, strong/tail step counts, projection/BN scale trajectories, evaluation count, peak memory, and total wall. These explain a result but never alter the formal threshold or authorize a rerun.

## Production and verification

Only after all static, semantic, immutable-corpus, safety, timing, exposure, memory, inference, and lifecycle gates pass:

1. Confirm moving baseline 94.15 at `7c1e7d8`, exactly one idle H20, only `train.py` tracked-modified, no stale `run.log` variant, and no competing compute process. Pass `py_compile`, Ruff, format/diff, `git diff --check`, exact topology/parameter/optimizer assertions, and complete source inspection.
2. Run seed 42 exactly once under the goal supervisor as `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`. Do not stream the full log, early-stop for weak accuracy, or retry a valid completion.
3. Require exit zero, the complete finite ten-field summary, approximately 300 counted training seconds, total time below 600 seconds, exactly 1,084,586 parameters, one augmentation switch near 80%, exactly eight strong workers stopped, hard weak targets, no repeated evaluation epoch, and at most 19 evaluations.
4. Require at least 26,000 actual optimizer steps for clean mechanism attribution. Parse all registered diagnostics and compare against EXP-010 and EXP-017 without changing the verdict rule.
5. Accept only if `best_test_acc >=94.25%`. Any complete finite lower result is `no-improvement`; any preflight/protocol failure is `invalid`; a crash is recorded without an architecture rescue. Never rerun a valid result.

## Abort criteria

Abort before production for any tracked change outside `train.py`; evaluator, seed, budget, data policy, optimizer, LR schedule, residual branch, width/depth, classifier, precision/layout, evaluation cadence, or worker lifecycle change; any pool/blur or non-`1x1` projection; projection on a same-shape block; bias in a projection; missing shortcut BN; nondefault BN start; shared-state or RNG drift; wrong parameter count; corpus/digest/source mismatch; unpersisted failure evidence; nonfinite state; unrecruited new parameter; registered concentration/loss/update/scale/BN failure; timing, exposure, memory, inference, wall, or shutdown miss; or GPU contention.

Do not rescue a failure within EXP-027 by trying pool-first, only one projection, Option C, zero gamma, partial-identity initialization, no BN, a smaller shortcut LR, excluded decay, another seed, relaxed concentration threshold, regenerated corpus, altered timing threshold, or an evaluation/data change. Each changes the hypothesis and would require new ideation.

## Risk assessment

- **Scientific risk — high:** the original paper chose Option A for CIFAR, and the closest local learned projection experiment improved early fit but slightly worsened late NLL and top-1. The expected effect is small relative to single-run noise.
- **Optimization risk — medium-high:** default BN makes both random projections active immediately. Exact shared initialization isolates attribution but cannot remove the candidate's genuinely different early function.
- **Runtime risk — low-medium:** two projection/BN backward paths add real counted work, although EXP-017 indicates exposure should remain near 26.5k steps and Option B removes its average pool.
- **Implementation risk — medium:** an ordinary constructor or model-wide initializer can silently shift shared RNG draws; explicit marker/init and bitwise shared-state gates are mandatory.
- **Attribution quality — high if all gates pass:** same-lattice sampling cleanly separates original Option B from both pool-first failures, while immutable data and common-state parity isolate the shortcut intervention.
- **Estimated effort:** medium. The production diff is small, but architecture safety, initialization parity, paired timing, and evidence-before-assertion controllers remain essential.
