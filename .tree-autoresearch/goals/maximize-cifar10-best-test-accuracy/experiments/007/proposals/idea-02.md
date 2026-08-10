# Proposal: Literature-Scale ASAM in the Validated Clean Tail

## Summary

Replace only EXP-004's sparse late SAM perturbation geometry with element-wise p=2 ASAM. Preserve the first 75% exactly: independent 256-image batches, parent CutMix gate and RNG, WRN, optimizer, LR, and drop path. Preserve the final-quarter cadence at every second eligible step, the two-pass RNG/BatchNorm safeguards, and exactly one Nesterov update. Use the literature CIFAR values `rho=0.5` and `eta=0.01` without a scalar search.

This is a matched ASAM-package comparison against the accepted SAM parent. It does not add ASAM earlier or more often; it asks whether scale-aware late perturbations find a better clean solution at the same expensive-pass dose.

## Motivation and Evidence

EXP-004 established that a clean-tail SAM pulse every second eligible step improves the CutMix parent by 0.17 points, reaching 95.40% with 2,449 pulses and 25,560 total updates (`experiments/004/04-analysis.md`). The current bottleneck is detectable generalization gain amid at least 0.15-point tail variation, not throughput or memory (`02-system-understanding.md`). EXP-006 preserved throughput but exchanged a validated input regularizer for weak hidden mixing and gained only 0.01 points; ASAM instead leaves every accepted data and augmentation dose intact (`experiments/006/04-analysis.md`).

Kwon et al. report ASAM-over-SAM gains of 0.20 on WRN-28-2, 0.30 on WRN-28-10, 0.24 on ResNet-56, and 0.46 on ResNeXt29-32x4d using CIFAR defaults `rho=0.5`, `eta=0.01` (`knowledge/papers/adaptive-sharpness-aware-minimization.md`). Scale adaptation is relevant because convolutional, BatchNorm, and classifier weights have different magnitudes, so a single Euclidean SAM radius need not represent a comparable functional neighborhood across parameter groups.

The evidence is imperfect: published ASAM runs throughout training, whereas this recipe applies it only to half of the last quarter. The modern sharpness study also warns that adaptive sharpness is not a universal predictor of generalization (`experiments/007/papers/modern-sharpness-generalization.md`). The proposal therefore relies on the direct optimizer results, not on the claim that a lower sharpness measurement must improve accuracy.

## Exact ASAM Geometry

Let `w_i` be an unperturbed parameter tensor and `g_i` its first-pass loss gradient. Define an element-wise scale

```text
s_i = abs(w_i) + eta    for every parameter whose name does not end in ".bias"
s_i = 1                 for bias parameters
eta = 0.01
```

This adapts convolution, linear, and BatchNorm weight tensors, including one-dimensional BatchNorm gamma, while leaving all `.bias` tensors unadapted as recommended by the ASAM reference. Scale must be computed from the exact unperturbed snapshot, not from a parameter after partial mutation.

For p=2 ASAM, compute one global FP32 denominator over all trainable parameters:

```text
D = sqrt(sum_i ||s_i * g_i||_2^2)
```

and perturb each parameter by

```text
epsilon_i = rho * s_i^2 * g_i / (D + ASAM_EPS)
rho = 0.5
ASAM_EPS = 1e-12
w_i <- w_i + epsilon_i
```

The second scale multiplication is mandatory. `rho*s*g/D` would be ordinary normalized scaled-gradient perturbation, not element-wise p=2 ASAM. With nonzero `D`, the constructed adaptive-coordinate radius is

```text
sqrt(sum_i ||epsilon_i / s_i||_2^2) = rho = 0.5
```

The first-pass gradient is pure training-loss gradient. PyTorch SGD's `weight_decay=1e-4` remains applied only when the restored base parameters receive the single optimizer step from the second-pass gradient; weight decay must not be folded into `g_i` or the ASAM direction.

## Concrete `train.py` Change

Modify only `train.py` from EXP-004.

1. Replace `SAM_RHO=0.05` with explicit `ASAM_RHO=0.5`, `ASAM_ETA=0.01`, `ASAM_EPS=1e-12`, and `ASAM_RADIUS_TOL=1e-3`. Keep `SAM_START=0.75` and `SAM_PERIOD=2` unchanged so only geometry changes.
2. Build an ordered list of `(name, parameter, is_bias)` from `model.named_parameters()`. Preserve the current requirement that every trainable parameter has a gradient.
3. Retain preallocated exact parameter snapshots. Add preallocated FP32-or-parameter-dtype scale and direction buffers with matching shapes and preserved memory format. At each scheduled pulse:
   - snapshot every parameter before computing scales;
   - fill bias scales with one and other scales with `abs(snapshot)+0.01`;
   - form `s*g` in FP32 and compute global `D`;
   - reject nonfinite or nonpositive `D`;
   - form `epsilon=rho*s*(s*g)/(D+eps)`, verify the adaptive-radius contract, then add it to parameters;
   - return audit values before the second pass.
4. Keep the current exception-safe sequence: clear first gradients, replay saved CUDA RNG, disable BatchNorm running-stat tracking only for the second pass, run a separately autocast clean forward/loss, backpropagate, restore BatchNorm flags and exact parameter snapshots in `finally`, then call `optimizer.step()` once.
5. Preserve the same inputs and hard targets for both ASAM passes. CutMix remains impossible after 75%, and the existing no-overlap assertion stays active. Early batches and non-ASAM late steps remain byte-for-logic parent behavior.
6. Rename user-facing `sam` audit text to `asam` and log the full geometry. Do not change the required final summary keys or frozen evaluator call.

Use fused foreach operations where they preserve exact ordering and dtype, but prefer clear per-tensor code over an incorrect one-scale formula. Parameters are FP32 under BF16 autocast, so gradient/scale/norm arithmetic should remain FP32; the model's forward path remains BF16 autocast.

## Safety and Correctness Contract

The literature radius is fixed before metrics are observed. A performance-independent safety check must pass on a full WRN training batch before launch and on every production perturbation:

- all scales, gradients, denominator, perturbations, and audit values are finite;
- `D > 0`;
- actual constructed `||epsilon/s||_2` lies in `[0.499, 0.501]`;
- `max(abs(epsilon/s)) <= 0.501`, which follows from the global radius but catches scaling/index errors;
- no parameter is mutated before its snapshot and scale are complete;
- all parameters are restored exactly before the optimizer step;
- perturbed loss differs from the base loss and is finite.

If this contract fails, abort and classify the experiment as an implementation/safety failure. Do not lower rho, change eta, exclude a weight group, or inspect accuracy to rescue the run. This makes `rho=0.5, eta=0.01` a single literature-derived hypothesis rather than a hidden scalar search.

Bias handling must be audited by names and element counts. Expected adapted tensors are all `.weight` parameters; expected unit-scale tensors are all `.bias` parameters. A missing gradient, duplicate parameter, or name/parameter-count mismatch is fatal.

## Cadence, RNG, and Budget Preservation

The scheduling predicate remains exactly

```text
progress >= 0.75 and next_one_based_step % 2 == 0
```

Thus the first 75% retains the full parent CutMix exposure, and the final quarter retains one ASAM second pass for every two eligible batches. Saving CUDA RNG before the first pass and restoring it before the perturbed pass preserves identical drop-path masks. BatchNorm running buffers update only on the primary pass. There is one optimizer update per batch, from the second gradient on ASAM pulses and the ordinary gradient otherwise.

All scale, norm, perturb, second-forward, restoration, and optimizer work stays inside the existing charged `t0` through CUDA synchronization interval. ASAM adds several element-wise passes over 2.75M parameters on the same 2,449-ish scheduled pulses but no extra model forward beyond the parent's SAM second pass. H20 memory bandwidth and the roughly 97 GB capacity make this modest relative to the second forward/backward. Two additional model-sized buffers add roughly 22 MiB to the parent's 1,190.5 MiB peak. Expected throughput loss versus EXP-004 is below 3%, with at least 24,000 total optimizer steps preregistered as the exposure floor.

The run remains fixed seed 42 on physical GPU 0. No generator, data sampler, transform, model forward, evaluation, or CutMix call changes. ASAM itself consumes no random numbers.

## Audit Contract

Startup config must print `optimizer_geometry=asam`, `asam_rho=0.5`, `asam_eta=0.01`, `sam_start=0.75`, `sam_period=2`, adapted/bias tensor counts, and adapted/bias element counts.

Final non-summary audit output must include:

- ASAM eligible/applied counts, ratio, first step, and first progress;
- min/mean/max adaptive radius;
- min/mean/max scaled-gradient denominator `D`;
- mean/max Euclidean perturbation norm;
- maximum observed `abs(epsilon/s)`;
- nonfinite count, radius-violation count, restoration-failure count;
- unchanged CutMix eligible/applied counts and ratio.

The expected cadence is approximately the parent: first ASAM progress `0.7500`, applied/eligible ratio `0.5000`, no CutMix/ASAM overlap, and about 2,400-2,500 pulses depending only on charged-time throughput. All failure counters must be zero and adaptive radii must stay within tolerance.

## Attribution

This experiment replaces the complete SAM perturbation package `rho=0.05, s=1` with the literature ASAM package `rho=0.5, s=|w|+0.01` for weights and `s=1` for biases. A gain is attributable to that package at fixed cadence, not to scale adaptation alone: radius and geometry change together. Isolating adaptation would require a separate matched-rho ablation, which is not part of this experiment and must not be inferred from its result.

Everything else is held fixed, so unlike EXP-006 there is no loss of validated CutMix dose, independent-image exposure, or SAM-pulse probability. Small pulse-count differences can still arise because the wall-clock budget translates geometry overhead into a slightly different number of steps; audit them rather than claiming exact phase-dose parity.

## Expected Effect and Testable Hypothesis

The preregistered exploratory effect target is **+0.30 percentage points**, targeting `best_test_acc >= 95.70%` from the 95.40% parent while retaining at least 25,000 steps. It matches the ASAM-over-SAM result on WRN-28-10 but is not claimed as a prediction from that paper: the published 0.20-0.46 range comes from full-run ASAM, whereas this sparse late schedule uses roughly one eighth of all steps. The experiment tests whether concentrating scale-aware geometry in the phase where EXP-004 already validated SAM can nevertheless reach a mechanism-sized effect.

The formal improvement gate remains 95.50%. A result from 95.50% to 95.69% is a valid improvement but below the proposal's expected effect. Final accuracy close to best and final loss at or below EXP-004's 0.1654 are supportive only, not substitutes for the primary metric.

## Strongest Risks

- Full-training literature gains may attenuate sharply when ASAM is used only on half of the final-quarter steps; the +0.30 expectation may not survive the sparse schedule.
- `rho=0.5` is ten times the parent's Euclidean SAM scalar and geometry-dependent. The adaptive-radius contract proves implementation and bounds normalized movement, not that the perturbation is optimal for this low-LR tail.
- Adapting BatchNorm gamma may disproportionately alter normalization behavior; excluding it after seeing instability would be a new experiment, not a repair.
- ASAM's extra element-wise kernels may reduce pulse and optimizer exposure more than estimated, offsetting a per-pulse benefit.
- Modern evidence shows that reparameterization-aware sharpness is not universally causal for generalization, so a correct lower-sharpness trajectory may still leave accuracy unchanged.
- The remaining 0.10-point formal gate is below observed tail/run variation. There must be one fixed-seed run, no rho/eta/cadence sweep, and no retry based on test accuracy.

## Smokes and Verification

Before launch:

1. **Closed-form geometry**: on named toy weight and bias tensors, compare `D`, `epsilon`, Euclidean norm, adaptive radius, and bias unit scale against manual FP64 calculations. Explicitly catch the missing-second-scale implementation.
2. **Scale snapshot**: prove scales use unperturbed snapshots and that no early tensor mutation affects later scales.
3. **Full WRN radius**: run a BF16/channels-last batch, verify radius `[0.499,0.501]`, normalized-coordinate bound, finite perturbed loss, and exact parameter restoration.
4. **Name coverage**: reconcile every trainable named parameter, gradient, snapshot, scale, and direction buffer; verify only `.bias` uses unit scale and all `.weight` tensors use `abs(w)+0.01`.
5. **Gradient semantics**: prove first gradients exclude optimizer weight decay, first gradients are cleared, second gradients drive the sole Nesterov update, and momentum changes once.
6. **RNG/BatchNorm**: verify primary and perturbed passes use identical drop-path masks, BatchNorm running buffers update once, tracking flags restore, and global RNG advances as one logical forward.
7. **Failure restoration**: inject a second-pass exception and verify parameter snapshots and BatchNorm flags restore exactly.
8. **Parent parity**: verify an early CutMix step and an unscheduled late step match EXP-004 code paths, counters, RNG consumption, and loss logic.
9. **Cadence/audit**: simulate boundary steps around progress 0.75 and odd/even step indices; validate every audit accumulator and zero failure counters.
10. **Runtime smoke**: measure warm parent-SAM versus ASAM pulse latency without accuracy feedback; require projected `num_steps>=25,000` and total runtime below 600 seconds, otherwise classify the fixed recipe as infeasible rather than tune it.
11. **Activation geometry audit**: record adaptive-coordinate radius and Euclidean perturbation norm on the first production ASAM pulse and compare them descriptively with the parent's fixed Euclidean radius 0.05. This is an attribution diagnostic, not a metric-driven pruning or tuning signal.

Run once after confirming physical GPU 0 is the approximately 98 GB NVIDIA H20:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1
```

Verify exit 0, no nonfinite/CUDA errors, 300-second charged budget, total runtime below 600 seconds, one evaluation per completed epoch, `num_params=2,748,890`, `num_steps>=24,000`, exact CutMix preservation, ASAM cadence and radius contract, complete summary, and `best_test_acc>=95.50%`. Remove `run.log` after analysis. Do not rerun or tune from accuracy.

## Effort

**Medium.** The training schedule and second pass already exist, but the scale-aware perturbation needs careful named-parameter handling, preallocated buffers, exact radius/restoration checks, and richer audit output.
