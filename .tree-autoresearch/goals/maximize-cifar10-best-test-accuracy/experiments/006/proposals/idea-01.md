# Proposal: Adaptive Clean-Tail Sharpness Minimization

## Summary

Replace only EXP-004's plain-SAM perturbation geometry with element-wise p=2 Adaptive Sharpness-Aware Minimization (ASAM). Preserve the validated sharpness schedule exactly: the first 75% of charged time remains the unchanged WRN/CutMix path, and the clean final quarter uses a second sharpness pass on every even upcoming one-based step. Preserve the same batch and hard-label loss on both passes, CUDA RNG replay, second-pass BatchNorm-stat suppression, exact parameter restoration, and one Nesterov-SGD update.

Fixed settings:

- `ASAM_RHO = 0.5`
- `ASAM_ETA = 0.01`
- element-wise p=2 adaptation
- `SAM_START = 0.75`
- `SAM_PERIOD = 2`
- no adaptive scaling for parameters named `*.bias`

This is a replacement, not ASAM added on top of SAM. `rho=0.5` is the paper's common CIFAR setting transferred without tuning; it is not comparable to plain SAM's Euclidean `rho=0.05` and is not claimed optimal for a late-only periodic schedule.

## Motivation and Evidence

EXP-004 established that clean-tail period-two SAM improves this lineage from 95.23% to 95.40%, with final accuracy equal to best accuracy and final loss improving from 0.2044 to 0.1654. It retained 25,560 steps, applied 2,449 sharpness pulses, and preserved parent RNG and BatchNorm semantics. EXP-005 then failed because its overlap sampler halved new-image introduction; EXP-006 returns to EXP-004's independent-image stream.

ASAM replaces SAM's scale-sensitive spherical neighborhood with a parameter-adaptive neighborhood. The ICML paper reports ASAM above SAM on CIFAR-10 across ResNet, ResNeXt, WRN, DenseNet, and PyramidNet. Reported ASAM-over-SAM gains include 0.20 points on WRN-28-2 and 0.30 on WRN-28-10, both above this goal's 0.10-point gate. The evidence uses full-run ASAM, so transfer to 2,449 late periodic pulses remains uncertain.

Sources:

- `experiments/006/papers/asam.md`
- `experiments/004/04-analysis.md`
- `experiments/005/04-analysis.md`
- `experiments/006/00-navigate.md`
- https://proceedings.mlr.press/v139/kwon21b.html

## Exact Element-Wise p=2 Formula

Let `g_i` be the first-pass data-loss gradient of trainable parameter element `w_i`. Define the element-wise scale

```text
s_i = abs(w_i) + eta    for non-bias trainable parameters
s_i = 1                 for parameters whose name ends with ".bias"
```

Then compute globally across every trainable parameter:

```text
u_i       = s_i * g_i
D         = sqrt(sum_i u_i^2)
epsilon_i = rho * s_i * u_i / (D + eps)
          = rho * s_i^2 * g_i / (D + eps)
w_i_adv   = w_i + epsilon_i
```

This is not `rho * scaled_grad / ||scaled_grad||`; the second multiplication by `s_i` is essential. The invariant to verify is the adaptive-coordinate radius:

```text
||epsilon / s||_2 = rho * D / (D + eps) ~= rho
```

The ordinary Euclidean norm `||epsilon||_2` is neither fixed at 0.5 nor a valid correctness check.

## Parameter Inclusion Policy

Every trainable parameter remains in the global denominator and receives a perturbation. There are no perturbation exclusions.

- Convolution and linear weights: adaptive scale `abs(w)+0.01`.
- BatchNorm affine `weight` (`gamma`): adaptive scale `abs(w)+0.01`.
- Linear and BatchNorm `bias`: unadapted unit scale, but still perturbed.
- BatchNorm `running_mean`, `running_var`, and `num_batches_tracked`: buffers, never included or perturbed.

Classify by stable parameter name (`name.endswith(".bias")`), not tensor rank. This makes the paper's preferred no-bias-adaptation policy explicit while retaining BN gamma adaptation; using `ndim == 1` would accidentally disable adaptation for BN weights as well.

## Integration with the Existing Training Step

Ordinary and CutMix batches remain bit-for-bit on the EXP-004 path. On a scheduled clean ASAM batch:

1. Save the CUDA RNG state immediately before the unperturbed primary forward.
2. Run the existing BF16-autocast hard-label loss and first backward. Model parameters and accumulated gradients remain FP32; no gradient scaler is used.
3. Outside autocast and under `torch.no_grad()`, compute scales, `u`, the global FP32 norm, and the ASAM perturbation. Require all gradients, the denominator, scales, and perturbations to be finite and `D > 0`.
4. Copy each unperturbed parameter into the existing preallocated exact snapshot, then add `epsilon` to the live parameter.
5. Clear first gradients, replay the saved CUDA RNG state, disable BatchNorm running-stat tracking, and execute the same separately autocast second forward/backward at perturbed weights.
6. In the existing `finally` path, restore every BatchNorm tracking flag and copy exact parameter snapshots back before the optimizer update.
7. Call the unchanged Nesterov-SGD optimizer exactly once using the second-pass gradients at restored weights.

The first gradient defines only the perturbation. PyTorch SGD applies `weight_decay=1e-4` during the sole optimizer step, so weight decay must not be inserted into `g`, `u`, or `epsilon`; it must not run at perturbed weights or after the first backward. This preserves EXP-004's optimizer semantics: momentum and weight decay each update exactly once from the second gradient and restored parameters.

## Foreach and Snapshot Implementation

Keep `named_parameters()` order fixed and preallocate three same-shaped FP32 lists during startup:

- exact unperturbed snapshots;
- element-wise scale tensors;
- work tensors reused for `u` and then `epsilon`.

On each ASAM pulse:

1. `foreach_copy_` snapshots from parameters.
2. For non-bias entries, copy parameters into scale tensors, then `abs_()` and add `ASAM_ETA`; fill bias scales with `1.0`.
3. `foreach_copy_` work tensors from first gradients and `foreach_mul_` by scales, yielding `u`.
4. Compute `D` as the FP32 vector norm of per-work-tensor L2 norms.
5. Multiply work tensors by scales a second time, then multiply by `ASAM_RHO / (D + eps)`, yielding `epsilon`.
6. `foreach_add_` parameters with work tensors; after the second backward, `foreach_copy_` parameters from snapshots.

Do not modify the original `.grad` tensors in place. Reusing preallocated work avoids per-pulse model-sized allocations and makes the formula auditable. Current SAM already stores one approximately 11 MiB snapshot; scales and work add about 22 MiB, negligible against the 98 GB H20. Foreach lists must have matching device, dtype, layout, and order, and exact restoration must occur on every exception path.

## Compute and Budget

ASAM uses the same number of model forwards/backwards and optimizer updates as EXP-004. It adds several element-wise foreach operations over 2.75M parameters on only about 2,450 scheduled pulses. These operations are small relative to each extra WRN forward/backward, but they are fully charged between the existing `t0` and CUDA synchronization.

Expect approximately 24,800-25,560 total steps versus EXP-004's 25,560 and require at least 24,500. The preallocated buffers are startup work excluded under the existing protocol; their per-step fills/multiplies and norm synchronization are charged. Validation remains once per epoch and the outer runtime remains capped at 600 seconds.

## Implementation Scope

Modify only `train.py`:

- Replace `SAM_RHO=0.05` and `sam_perturb` with fixed ASAM constants, named-parameter classification, buffers, and the exact formula above.
- Preserve `SAM_START=0.75`, `SAM_PERIOD=2`, scheduling, counters, first-step/progress logging, RNG replay, BatchNorm handling, and exact restoration.
- Extend configuration/mechanism output with `sharpness_method=asam`, `asam_rho=0.5`, `asam_eta=0.01`, adaptive and unit-scale parameter counts, first adaptive-coordinate norm, and first Euclidean perturbation norm.

Do not change architecture, parameter count, data loader/transforms, CutMix constants/generators, optimizer/LR/drop-path, seed, timing boundary, evaluator, validation cadence, metric accumulation, or required summary keys. No dependency is added.

## Discriminating Smokes

1. **Reference formula:** On tiny hand-specified weight, BN-weight, and bias tensors with fixed gradients, compare foreach output element-for-element with a simple loop implementing `rho*s^2*g / ||s*g||`. Require adaptive-coordinate norm approximately 0.5.
2. **Second-scale-factor trap:** Compare against the incorrect `rho*s*g / ||s*g||` formula and require a deliberate mismatch whenever adaptive scales differ.
3. **Bias/BN policy:** Change only bias magnitudes and require unchanged unit bias scales; change BN gamma magnitudes and require changed adaptive scales. Assert all biases are perturbed, all BN weights are adapted, and no BN buffer appears in the parameter list.
4. **ASAM versus SAM:** Give two equal-gradient weights different magnitudes. Plain SAM would preserve their gradient ratio; ASAM must produce the squared-scale perturbation ratio. Require the Euclidean perturbation norm not to be forced to 0.5.
5. **Foreach parity and restore:** Compare foreach and loop perturbations, inject a second-pass exception, and require bitwise restoration of every parameter, unchanged optimizer state, and restored BN flags.
6. **Weight-decay semantics:** On a tiny model, analytically verify one Nesterov-SGD update uses the restored parameter, second gradient, and weight decay exactly once; first gradients must not enter momentum or decay.
7. **Full WRN BF16 pulse:** On GPU 0, require finite first/second losses and gradients, finite `D`, adaptive norm approximately 0.5, distinct perturbed loss, exact restore, one BatchNorm-buffer update, CUDA RNG parity with a one-forward reference, and one optimizer update.
8. **Cadence/isolation:** Require no pulse below progress 0.75, every even upcoming step above it, no CutMix overlap, and unchanged CutMix/SAM exposure arithmetic.

## Full-Run Verification

Run exactly once with `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. Require physical GPU 0 to be the 97,871 MiB NVIDIA H20, exit 0, 299.5-301.0 charged seconds, total time below 600 seconds, at least 24,500 steps, all ten summary keys exactly once, unchanged 2,748,890 parameters, one evaluation per epoch, and no NaN/Inf, traceback, CUDA error, or timeout.

Mechanism verification must require ASAM first progress at approximately 0.75, applied/eligible ratio exactly period-two within parity arithmetic, no CutMix after the transition, finite positive first Euclidean perturbation norm, first adaptive-coordinate norm within tolerance of 0.5, expected adaptive/unit parameter counts, exact snapshot restore, and unchanged RNG/BatchNorm/optimizer invariants. Success requires `best_test_acc >= 95.50%` against parent 95.40%.

## Risks

- The paper validates full-run ASAM, not a late periodic dose; 2,450 pulses may be insufficient to realize its advantage over SAM.
- `rho=0.5` was selected from a broad literature grid and may be miscalibrated for this low-LR tail. It is fixed before execution and must not be tuned from test accuracy.
- Squared weight scaling can concentrate perturbation in large-magnitude parameters and under-perturb near-zero weights despite `eta=0.01`; nonfinite and scale-distribution smokes guard correctness, not efficacy.
- Adapting BN gamma may interact with normalization geometry, while leaving BN bias unadapted breaks full element-wise symmetry. This follows the declared policy and must not be changed post hoc.
- Extra parameterwise operations may reduce steps enough to erase an accuracy benefit; the 24,500-step floor distinguishes an unexpectedly expensive implementation.
- EXP-003 observed 0.14-0.29-point selected-run variability. A marginal gain must still clear the fixed 0.10-point gate in the single preregistered run without seed rerolling.
- Replacing a validated SAM geometry can regress the global best; failure does not invalidate the EXP-004 cadence or ordinary SAM result.

## Testable Hypothesis

Element-wise p=2 ASAM with `rho=0.5`, `eta=0.01`, adaptive non-bias weights, and the unchanged final-quarter period-two cadence will retain at least 24,500 optimizer steps and achieve `best_test_acc >= 95.50%` versus EXP-004 at 95.40%. The predicted improvement comes from a scale-aware neighborhood providing a more meaningful late flatness signal than spherical SAM without adding model passes. A result below 95.50%, an adaptive-norm/formula mismatch, or any scope/timing failure falsifies this proposal without a rerun or post-hoc geometry change.
