# Proposal: Front-Loaded HMix with the Validated SAM Tail

## Summary

Replace only the selected early CutMix operation in the accepted EXP-004 recipe with HMix at `r=0.5`. Preserve the parent's 0.5 application gate, `Beta(1,1)` lambda stream, 75% charged-time cutoff, hard-label clean batches, model, optimizer, drop path, and period-two SAM final quarter. HMix combines a smaller pure pasted box with whole-image interpolation outside that box, changing the form of input-gradient regularization without repeating EXP-003's inconclusive CutMix-probability tuning.

## Mechanism

CutMix gives the model a locally pasted region but leaves the rest of the original image unchanged. Mixup blends corresponding pixels globally but has no pure local replacement. HMix interpolates between these endpoints: it uses a reduced CutMix box containing only the paired image and linearly blends the two images everywhere outside the box. This exposes the early model simultaneously to local occlusion/composition and global interpolation while preserving a label coefficient equal to the actual source-image contribution.

The cited NeurIPS 2022 study reports HMix gains of 0.16-0.59 points over CutMix across CIFAR-100 backbones, including 75.68% versus 74.79% on WRN-28-2, with negligible extra computation (`experiments/005/papers/hmix.md`). The evidence is not direct for CIFAR-10 or this short schedule, but the mechanism is materially different from changing CutMix probability or drop path. It therefore has a plausible effect ceiling above the 0.10-point gate while retaining the successful front-loaded mixed/clean phase structure.

## Exact Mask, Image, and Label Math

Let `x_a, y_a` be the original batch, `x_b=x_a[permutation], y_b=y_a[permutation]`, sampled `lambda in [0,1]` be the nominal contribution of `x_a`, and fix `r=0.5`.

Define the nominal pure-paste area fraction and outside-box coefficient as

```text
q = r * (1 - lambda)
a = lambda / (1 - q)
```

Sample a shared rectangular mask `M` by the existing center procedure, with nominal side ratios based on `sqrt(q)`:

```text
cut_width  = int(width  * sqrt(q))
cut_height = int(height * sqrt(q))
```

`M=1` inside the clipped rectangle and `M=0` outside. Construct the image as

```text
x_hmix = (1 - M) * (a * x_a + (1 - a) * x_b) + M * x_b
```

Thus, the pasted rectangle is pure `x_b`; all pixels outside are globally mixed. Let the exact realized mask fraction after integer rounding and boundary clipping be

```text
q_hat = rectangle_area / (height * width)
```

The actual global contribution of `x_a` is then

```text
lambda_eff = (1 - q_hat) * a
```

and the loss must be

```text
loss = lambda_eff * CE(logits, y_a)
     + (1 - lambda_eff) * CE(logits, y_b)
```

Recomputing `lambda_eff` is necessary because a clipped/rounded box generally has `q_hat != q`. It keeps labels consistent with the pixels actually presented rather than pretending the nominal area was realized. Since clipping only reduces the box, `0 <= lambda_eff <= 1`; clamp only as a numerical guard after asserting the derivation in tests.

The formulation has the required endpoints. At `r=0`, `q=0`, the box is empty, `a=lambda`, and HMix is ordinary Mixup with `lambda_eff=lambda`. At `r=1` and without clipping, `q=1-lambda`, `a=1`, and HMix is ordinary CutMix with `lambda_eff=1-q_hat`. For the proposed `r=0.5`, it is a fixed hybrid, not a new tunable sweep.

## Concrete Change in `train.py`

Start from the current EXP-004 `train.py` and modify no other tracked file.

1. Add `HMIX_RATIO = 0.5` beside the existing mixing constants. Preserve `CUTMIX_PROB=0.5`, `CUTMIX_ALPHA=1.0`, `CUTMIX_END=0.75`, and `CUTMIX_SEED=42`; these continue to define the gate, lambda distribution, phase, and dedicated RNG streams even though selected batches now use HMix geometry.

2. Replace `cutmix_batch` with an `hmix_batch` helper accepting the same deterministic test overrides (`lam`, `center`, and `permutation`) plus `ratio=HMIX_RATIO`. Keep the exact sequence of random draws on selected batches: one CPU uniform lambda, two CPU center draws, and one CUDA permutation. The helper should:

   - compute `q`, `a`, clipped bounds, `q_hat`, and `lambda_eff` as above;
   - materialize `paired_inputs = inputs[permutation]` before mutating `inputs`, so all source pixels come from the original batch;
   - blend the full image in-place with `inputs.mul_(a).add_(paired_inputs, alpha=1-a)`;
   - overwrite the rectangle with the pure paired patch;
   - return `inputs, targets, paired_targets, lambda_eff, area`.

   Inputs do not require gradients, so in-place construction is valid. Materializing the paired batch is essential: using a view or reading from an already modified tensor would corrupt fixed points and permutation cycles.

3. At the existing `progress < CUTMIX_END` gate, leave the Bernoulli decision unchanged and call `hmix_batch` only when selected. Clean batches must remain unmodified and use hard-label cross-entropy. Rename audit counters/log text to `hmix_eligible_batches` and `hmix_applied_batches`, or explicitly log `mix_mode=hmix`; do not report HMix as if it were unchanged CutMix.

4. Use the returned `lambda_eff` in the existing single-forward two-term cross-entropy. Do not add label smoothing, another forward, per-sample masks, or a stochastic HMix/CutMix chooser; each would bundle an unreviewed mechanism or change cost and RNG semantics.

5. Preserve all SAM code exactly. HMix ends at `progress >= 0.75`, while SAM begins there, so the existing `apply_sam and targets_b is not None` assertion should remain and can be reworded to mention HMix. The first 75% remains one-pass training; every eligible even step in the final quarter retains the validated normal-plus-perturbed SAM procedure, CUDA RNG replay, BatchNorm suppression, exact restoration, and single optimizer update.

6. Add `hmix_ratio=0.5` and `mix_mode=hmix` to the startup config. Keep the required final summary keys unchanged. The final audit line should report applied/eligible ratio, which should remain near the parent's 0.4962 because the gate stream is unchanged.

## Schedule, RNG, and Budget Preservation

The proposal changes mask construction only after the parent's existing early gate succeeds:

- `0 <= progress < 0.75`: the same 0.5 gate chooses HMix versus an unchanged clean batch.
- `progress >= 0.75`: no HMix or CutMix; the same clean-tail periodic SAM schedule runs on every even one-based step.

All phases remain tied to `total_training_time / TIME_BUDGET_S`, so small throughput changes cannot shift the 75% transition in wall-clock terms. The fixed `TIME_BUDGET_S=300` imported from read-only `prepare.py` remains unchanged, and validation remains the frozen `Eval.evaluate` at most once per epoch.

No new stochastic decision is introduced. Keep the dedicated seed-42 CPU generator for the gate/lambda/center and CUDA generator for the permutation. HMix consumes the same number and ordering of draws as parent CutMix for every selected batch; it does not consume the global RNG used by data augmentation and drop path. The SAM tail remains non-overlapping and retains its existing CUDA RNG replay.

The full paired-batch gather and one multiply-add over `256*3*32*32` values occur inside the existing `t0` through CUDA synchronization interval, so their cost is charged. This is less than one million scalar pixels and is negligible relative to the WRN forward/backward; expected early one-pass overhead is low, with total optimizer exposure likely near EXP-004's 25,560 steps and peak VRAM increasing by only a few MiB for `paired_inputs`. A material drop below the parent exposure should be treated as a failed cost assumption rather than hidden by the fixed timer.

## Expected Effect and Testable Hypothesis

The accepted parent is EXP-004 at 95.40%, so the necessary threshold is 95.50%. The hypothesis is:

> Replacing selected early CutMix masks with fixed-`r=0.5` HMix, while preserving the 0.5/75% schedule and validated SAM tail, will improve early input-gradient regularization enough to produce `best_test_acc >= 95.50%` in one fixed-seed GPU-0 run without reducing optimizer exposure below 24,000 steps.

The plausible gain is approximately 0.10-0.30 points. This is deliberately below the larger CIFAR-100 paper gains because CIFAR-10 is easier, the current stack is already at 95.40%, and HMix is active on only about half of the first 75% of batches. Secondary positive signals would be final accuracy close to best and final test loss no worse than EXP-004's 0.1654, but neither substitutes for the primary accuracy threshold.

## Strongest Risks

- **Saturation and measurement noise**: EXP-003's selected sweep winners fell by 0.14-0.29 points on confirmation, demonstrating that apparent gains near this operating point can exceed the 0.10 threshold. HMix must be preregistered as one mechanism run, not selected from `r`, alpha, or probability trials. Do not retry or change `r` after seeing test accuracy. A marginal result below 95.50% is no improvement.
- **Evidence transfer**: the paper's reported gains are on CIFAR-100 and different backbones, not CIFAR-10 under a 300-second time budget. HMix may add little after successful CutMix, or global blending may remove useful local detail and overregularize the already regularized early phase.
- **Interaction with the SAM tail**: HMix and SAM do not overlap in time, but HMix changes the solution entering SAM. The clean tail may wash out the early representation benefit, or periodic SAM may amplify it; there is no direct paper evidence for this composition.
- **Mask/label mismatch**: using nominal lambda after clipping, using the CutMix adjusted lambda `1-q_hat`, or failing to preserve the paired source before in-place blending would train against incorrect targets. Exact contribution tests are mandatory.
- **Throughput**: unlike CutMix's patch-only clone, HMix gathers and blends a full batch on each selected step. The operation is small, but if it materially lowers the already reduced SAM-parent exposure, its regularization gain may be offset by fewer optimizer steps.

## Verification Plan

Before the full run:

1. **Closed-form mask smoke**: use tiny images whose pixels encode source identity, fixed lambda/center/permutation, and an unclipped rectangle. Verify every inside pixel is exactly from `x_b`, every outside pixel equals `a*x_a+(1-a)*x_b`, and returned `lambda_eff` equals `(1-q_hat)*a`.
2. **Clipping/rounding smoke**: force a corner-centered box and verify the returned label weight matches the mean source contribution measured directly from the synthetic output, not nominal lambda or `1-q_hat`.
3. **Endpoint smoke**: verify `r=0` produces Mixup and `r=1` with an unclipped rectangle produces the current CutMix result and label weight.
4. **Aliasing/orientation smoke**: use a nontrivial permutation including cycles and fixed points; verify all paired pixels and `targets_b` come from the pristine source batch and the loss weights label A by `lambda_eff`.
5. **Bounds and degenerate cases**: test lambda 0/1, zero-area boxes, all edges, and assert finite `a`/`lambda_eff` in `[0,1]`.
6. **RNG parity**: from identical dedicated generator states, compare the sequence and final states for one parent CutMix selection and one HMix selection. Gate, lambda, center, and permutation draws must match; global CPU/CUDA RNG states must remain untouched by HMix construction.
7. **GPU-path smoke**: run one BF16/channels-last WRN forward/backward with HMix and one late scheduled SAM step. Verify finite loss/gradients, no HMix/SAM overlap, exact SAM restoration, one BatchNorm update, and one optimizer update.
8. **Static checks**: compile `train.py`, run formatting/lint checks, confirm only `train.py` would differ from the EXP-004 commit, and confirm startup logs contain the complete HMix and unchanged SAM recipe.

Run exactly once from the repository root after confirming physical GPU 0 is the approximately 98 GB NVIDIA H20:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1
```

After completion, verify exit code 0; no traceback, NaN/Inf, CUDA, or memory errors; `training_seconds` approximately 300; `total_seconds < 600`; one evaluation for every completed epoch; unchanged `num_params=2,748,890`; `num_steps >= 24,000`; HMix applied/eligible ratio near 0.5; SAM applied/eligible ratio exactly consistent with every second eligible step; first SAM progress at approximately 0.75; full required summary; and `best_test_acc >= 95.50%`. Remove `run.log` after analysis. No seed reroll, hyperparameter selection, evaluator change, or repeat based on the result is permitted.

## Effort

**Low to medium.** The production change is a localized replacement of the current mixing helper and audit names, but the exact source-contribution, clipping, aliasing, RNG-parity, and integration smokes are necessary to make the comparison trustworthy.
