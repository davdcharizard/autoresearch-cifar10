# Proposal: Soft-Target-Aware Poly-1 Loss on the WRN/CutMix/SAM/EMA Stack

## Summary

Replace every online training cross-entropy in EXP-011 with Poly-1 using the paper's simple fixed coefficient `epsilon_1=1.0`. Define the loss for a general target distribution, so hard labels and area-corrected CutMix targets use one coherent objective. Apply the same loss to the normal SAM pass and perturbed SAM pass. Preserve the WRN, independent-image DataLoader, CutMix geometry/RNG/dose, SAM cadence/radius/replay, charged-time EMA, optimizer, LR/drop path, frozen evaluator, one evaluation per epoch, and 300-second charged budget.

This is a loss-only package with no new dependency, model pass, state, or random decision. It targets the current stable generalization plateau by changing probability-gradient geometry rather than replacing a validated data or optimizer mechanism.

## Exact Loss

Let logits `z` produce probabilities `p=softmax(z)`, and let `q` be any normalized target distribution. Define soft-target Poly-1 as

```text
L_poly1(q, p) = sum_c q_c * [-log(p_c) + epsilon_1 * (1 - p_c)]
               = CE(q, p) + epsilon_1 * (1 - sum_c q_c p_c)
epsilon_1 = 1.0
```

The second equality follows because `sum_c q_c=1`. This is not a heuristic confidence penalty: it applies the paper's hard-label Poly-1 basis independently to every nonzero target component and then takes the target-distribution expectation.

For a hard target `y`, `q=one_hot(y)`:

```text
L_hard = CE(z, y) + (1 - p_y)
```

For CutMix labels `y_a`, `y_b` and the existing clipped-area original-label coefficient `lambda`:

```text
q = lambda * one_hot(y_a) + (1 - lambda) * one_hot(y_b)

L_cutmix = lambda * CE(z, y_a) + (1 - lambda) * CE(z, y_b)
           + 1 - [lambda * p(y_a) + (1 - lambda) * p(y_b)]
```

Equivalently, this is

```text
lambda * [CE(z,y_a) + (1-p(y_a))]
+ (1-lambda) * [CE(z,y_b) + (1-p(y_b))]
```

so target orientation and area weighting remain exact. At `lambda=1` or a zero-area CutMix box it reduces to hard-label Poly-1 on `y_a`; at `lambda=0` it reduces to hard-label Poly-1 on `y_b`. If a permutation maps an example to the same class, the duplicate terms collapse correctly.

For hard labels, the added term strengthens intermediate-confidence gradients while vanishing as `p_y` approaches one. In the cross-entropy expansion `-log p_y=sum_j (1-p_y)^j/j`, `epsilon_1=1` changes only the first-order coefficient from 1 to 2. It does not add label smoothing or change the target distribution.

## Coefficient Rationale

Set `POLY1_EPSILON=1.0` globally for the sole run. This is the canonical simple Poly-1 coefficient described by the paper and corresponds to a transparent doubling of the first polynomial coefficient (`experiments/012/papers/polyloss.md`). It is chosen before any metric observation and is applied identically to hard and soft targets.

Do not sweep `{-1, 0.5, 1, 2}`, scale epsilon by CutMix lambda, anneal it, or choose it from test loss. The paper explicitly says the optimum is task-dependent, so coefficient transfer is the main scientific risk. Using the simplest positive unit coefficient is more defensible than inventing a CIFAR/EMA-specific scalar. A valid miss rejects this fixed unit-Poly-1 package, not the entire polynomial-loss family.

Keep peak LR 0.2 and all optimizer settings unchanged. Rescaling LR to compensate for the added gradient would bundle a second intervention and obscure whether the paper-default loss improves this stack.

## Concrete `train.py` Implementation

Add `POLY1_EPSILON=1.0` near the other fixed hyperparameters and include it in the startup config. Introduce one helper that accepts logits, primary targets, optional paired targets, and the area-corrected coefficient:

```python
def poly1_loss(logits, targets_a, targets_b=None, lam=1.0):
    probabilities = F.softmax(logits.float(), dim=1)
    p_a = probabilities.gather(1, targets_a[:, None]).squeeze(1)

    if targets_b is None:
        ce = F.cross_entropy(logits, targets_a)
        target_probability = p_a
    else:
        p_b = probabilities.gather(1, targets_b[:, None]).squeeze(1)
        ce = lam * F.cross_entropy(logits, targets_a)
        ce += (1.0 - lam) * F.cross_entropy(logits, targets_b)
        target_probability = lam * p_a + (1.0 - lam) * p_b

    poly1 = POLY1_EPSILON * (1.0 - target_probability).mean()
    return ce + poly1, ce.detach(), poly1.detach(), target_probability.detach()
```

Use explicit FP32 softmax for stable probabilities under the surrounding BF16 autocast. Retain PyTorch's existing autocast-aware `F.cross_entropy` calls rather than reimplementing log-softmax, keeping the CE component as close as possible to the parent. `lam` is the current Python float `adjusted_lam`; do not construct a dense 256x10 target tensor.

Replace the primary loss branch with exactly one helper call:

```python
loss, ce_component, poly_component, p_target = poly1_loss(
    outputs,
    targets_a,
    targets_b,
    adjusted_lam,
)
```

No other primary-forward behavior changes. Early selected CutMix batches use the soft formula; early non-CutMix and all late batches use the hard formula.

## SAM Semantics

SAM starts at progress 0.75, exactly when CutMix ends, so production SAM batches are hard-target batches. Both SAM passes must nevertheless use the same unit Poly-1 helper:

1. The normal forward computes hard-label Poly-1 and backpropagates it. `sam_perturb` therefore constructs the fixed-rho adversarial direction from the Poly-1 gradient, not the old CE gradient.
2. After perturbation, gradient clearing, CUDA RNG replay, and BatchNorm tracking suppression, the perturbed forward computes hard-label Poly-1 again on the same inputs/targets.
3. Backpropagate that perturbed Poly-1 loss, restore exact parameters/BN flags, and perform the sole Nesterov update from the perturbed Poly-1 gradient.

Do not use Poly-1 only on the optimizer gradient while retaining CE for the perturbation, or vice versa. Such a hybrid would no longer optimize the stated robust objective. Preserve rho 0.05, period two, identical drop-path masks, one online BN update, and one optimizer update.

The helper supports soft targets for the early phase, but the existing `SAM and CutMix must not overlap` assertion remains unchanged and must continue to pass.

## EMA and Evaluation Preservation

Keep EXP-011's `ChargedTimeEMA` implementation and constants exactly: start 0.75, cadence 31, four tail half-lives, 18.75-second half-life, full floating/integer state handling, exact evaluation swap/restore, and live-before/EMA-after routing. Poly-1 changes the online trajectory sampled by EMA but not update eligibility, decay math, state inventory, or evaluation semantics.

The frozen `Eval.evaluate` must continue to report ordinary hard-label cross-entropy and top-1 accuracy. Do not evaluate Poly-1 loss on the test set or alter summary meaning. Final test loss remains directly comparable with EXP-011's 0.1552, while smoothed training loss now represents Poly-1 and is not numerically comparable with the parent's CE training trace.

## Mechanism and Evidence

PolyLoss views CE as a polynomial series in target error probability and reports image-classification gains from changing only its leading coefficient (`experiments/012/papers/polyloss.md`). The added unit term gives more weight to examples whose target-assigned probability remains low or intermediate, potentially improving class boundaries that the existing CE/CutMix/SAM/EMA stack still misclassifies. It costs one tiny ten-class softmax/gather and does not consume the sample, architecture, SAM-pass, or EMA exposure that prior failures disturbed.

This mechanism is complementary in principle:

- CutMix defines a spatially mixed target distribution; Poly-1 changes how probability assigned to that distribution is rewarded.
- SAM changes the parameter-space neighborhood used for the update; using Poly-1 in both passes changes the loss surface consistently.
- EMA averages the resulting online trajectory and can preserve a stable gain rather than only one live checkpoint.

The evidence transfer is uncertain. The paper packet does not establish CIFAR-10 gains on this exact WRN/CutMix/SAM stack, does not validate the soft-target extension, and warns that epsilon is task-dependent. The current formal best 95.61 is already a maximum above a 95.493 tail mean, so a small loss change can appear successful through checkpoint selection without lifting the stable plateau.

## Compute, Memory, and RNG Feasibility

Each primary forward adds one FP32 softmax over 2,560 logits, one or two gathers, and a few vector operations. Each scheduled SAM pulse adds the same work to its second pass. This is negligible beside WRN convolutions and adds no persistent model-sized state. Peak VRAM should remain near EXP-011's 1,222.4 MiB; temporary probability storage is about 10 KiB in FP32.

All helper arithmetic occurs before the existing synchronization and is charged. It consumes no random numbers, so global shuffle/crop/drop-path RNG, private CutMix generators, SAM replay, and EMA cadence remain structurally unchanged. Different gradients intentionally change weights and later BN states, but not RNG draw counts.

Run a production-weighted parent/candidate GPU-0 latency preflight including ordinary, CutMix, SAM two-pass, and cadence-31 EMA steps in their expected proportions. Proceed only if candidate/parent weighted median latency is at most 1.02, projected optimizer steps are at least 25,000, projected EMA updates are at least 150, and projected total runtime is below 570 seconds. Measure the parent in the same harness; do not use an absolute gate the parent could fail.

## Audit Contract

Startup config must print `loss=poly1`, `poly1_epsilon=1.0`, `poly1_soft_target=area_weighted_probability`, and unchanged CutMix/SAM/EMA constants.

Maintain no per-step host synchronization beyond the existing loss read. Accumulate detached GPU-side sums/counts and report at the end:

- primary hard calls/examples; primary CutMix calls/examples; SAM second-pass calls/examples;
- mean CE component, mean Poly-1 component, and their ratio for each of those three paths;
- min/mean/max area-corrected lambda on mixed calls;
- min/mean/max `p_target` and mean added term for hard, mixed, and SAM-second paths;
- total helper calls equals `num_steps + sam_applied_batches`;
- mixed helper calls equals `cutmix_applied_batches`;
- SAM-second helper calls equals `sam_applied_batches`;
- nonfinite logits/probability/loss/gradient counts all zero;
- unchanged CutMix exposure, SAM cadence/first progress, EMA ordinary/SAM sample balance, update/evaluation counts, exact restoration, RNG, and inventory failure counters.

Audit `p_target` within `[0,1]` up to numerical tolerance and Poly component within `[0,1]` for epsilon one. Record the final 16 EMA evaluation accuracies, their mean/range, final accuracy/loss, and best epoch so the 95.61 parent's selected maximum and 95.493 plateau are both compared durably before transient log deletion.

## Correctness Smokes

Before launch:

1. **Hard formula**: compare helper output and logits gradients with a direct FP64 `CE + 1-p_y` calculation on fixed logits/targets.
2. **Soft formula**: compare against a dense target tensor implementing `sum q[-log p + (1-p)]` for fixed labels/lambdas.
3. **Weighted equivalence**: prove CutMix loss equals lambda-weighted hard Poly-1 terms, including lambda 0, 0.5, 1, fixed points, same-class pairs, and clipped-area values from the existing helper.
4. **Autocast stability**: run CPU FP32 and H20 BF16/channels-last forward/backward; require finite FP32 probabilities, finite loss/gradients, and no out-of-range target probabilities.
5. **SAM two-pass parity**: instrument helper calls to prove identical epsilon/formula in normal and perturbed passes, identical replayed drop-path masks, one BN update, exact restore, and one momentum update.
6. **Parent mechanism parity**: verify CutMix pixels/lambda/RNG, non-loss model outputs, schedule decisions, SAM cadence, and EMA cadence/state operations match EXP-011 for a fixed decision trace.
7. **RNG isolation**: snapshot global CPU/CUDA and CutMix generators around helper calls and prove no state advances.
8. **Audit arithmetic**: simulate hard/mixed/SAM paths and reconcile all sums/counts, component bounds, and helper-call identities.
9. **EMA integration**: run enough synthetic steps to trigger ordinary and SAM EMA samples and one EMA evaluation; verify Poly-1 changes no update/swap/restore contract.
10. **Static/latency**: compile/lint `train.py`, confirm only it differs from EXP-011, validate config output, and pass the fixed parent-relative GPU preflight.

## Expected Effect, Threshold, and Falsification

The formal parent-relative threshold is **95.71%** over EXP-011's 95.61%. Because the parent's final 16 EMA checkpoints averaged only 95.493%, a meaningful result should also raise stable tail behavior. The preregistered mechanism target is `best_test_acc >= 95.86%` (+0.25 points) with final-16 EMA mean at least 95.70%, while retaining at least 25,000 steps and 150 EMA updates.

It is **plausible but not high-confidence** that unit Poly-1 clears 95.71. The intervention directly affects every online gradient, has paper-level image-classification support, and is effectively free, so a 0.10-0.25-point gain is credible. A stable +0.25-point gain is less certain because coefficient transfer and mixed-target/SAM composition are unvalidated. This candidate has a better effect ceiling than another averaging micro-tune but weaker local evidence than a matched CIFAR ablation.

Falsification rules:

- below 95.71: no formal improvement, regardless of test loss or Poly training loss;
- 95.71-95.85: formal improvement but below the mechanism-sized target;
- best >=95.86 with final-16 mean <95.70: selected-maximum improvement without the intended stable plateau lift;
- fewer than 25,000 steps, fewer than 150 EMA updates, or latency ratio above 1.02: feasibility/dose failure;
- any mismatch in hard/soft formula, SAM-pass use, RNG, evaluation, or EMA restore: invalid result.

Do not change epsilon, disable Poly-1 on CutMix or one SAM pass, rescale LR, rerun the seed, or choose a checkpoint/model path after observing metrics. A miss supports only that this fixed soft-target-aware unit-Poly-1 package failed on the EXP-011 lineage.

## Execution and Verification

After passing smokes and confirming physical GPU 0 is the approximately 97,871 MiB NVIDIA H20, launch once:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1
```

Verify exit 0, 300-second charged budget, total runtime below 600 seconds, no traceback/nonfinite/CUDA errors, one evaluation per epoch, `num_params=2,748,890`, complete helper/component audits, exact CutMix/SAM/EMA preservation, full durable metric transcription, and `best_test_acc>=95.71%`. Remove `run.log` only after analysis and independent result review.

## Effort

**Low implementation, medium verification.** The production change is one small loss helper and two call-site replacements; soft-target algebra, two-pass SAM consistency, audit accounting, and stable-tail interpretation require rigorous tests.
