# Proposal: Bounded Soft-Target-Aware Poly-1 Loss

## Summary

Add Poly-1 to EXP-011 with a preregistered coefficient derived from a maximum per-target gradient-inflation budget, not from an accuracy sweep. Use one mathematically coherent loss for hard labels and area-corrected CutMix soft targets, and use it in both SAM passes. Preserve the complete WRN/CutMix/SAM/charged-time-EMA package, independent-image stream, optimizer, schedules, evaluation semantics, GPU-0 protocol, and 300-second charged budget.

Set `POLY1_EPSILON=0.25`, which bounds every hard-label or CutMix-constituent logit-gradient multiplier to at most 1.25. This is intentionally more conservative than blindly copying a task-dependent published coefficient into an already tuned high-accuracy stack.

## Goal and Limiter Match

EXP-011 is the global best at 95.61%, with 25,798 steps, 160 balanced EMA samples, negligible averaging overhead, and a final-16 EMA mean of 95.493 (`experiments/011/04-analysis.md`). EXP-012's complementary Cutout regressed to 95.52 and missed its realized-dose floor, so another spatial regularizer is poorly motivated (`experiments/012/04-analysis.md`). The current limiter is a stable decision-boundary/generalization lift above the EMA plateau, not memory or raw throughput (`02-system-understanding.md`).

Poly-1 changes the classification gradient on every example while adding no model pass, data transport, or persistent model state. It therefore targets the loss geometry directly and preserves every validated mechanism dose. The PolyLoss sources report image-classification improvements but emphasize task-dependent coefficients; the goal knowledge specifically recommends auditing effective gradient inflation and defining target probability coherently for CutMix (`knowledge/papers/polyloss.md`; `experiments/013/papers/poly-loss.md`).

## Exact Hard-Target Loss and Gradient

Let logits `z` produce `p=softmax(z)` and let `y` be a hard class. Define

```text
L_hard(z,y) = -log(p_y) + epsilon * (1 - p_y)
epsilon = 0.25
```

For softmax cross-entropy,

```text
gradient_z CE = p - one_hot(y)
gradient_z [epsilon*(1-p_y)] = epsilon*p_y*(p-one_hot(y))
```

therefore

```text
gradient_z L_hard = (1 + epsilon*p_y) * (p-one_hot(y))
```

Poly-1 does not rotate a hard example's CE logit gradient; it rescales that example by

```text
m_y = 1 + 0.25*p_y, so 1.0 <= m_y <= 1.25.
```

At an initially uniform CIFAR-10 prediction, `p_y=0.1` and the multiplier is only 1.025. At `p_y=0.5` it is 1.125; at `p_y=0.9` it is 1.225. The fixed bound therefore leaves uncertain early high-LR examples nearly parent-like while increasing the relative influence of residual confident-example errors in the low-LR tail.

## Exact CutMix Soft-Target Loss and Gradient

For original/paired classes `a,b` and the existing clipped-area original-label coefficient `lambda`, define

```text
q = lambda*one_hot(a) + (1-lambda)*one_hot(b)
p_t = q dot p = lambda*p_a + (1-lambda)*p_b

L_soft = CE(q,p) + epsilon*(1-p_t)
       = lambda*[-log(p_a) + epsilon*(1-p_a)]
         + (1-lambda)*[-log(p_b) + epsilon*(1-p_b)]
```

This uses the actual area-corrected lambda returned by `cutmix_batch`, not the sampled pre-clipping lambda. It creates no dense target tensor.

The exact logit gradient is

```text
gradient_z L_soft
  = p - q + epsilon * p elementwise-multiplied-by (p_t - q)
```

and equivalently

```text
= lambda*(1+epsilon*p_a)*(p-one_hot(a))
  + (1-lambda)*(1+epsilon*p_b)*(p-one_hot(b)).
```

Thus each target constituent's hard-label CE direction is inflated by a factor in `[1,1.25]` before the same area weighting. The norm of the final summed soft gradient is not guaranteed to be within 1.25x of soft CE because its two constituent vectors can cancel; report that combined ratio rather than treating it as a safety bound. At lambda 0/1, zero-area boxes, fixed-point permutations, or same-class pairs, the formula reduces exactly to the corresponding hard-target result.

## Accuracy-Blind Coefficient Derivation

Preregister a maximum constituent multiplier `POLY1_MAX_INFLATION=1.25`. Since `p_k<=1`, the hard/constituent multiplier satisfies

```text
max(1 + epsilon*p_k) = 1 + epsilon.
```

Solving `1+epsilon=1.25` gives the single coefficient

```text
POLY1_EPSILON = POLY1_MAX_INFLATION - 1 = 0.25.
```

The 25% ceiling is a conservative optimizer-compatibility choice: it avoids the 2x worst-case multiplier from unit Poly-1, leaves the parent's peak-LR early regime almost unchanged at low confidence, and does not require labels, test accuracy, or a preliminary metric run. It is a fixed engineering bound, not a claim that 1.25 is optimal.

Do not sweep epsilon, normalize by observed batch confidence, anneal the coefficient, change LR, or retry a second bound after metrics. A miss rejects only this bounded 0.25 package; the paper's task-dependent family remains broader.

## Concrete `train.py` Change

Add:

```python
POLY1_MAX_INFLATION = 1.25
POLY1_EPSILON = POLY1_MAX_INFLATION - 1.0
```

and assert both exact values at startup. Implement one helper returning per-example loss plus detached audit quantities:

```python
def poly1_loss(logits, targets_a, targets_b=None, lam=1.0):
    probabilities = F.softmax(logits.float(), dim=1)
    p_a = probabilities.gather(1, targets_a[:, None]).squeeze(1)
    ce_a = F.cross_entropy(logits, targets_a, reduction="none")

    if targets_b is None:
        ce = ce_a
        p_target = p_a
        multiplier_a = 1.0 + POLY1_EPSILON * p_a
        multiplier_b = None
    else:
        p_b = probabilities.gather(1, targets_b[:, None]).squeeze(1)
        ce_b = F.cross_entropy(logits, targets_b, reduction="none")
        ce = lam * ce_a + (1.0 - lam) * ce_b
        p_target = lam * p_a + (1.0 - lam) * p_b
        multiplier_a = 1.0 + POLY1_EPSILON * p_a
        multiplier_b = 1.0 + POLY1_EPSILON * p_b

    poly = POLY1_EPSILON * (1.0 - p_target)
    return (ce + poly).mean(), ce.detach(), poly.detach(), p_target.detach(), multiplier_a.detach(), multiplier_b
```

Use explicit FP32 softmax for stable target probabilities under BF16 autocast. Keep `F.cross_entropy` as the CE implementation so the parent component remains autocast-aware. Do not materialize `q`, add label smoothing, or modify the frozen evaluation loss.

Replace the current primary hard/CutMix branch with one call using `targets_a`, optional `targets_b`, and `adjusted_lam`. The model forward, CutMix image mutation and labels, backward, optimizer, counters, and smooth-loss display remain otherwise unchanged; the displayed training loss is now Poly-1 and is not directly comparable with EXP-011's CE training loss.

## SAM Two-Pass Semantics

SAM begins at progress 0.75 after CutMix ends, so production SAM targets are hard. Apply the same `poly1_loss(outputs, targets)` in both passes:

1. The normal pass backpropagates bounded Poly-1; `sam_perturb` uses its gradient to define the rho-0.05 direction.
2. Clear first gradients, restore the saved CUDA RNG state, disable second-pass BN tracking, and compute bounded Poly-1 again on the perturbed weights with the same inputs/targets.
3. Backpropagate perturbed Poly-1, restore exact base parameters and BN flags, and make the sole Nesterov update from that gradient.

Using CE in either pass would optimize a hybrid objective and invalidate the proposal. Keep SAM start, period, rho, global norm, RNG replay, one BN update, one optimizer update, and CutMix non-overlap assertion exact.

SAM normalizes the first gradient globally before applying rho, so Poly-1 mostly changes the relative example/tensor composition of the perturbation rather than its Euclidean norm. The second-pass optimizer gradient is not globally normalized and retains the bounded hard-example multipliers.

## EMA and Evaluation Preservation

Keep EXP-011's full-state `ChargedTimeEMA` bit-for-logic unchanged: progress start 0.75, cadence 31, 18.75-second half-life, post-optimizer/post-SAM-restore sampling, ordinary/SAM balance, floating/integer state treatment, and exception-safe evaluation swap/restore. The new loss changes the online trajectory but not EMA eligibility or decay.

The frozen evaluator must continue to compute ordinary hard-label CE and top-1 accuracy once per epoch, live before EMA activation and EMA afterward. Do not evaluate Poly-1 on test data. Final test loss remains comparable with the parent's 0.1552.

## Overhead and Feasibility

Each primary call adds one explicit FP32 softmax over only `256x10` logits, gathers, and vector arithmetic; each scheduled SAM second pass repeats that small work. CutMix already computes two CE terms. Temporary probability/audit tensors are tens of KiB, with no model-sized persistent state beyond a few device accumulators.

All work occurs before the existing synchronization and is charged. The helper consumes no RNG, so data order, crop/flip, drop path, CutMix generators, SAM replay, and cadence-31 EMA sampling decisions remain structurally identical. Expected weighted latency overhead is below 1% and peak VRAM should remain close to EXP-011's 1,222.4 MiB.

Use a same-harness, five-round production-weighted GPU-0 preflight covering ordinary, CutMix, SAM, and EMA cadence paths. Proceed only if median candidate/parent latency is at most 1.01, paired-ratio dispersion is stable, projected steps are at least 25,600, projected EMA samples at least 158, and projected total runtime below 570 seconds. Because EXP-012 showed preflight can overpredict realized dose, the metric run separately requires at least 25,500 actual steps and 158 EMA updates; a miss is a dose failure and must not trigger a retry.

## Audit Contract

Startup config must print `loss=poly1`, `poly1_max_inflation=1.25`, `poly1_epsilon=0.25`, `soft_target=area_weighted`, and unchanged CutMix/SAM/EMA constants.

Accumulate diagnostics on-device and synchronize only through the existing step boundary. Report separately for primary hard, primary CutMix, and SAM-second calls:

- calls and examples;
- mean CE and Poly components and mean Poly/CE ratio;
- min/mean/max `p_target`;
- min/mean/max constituent multipliers, all within `[1,1.25]`;
- for CutMix, min/mean/max adjusted lambda and both `m_a`, `m_b` distributions;
- analytic per-example `||g_poly_logits||/||g_ce_logits||` distribution for hard and soft targets, with hard ratios equal to the multiplier and soft ratios treated as descriptive;
- nonfinite/out-of-range counts.

Reconcile:

```text
total loss calls = num_steps + sam_applied_batches
primary CutMix calls = cutmix_applied_batches
SAM-second calls = sam_applied_batches
primary hard calls + primary CutMix calls = num_steps
```

Also retain exact CutMix exposure, SAM first step/cadence, EMA update and 80/80-style parity accounting, swap/restore/RNG/state failure counts, and final 16 EMA accuracies with mean/range/final. Copy all durable evidence before transient log deletion.

## Correctness Smokes

1. Compare hard loss and autograd gradients against FP64 manual Poly-1; verify exact multiplier `(1+0.25*p_y)`.
2. Compare CutMix loss/gradient against both a dense-q implementation and the area-weighted sum of hard Poly-1 terms over lambda 0, 0.5, 1, clipped values, fixed points, and same-class pairs.
3. Verify the analytic soft gradient `p-q+epsilon*p*(p_t-q)` and constituent decomposition against autograd.
4. Test the inflation criterion at uniform, low-, medium-, and high-confidence logits; require every constituent multiplier in `[1,1.25]` without imposing that bound on the summed soft-gradient ratio.
5. Run CPU FP32 and H20 BF16/channels-last forward/backward with finite FP32 probabilities/loss/gradients and no range failure.
6. Instrument both SAM passes to prove identical loss/coefficient use, replayed drop-path masks, one BN update, exact parameter restore, and one momentum update.
7. Prove the loss helper consumes no global/CutMix RNG and leaves all non-loss parent decisions unchanged.
8. Trigger ordinary and SAM EMA samples plus an EMA evaluation; verify update cadence, state coverage, exact restore, optimizer identity, and metric routing remain parent-identical.
9. Reconcile all audit counters/components against a fixed scripted trace.
10. Compile/lint `train.py`, confirm only it differs from EXP-011, and pass the fixed GPU-0 latency/dose preflight.

## Risks and Attribution

- The 0.25 coefficient is stability-derived, not accuracy-optimal; it may be too weak to lift the 95.49 plateau or still be wrong for CIFAR-10.
- For hard labels Poly-1 only reweights examples, favoring higher-confidence examples relatively more. That may improve boundary refinement or may spend capacity on already easy examples.
- For CutMix the summed soft gradient is not a scalar rescale of soft CE and can shift the optimum away from the exact area distribution; constituent bounds do not bound the combined ratio under cancellation.
- Both SAM direction and second-pass update change. A result tests bounded Poly-1 plus existing SAM, not a standalone CE replacement in ordinary SGD.
- EMA may smooth a genuine gain or a regression; the formal maximum still sits above a noisy stable tail.
- The expected improvement is near protocol resolution, so best accuracy alone is insufficient context.

Everything except the loss is held fixed, but attribution remains package-level across hard Poly-1, soft-target extension, SAM composition, and EMA evaluation. Do not claim the coefficient or soft extension is individually causal.

## Hypothesis and Falsification

The formal parent-relative threshold is `best_test_acc >= 95.71%` over EXP-011's 95.61%. A mechanism-sized result is `best_test_acc >= 95.86%` (+0.25) with final-16 EMA mean at least 95.70%, at least 25,500 steps, and at least 158 EMA updates.

Bounded Poly-1 **plausibly but not confidently** clears 95.71. It touches every training gradient with negligible compute and has image-classification evidence, but `epsilon=0.25` is conservative and the soft-target/SAM composition is unvalidated. A realistic expected gain is about 0.05-0.20 points; a stable +0.25 is an upside case rather than the median expectation.

Falsification rules:

- below 95.71: no formal improvement;
- 95.71-95.85: formal improvement below the mechanism-sized target;
- best at least 95.86 but final-16 mean below 95.70: selected-maximum gain without stable plateau lift;
- fewer than 25,500 steps or 158 EMA updates: realized-dose failure even if accuracy passes;
- any formula, multiplier-bound, SAM-pass, RNG, or EMA integrity failure: invalid result.

Do not tune epsilon, relax the inflation ceiling, change LR, disable Poly-1 on CutMix/SAM, rerun the seed, or select a different evaluation path after metrics.

## Execution

After smokes and preflight, confirm physical GPU 0 is the approximately 97,871 MiB H20 and launch once:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1
```

Verify exit 0, 300-second charged budget, total below 600 seconds, one evaluation per epoch, `num_params=2,748,890`, complete loss/dose/state audits, durable final-16 transcription, and `best_test_acc>=95.71%`. Remove `run.log` only after analysis and independent result review.

## Effort

**Low implementation, medium verification.** The loss helper is small; soft-gradient algebra, bounded-inflation audits, both SAM passes, EMA integrity, and realized-dose verification require rigor.
