# Proposal: Training-Only Direct-Path Auxiliary CE

## Claim and Scope

EXP036 established that `z_refined = z + 0.1 * pooled_head(z)` improves both
accuracy and loss, but it did not test whether the raw pooled vector `z` remains
independently class-useful while the nonlinear branch learns. Add a secondary
training-only classification objective on `fc(z)` while leaving `fc(z_refined)`
as the sole inference/evaluator output. This is a cheap deep-supervision test of
the accepted representation, not a claim that the direct path is currently
collapsed.

## Exact Semantics

Preserve the accepted graph, parameters, initialization, optimizer groups, RNG,
data, temporal gates, and evaluator path. Refactor `WideResNet.forward` only so
that, after obtaining the raw pooled feature `z`, it computes the accepted main
logits first:

```python
z_refined = z + POOLED_HEAD_SCALE * self.pooled_head(z)
main_logits = self.fc(z_refined)
if return_direct:
    return main_logits, self.fc(z)
return main_logits
```

The default must remain `return_direct=False`, so every evaluation call receives
exactly the accepted main tensor with no tuple, auxiliary loss, or direct-logit
computation. Training requests both tensors. Computing main logits first avoids
changing the accepted main GEMM merely to fuse the two classifier calls.

For either label `y`, define `C_m(y)=CE(main_logits,y)` and
`C_d(y)=CE(direct_logits,y)`. Reuse the accepted structural scale `s =
POOLED_HEAD_SCALE = 0.1` as a loss fraction, with no new coefficient:

```python
L(y) = (1 - s) * C_m(y) + s * C_d(y)
```

In the mixup window use the exact accepted batch-shared `mix` and permutation
for both paths:

```python
main = mix * C_m(targets_a) + (1 - mix) * C_m(targets_b)
direct = mix * C_d(targets_a) + (1 - mix) * C_d(targets_b)
loss = (1 - s) * main + s * direct
```

In the hard tail use the same convex blend with the single hard target. This is
preferable to `main + 0.1 * direct`: when the two CEs are similar, the loss and
shared-gradient scale remain near the accepted CE scale instead of silently
raising effective LR by roughly 10%. Tying the fraction to the already accepted
residual scale is prospective and removes a second knob, although it is a
structural convention rather than evidence that 0.1 is optimal. Do not tune the
fraction, schedule it, detach either path, or add consistency/distillation terms.

Gradient semantics are intentional: both losses update the shared backbone and
the same `fc`; only the 90%-weighted main loss updates `pooled_head`. No extra
classifier is introduced. Consequently the treatment both rewards a linearly
useful `z` and slightly reduces head supervision; those effects cannot be
separated by this one test.

## Expected Benefit and Risks

The direct objective may prevent the residual MLP from relying on brittle
feature corrections and keep the dominant direct representation well shaped,
potentially improving the remaining boundary/generalization gap at negligible
spatial cost. The strongest counterargument is local: EXP036's nonlinear
remapping is the accepted gain, so reallocating 10% of its CE gradient can
suppress precisely the useful mechanism. Shared `fc` gradients can also pull
class vectors toward a compromise between two feature geometries. This proposal
therefore has moderate downside and weaker evidence than the accepted head
itself; it should receive one preregistered score, not coefficient rescue.

Implementation risks are an evaluator tuple leak, applying different mixup
targets/coefficients to the two paths, computing the auxiliary path during
evaluation, changing main-logit numerics through concatenated/fused GEMM, or
accidentally detaching `z`/`fc`. The extra training classifier GEMM and CE are
small, but their backward launches can still reduce exposure.

## Preflight and Timing Gates

Use an evaluator-free oracle against `git show a7c42dc:train.py` and verify:

- identical parameter names, shapes, bytes, CPU/CUDA RNG states, optimizer
  membership, augmentation, schedule, and temporal thresholds;
- candidate default/eval logits are bitwise equal to a manually reconstructed
  accepted `fc(z + 0.1 * pooled_head(z))` on fixed inputs;
- `return_direct=True` returns `(main, direct)` where main equals the default and
  direct is exactly `fc(z)`; evaluator/default mode performs only one classifier
  call;
- independent early-mixup and hard-tail loss oracles match the exact 90/10
  formulas, including the same `mix`, targets, and permutation;
- finite two-step updates, nonzero gradients to backbone and `fc` from both
  terms, direct CE gives zero gradient to `pooled_head`, and the combined head
  gradient equals `0.9` times the main-only head gradient on a fixed fixture;
- log main/direct CE and cosine similarity of their gradients for backbone,
  `fc`, and head as non-tuning diagnostics only.

Run four counterbalanced complete-step timing windows for accepted and candidate
in both mixup and hard regimes, printing measurements before assertions. Require
stable CV and throughput retention at least `127 / 130.304 = 0.97464`, projecting
at least 127 passes from EXP036. If timing fails, do not score: the mechanism is
inconclusive because the extra CE/backward violated the protected exposure
regime.

## Falsifiable Outcome

Score once with the exact local command and frozen seed/evaluator contract.
Success requires `best_test_acc >= 94.58%`, realized exposure at least 127 passes,
and valid completion; `final_test_acc >= 94.45%` and `final_test_loss <= 0.2456`
are corroboration against a sparse best-epoch accident.

A normal-exposure miss falsifies the proposition that preserving raw pooled
linear separability through this fixed shared-classifier 90/10 objective improves
the accepted head. It closes immediate auxiliary-weight, cutoff, detach,
separate-classifier, distillation, and head-scale rescues because those add new
degrees of freedom after observing the result. It does **not** show that `z` is
not linearly separable, nor does it reject unrelated hard-tail optimization or
loss-geometry mechanisms. A sub-127-pass run rejects only this implementation's
systems viability, not the representation hypothesis.

## Evidence

- `experiments/036/04-analysis.md`: the accepted pooled residual head improved
  best/final accuracy and loss at 130.304 passes.
- `02-system-understanding.md`: generalization and boundary quality, not memory
  or input delivery, limit the objective; spatial backward dominates cost.
- `03-experiment-learnings.md`: preserve the exact pooled head and avoid adjacent
  width/scale tuning; added regularization has repeatedly overconstrained this
  fixed-budget learner.
