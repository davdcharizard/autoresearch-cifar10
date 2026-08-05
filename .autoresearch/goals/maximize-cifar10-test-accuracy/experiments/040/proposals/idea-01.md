# Proposal: Frobenius-Preserving Equal-Row-Norm Classifier

## Recommendation

Test one forward-only classifier reparameterization on accepted `a7c42dc`.
Keep the raw `fc.weight` parameter, its initialization, optimizer membership,
coupled `5e-4` decay, LR, Nesterov state, and bias unchanged. In every training
and evaluation forward, replace its effective rows by their directions times
the differentiable root-mean-square raw row norm. Do not normalize pooled
features, add a learned/fixed temperature, remove the bias, project weights
after updates, or add an auxiliary geometry loss.

This is a narrow test of class-specific radial freedom. EXP037 and EXP038
bracketed the *amount of raw classifier decay* around accepted `5e-4`; they did
not test whether different class-vector radii should directly enter logits.
The proposed map preserves the classifier's complete Frobenius scale at every
forward while making the ten effective radii equal, so no logit-scale sweep is
needed.

## Exact Forward Map

Let raw classifier weight `W` have shape `C x D = 10 x 128`, with row vectors
`w_i`. Define

```text
r_i = ||w_i||_2
s   = sqrt((1/C) * sum_i r_i^2) = ||W||_F / sqrt(C)
n_i = w_i / r_i
W_eff[i] = s * n_i
```

The production expression should be the direct differentiable equivalent:

```python
classifier_weight = self.fc.weight
row_norms = torch.linalg.vector_norm(classifier_weight, dim=1, keepdim=True)
rms_row_norm = torch.linalg.vector_norm(classifier_weight) / math.sqrt(
    classifier_weight.size(0)
)
effective_weight = classifier_weight * (rms_row_norm / row_norms)
return F.linear(out, effective_weight, self.fc.bias)
```

Using `classifier_weight.square().mean(dim=0).sum().sqrt()` for `s` is
algebraically equivalent, but the independent verifier should use a different
formula. Do not detach either norm, use the arithmetic mean row norm, normalize
features, or mutate `fc.weight.data`. Do not add an epsilon or clamp in the
scored definition: that would sacrifice exact Frobenius preservation and
introduce an arbitrary scale. The semantic gate must instead prove all raw row
norms are finite and safely nonzero at initialization and after fixed finite
update fixtures; production already fails on non-finite loss if later training
becomes invalid.

For all nonzero rows, the intended invariants hold exactly in real arithmetic:

```text
||W_eff[i]||_2 = s                     for every class i
||W_eff||_F^2  = sum_i s^2
               = C * (||W||_F^2 / C)
               = ||W||_F^2
W_eff[i] / ||W_eff[i]|| = w_i / ||w_i||
```

Thus the treatment preserves the existing global classifier scale and every
raw row direction at a given parameter state. It removes only direct
class-specific radius differences from the forward map. `fc.bias` still allows
class-specific additive offsets; removing it would be a separate intervention.

Implementation should preferably add a small `classifier` method or keep the
four operations at the existing final return. All other forward operations,
including `out + 0.1 * pooled_head(out)`, remain in the accepted order.

## Gradient and Decay Semantics

The reparameterization changes training, not just inference. Let
`q_i = dL/d W_eff[i]`, `n_i = w_i/r_i`, and
`A = sum_k <q_k, n_k>`. Differentiating the exact map gives

```text
dL/dw_i = (s/r_i) * (I - n_i n_i^T) q_i
          + (w_i/(C*s)) * A.
```

The first term is the class-specific tangential update to direction. The second
is a globally coupled radial term arising from the shared RMS scale. Its
first-order fractional radial effect is common across rows:

```text
<n_i, dL/dw_i> / r_i = A / (C*s).
```

Consequently raw radii are not separately trained by class-local radial loss;
to first order they scale together, while each class direction remains free.
Finite optimizer steps can still change raw norm ratios through tangential
second-order effects and Nesterov history. Raw unequal norms are therefore
latent parameterization state, not a per-class inference scale.

Accepted SGD uses coupled L2 decay. The optimizer receives the autograd
gradient above and then adds `WEIGHT_DECAY * w_i` before momentum/Nesterov.
Pure coupled decay scales every raw row equally and leaves directions and raw
norm ratios unchanged while shrinking `s`, so decay still regularizes the
effective global logit scale. It no longer directly imposes independently
visible per-class radii. This is distinct from EXP037/038: their raw forward map
was unchanged and only the coefficient on `fc.weight` differed; this proposal
keeps the accepted coefficient but alters its geometric meaning.

There is no new parameter or optimizer group. The model remains 1,003,482
parameters. State-dict keys, tensor shapes, initialization bytes, optimizer
membership, and Nesterov buffer shapes must be identical to accepted.

## Initialization Diagnostics

An evaluator-free CPU construction with the accepted seed 42 and source
measured the following initial `fc.weight` geometry:

```text
raw row-norm min / max:          1.2768594 / 1.6248019
raw row-norm mean / RMS:         1.4198872 / 1.4233266
raw row-norm population CV:      0.0696447
raw max/min row-norm ratio:      1.2724986
relative ||W_eff - W||_F:        0.0695184
max absolute weight difference:  0.0399976
effective row-norm min / max:    1.4233265 / 1.4233267
raw/effective Frobenius norm:    4.5009542 / 4.5009542
```

On 32 fixed synthetic CIFAR-shaped inputs passed through the accepted
initialized backbone and pooled head, raw/effective logit RMS was
`1.2050481 / 1.1900173`; perturbation RMS was `0.0454522`, or 3.77% of raw
logit RMS, with maximum absolute logit delta `0.1385073`. Seven of 32 random
untrained argmaxes differed. These are characterization diagnostics, not
acceptance criteria or evidence of benefit.

The treatment is therefore neither an identity nor a violent scale reset. It
removes a real 27.25% initial max/min radius spread while preserving total
classifier energy, but the 6.95% weight-space perturbation is large enough to
change early predictions and gradients. This supplies a measurable mechanism
and also the primary risk: Kaiming row-norm variation may be benign finite-width
noise, or class-specific radius freedom learned later may be useful even on a
balanced dataset.

## Expected Benefit and Risks

Potential benefit is balanced angular competition among the ten class vectors.
Because pooled-feature norms, classifier bias, directions, and common
Frobenius scale remain available, the intervention is less restrictive than a
full cosine classifier and avoids choosing a temperature. It is essentially
free relative to spatial forward/backward and directly targets the remaining
boundary/generalization gap identified after EXP036.

Risks are material:

- Equal effective radii can erase useful class-specific margin/calibration
  differences caused by unequal within-class variation, despite CIFAR-10's
  balanced class counts.
- The differentiable common scale couples all classifier rows: a radial signal
  from one class changes the scale used by every class.
- Raw norm ratios become mostly latent degrees of freedom, which can interact
  with Nesterov conditioning; tangential gradients are scaled by `s/r_i`.
- Initial logits are changed from step zero. Exact parameter-byte preservation
  does not imply accepted trajectory preservation.
- `torch.linalg.vector_norm` adds small reduction/division kernels to every
  forward and their backward graph. The tensor is tiny, but launch overhead can
  matter in a 130-pass fixed-time regime.
- Ordinary coupled decay remains meaningful only as common-scale shrinkage plus
  raw-state regularization. The result must not be described as evidence for a
  different decay coefficient.
- A single fixed seed cannot estimate average treatment effect, and row-norm
  variance itself depends on the accepted initialization draw.

## Semantic Preflight

Use a disposable evaluator-free harness with an independent
`git show a7c42dc:train.py` accepted oracle. Before timing or scoring, require:

- production changes only the final classifier forward computation in
  `train.py`; `prepare.py` and every other production file are byte-identical;
- accepted/candidate state-dict keys, tensor shapes and bytes, parameter count
  1,003,482, buffers, post-construction CPU/CUDA RNG states, optimizer groups,
  group options/state, data/transforms, constants, schedule, temporal gates,
  loss branches, and evaluation cadence are exact;
- candidate `fc.weight` is the same parameter object present exactly once in
  the accepted rank-2 decay group at `5e-4`; `fc.bias` remains zero-decay;
- independent implementations of `r_i`, `s`, and `W_eff` agree with production
  on fixed CPU float64, CPU float32, and CUDA float32 fixtures;
- every effective row norm equals independently computed `s`, effective and raw
  Frobenius norms agree within dtype-appropriate tolerance, directions have
  cosine one, and bias contribution is unchanged;
- initial accepted-seed diagnostics reproduce the values above within declared
  tolerances and all row norms remain comfortably above zero;
- candidate logits match independent `F.linear(z, s*w_i/r_i, bias)` on fixed
  pooled features and full inputs; accepted/candidate pooled features before
  classification are bitwise equal, while candidate logits differ
  nontrivially from accepted;
- global positive scaling `W -> aW` produces `W_eff -> aW_eff`, while isolated
  raw row rescaling changes effective directions not at all and affects logits
  only through the recomputed shared scale;
- `torch.autograd.gradcheck` passes in float64 for raw weight, pooled features,
  and bias on a small nonzero fixture;
- the production raw-weight gradient matches the independent analytic formula
  above in float64 and a separately coded differentiable oracle in float32;
  explicitly verify the tangential projection and common fractional radial
  derivative;
- fresh and preseeded-momentum complete candidate updates match an independent
  coupled-SGD/Nesterov oracle using the transformed-forward gradient plus
  `5e-4 * W`; all other tensor updates are attributable only to changed logits
  and remain finite;
- restoring candidate model/optimizer/input/RNG reproduces early-mixup and
  hard-label steps exactly; no extra RNG is consumed;
- source audit confirms unchanged one-way mixup/RandAugment transitions,
  once-per-epoch maximum evaluation, time budget, and final summary.

Print every measured invariant, minimum row norm, analytic/autograd maximum
gradient delta, initialization diagnostics, optimizer memberships, and update
oracle deltas before assertions. A semantic failure closes only the exact
implementation. Do not repair it with epsilon, detachment, a learned gain,
post-step projection, or altered optimizer allocation; those are different
mechanisms.

## Throughput and Exposure Gate

Compare accepted and candidate complete production-equivalent steps on the
idle H20 in both early mixup and hard-label regimes. Include H2D, LR
calculation/group writes, zeroing, mixup when active, full forward, loss,
finite guard, backward, coupled Nesterov step, and final synchronization. Use
at least 20 warmups and four counterbalanced windows of at least 50 steps per
arm/regime, with fresh deterministic fixtures per replicate.

Print all raw windows before assertions. Require finite measurements and
population CV no greater than 5% for every arm/regime. Compute from median
seconds per complete step:

```text
retention =
    (0.65 / candidate_early_ms + 0.35 / candidate_hard_ms) /
    (0.65 / accepted_early_ms  + 0.35 / accepted_hard_ms)
projected_passes = 130.304 * retention
```

Proceed only if retention is at least
`127 / 130.304 = 0.9746439096` and projected passes are at least 127. A stable
miss is final: do not rerun timing, relax the floor, cache/detach norms, fuse
the operation into a persistent projected parameter, or project after
optimizer steps. No loader benchmark is needed because worker and transform
paths are source-identical.

## Sole Scored Run and Decision Contract

After gates pass, reconfirm baseline 94.48% at `a7c42dc`, one idle NVIDIA H20,
local CIFAR-10, frozen `prepare.py`, no stale log, and exact production diff.
Run exactly once at seed 42:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, one finite complete summary, 300.0-300.1 counted seconds,
wall time below 600 seconds, 1,003,482 parameters, accepted ordered temporal
transitions, unique every-fifth-epoch evaluations plus final partial epoch,
and no traceback, OOM, worker, evaluator, or non-finite error. Record realized
passes as `num_steps * 256 / 50000`.

Primary success is only `best_test_acc >= 94.58%`, exactly 0.10 points above
accepted 94.48%. Pre-register `final_test_acc >= 94.45%` and
`final_test_loss <= 0.2456` as non-decisive corroboration that equal-radius
geometry did not merely create a sparse best epoch or degrade confidence.
Neither rescues a primary miss; a primary success remains valid without them
but should be interpreted as fragile.

A valid score below 127 realized passes counts and cannot be rerun, but is
operationally inconclusive for the intended near-zero-cost geometry mechanism.
A timeout, malformed summary, wrong graph/state/transition, or repeated epoch
evaluation is a failure rather than a weak score.

## Interpretation and Restrained Closure

**Normal-exposure success:** At least 127 passes and 94.58% supports the narrow
claim that removing direct class-specific radius variation while preserving
common classifier energy improves this fixed-seed pooled-head learner. It does
not prove angular classifiers generally superior, justify feature
normalization, establish ideal calibration, or license a gain/epsilon/norm
sweep. Preserve the exact map if accepted.

**Normal-exposure miss:** At least 127 passes below 94.58% falsifies this exact
Frobenius-preserving equal-row-norm forward map as a useful standalone
refinement. Retain the ordinary affine classifier. Close immediate cosmetic
rescues of this same map: arithmetic-mean norm, detached RMS, epsilon/clamp,
bias removal added afterward, alternate norm implementation, another seed, or
a rerun. Do not infer that all normalized classifiers fail: learned global
gain, independently motivated cosine feature normalization, orthogonality
regularization, or training-only projection have materially different scale,
feature, objective, or optimizer semantics and remain formally untested, albeit
low priority without a new diagnosis.

**Low exposure or pre-score failure:** Below 127 realized passes, reject only
this implementation's systems viability. Semantic/timing failure supplies no
accuracy evidence and closes only the exact implementation. Do not convert the
forward map into an after-step projection to recover speed because that changes
momentum and decay semantics.

## Falsifiable Hypothesis

If class-specific classifier radii are a harmful source of radial class bias
after accepted nonlinear pooled refinement, then replacing the effective
`10 x 128` classifier rows by their directions times the differentiable RMS raw
row norm will preserve at least 127 projected and realized passes and raise
fixed-seed `best_test_acc` from 94.48% to at least 94.58%, with final accuracy
at least 94.45% and test loss no worse than 0.2456 as corroboration.

A valid normal-exposure miss falsifies only this parameter-free,
Frobenius-preserving equal-row-norm forward treatment and closes its immediate
implementation variants, not all classifier geometry mechanisms.

## Local Evidence

- `experiments/036/04-analysis.md`: the accepted pooled residual head improved
  best/final accuracy and loss at 130.304 passes, making cheap post-pooling
  geometry a plausible location for further work.
- `experiments/037/04-analysis.md` and `experiments/038/04-analysis.md`: zero
  and doubled classifier decay both missed at normal exposure, protecting the
  accepted `5e-4` coefficient while leaving forward geometry untested.
- `02-system-understanding.md`: boundary/generalization quality and counted GPU
  compute are binding; memory and input delivery are not.
- `03-experiment-learnings.md`: preserve the exact accepted head, LR, temporal
  regularization, and classifier decay; prefer orthogonal near-zero-spatial-cost
  mechanisms over adjacent tuning.
