# Proposal: Fixed-Scale Cosine Classifier

## Exact Treatment

Replace only the final affine application with a fixed-scale angular
classifier while retaining the accepted `fc.weight` tensor and initialization:

```python
CLASSIFIER_SCALE = 10.0
return CLASSIFIER_SCALE * F.linear(
    F.normalize(out, dim=1), F.normalize(self.fc.weight, dim=1)
)
```

Construct `self.fc = nn.Linear(128, 10, bias=False)` and preserve Kaiming
initialization. Keep `fc.weight` in the existing `5e-4` matrix-decay group.
There is no learned temperature, margin, bias, auxiliary loss, private seed,
or phase switch. All backbone, data, mixup, RandAugment, optimizer, LR, budget,
seed, and evaluator choices remain accepted.

## Rationale

The accepted learner nearly interpolates its hard tail but retains a 5.68%
top-1 error and 0.2523 loss. A cosine head makes class selection depend on
feature/class-vector angle rather than feature and weight norms, directly
changing boundary geometry after the expensive spatial backbone at negligible
MAC cost. It is qualitatively distinct from parameter averaging, SE gates,
spatial bottlenecks, and classifier decay allocation.

The weakness is scale selection. `10.0` is a prospective single operating
point, not locally calibrated evidence. A learned scale would add a second
optimization mechanism and a scale sweep would spend multiple fixed-seed
scores, so neither is allowed in this experiment.

## Preflight and Exposure Gate

Use an evaluator-free `git show 67c8e98:train.py` oracle. Require exact common
model tensors/RNG, 987,088 parameters after removal of ten classifier biases,
unit-norm normalized rows/features within FP32 tolerance, finite logits/losses,
nonzero gradients through feature and weight normalization, exact independent
formula agreement, scale exactly 10, and no RNG consumption. The accepted
direct affine path must remain independently reproducible from common state.

Time complete accepted/candidate early-mixup and hard updates on one idle H20
with at least 20 warmups and four counterbalanced windows of at least 50 steps.
Print raw window means and population CVs before assertions. Require every CV
at most 5%, fixed-time retention at least 0.9774, projected passes at least
130.0 from accepted 133.00736, and safe memory. No loader timing is needed.

## Sole Score and Closure

On all gates passing, run exactly once with seed 42:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require a valid finite completion, 300.0-300.1 counted seconds, total under
600, correct transitions/evaluation cadence, exact parameter count, and at
least 130 realized passes. Success is solely `best_test_acc >=94.42%`;
`final_test_acc >=94.32%` and `final_test_loss <=0.2523` are corroboration.

A valid miss closes fixed-scale cosine classification at scale 10. Do not
rescue it with scale 5/15/20/30, learned scale, angular margin, classifier
bias, another seed, or phase-specific normalization. A pre-score failure
closes only this exact implementation.

## Risks

- Scale 10 has no local calibration and changes initial logit magnitude and
  gradient scale across every layer, so this may be an optimizer intervention
  disguised as a boundary representation.
- Weight decay has no forward radial effect after weight normalization but
  still affects the underlying SGD parameterization, complicating mechanism
  interpretation.
- Normalizing near-small pooled features can amplify noise, and removing the
  bias may discard a useful class-prior offset.
- One fixed seed cannot estimate average effect.

## Falsifiable Hypothesis

If radial logit degrees of freedom are the main obstacle to generalizable
class boundaries, fixed-scale cosine classification will retain at least 130
passes and raise best accuracy from 94.32% to 94.42%. A valid miss falsifies
this exact scale-10 angular operating point, not every normalized classifier.
