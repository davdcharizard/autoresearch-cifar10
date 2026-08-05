# Proposal idea-03: Normalized cosine classifier + learnable temperature (output-layer geometry)

## Core change (train.py only)
The head is `self.fc(x) * self.scale_out` with a bias-free `nn.Linear(512,10)` and a FIXED logit scale `SCALE_OUT=0.125` (train.py:153,178). Replace it with a **cosine classifier**: L2-normalize the pooled feature AND the fc weight rows, then scale by a temperature τ. Logits become `τ · cos(θ_{class})` ∈ [−τ, τ].

```python
# __init__: self.fc = nn.Linear(512, num_classes, bias=False)
#           self.logit_scale = nn.Parameter(torch.tensor(float(INIT_TAU)))  # learnable τ
def _forward_once(self, x):
    ...; x = self.pool(x).flatten(1)
    if HEAD == "cosine":
        xn = F.normalize(x, dim=1); wn = F.normalize(self.fc.weight, dim=1)
        return F.linear(xn, wn) * self.logit_scale
    return self.fc(x) * self.scale_out
```
`HEAD` env (`linear` default / `cosine`), `INIT_TAU` (default ~12–16, since with LS=0.2 and 10 classes the target softmax mass needs a moderate scale). +1 scalar param.

## Mechanism — why this is a DIFFERENT (loss-geometry) lever
The current head is an UNnormalized linear map with a hand-tuned constant scale — class scores depend on feature MAGNITUDE, so the optimizer can trivially reduce loss by inflating feature norm rather than improving angular separation. The cosine classifier removes magnitude from the decision, forcing the network to separate classes by ANGLE on a hypersphere and decoupling "confidence" (τ) from the features. This is a genuinely different output geometry (NormFace / cosine-softmax / CosFace-without-margin), orthogonal to capacity/optimizer/aug/regularization/downsampling/attention.

## Why it targets the limiter
The limiter is a budget-limited generalization ceiling (project-insights High, EXP-014). This attacks it at the **readout/loss geometry** — the one place the DavidNet recipe uses an untuned convention (`SCALE_OUT=0.125`, a fixed scalar). Normalized-feature classification is reported to improve generalization and calibration on closed-set classification by regularizing the logit geometry, and it is throughput-free (two normalizations + a scalar multiply over a [N,512]/[10,512] tensor). It directly tests whether the max-magnitude-coupled linear head leaves angular-separation accuracy on the table.

## Why it's distinct from prior work here
EXP-012 retuned the loss SCALARS (wd allocation, LS level) and found them saturated, but never touched the logit GEOMETRY — the feature/weight normalization that changes WHAT the softmax sees, not just its temperature. `SCALE_OUT` was inherited from DavidNet ("output scale is important") and never reformulated as a normalized/learnable temperature.

## Throughput
Strictly neutral: replaces a matmul + constant-multiply with two `F.normalize` calls (cheap row-norms over 512-dim) + the same matmul + a scalar multiply. num_epochs ~150 (verify). No fused-kernel risk.

## Correctness / EMA / eval
- `logit_scale` is a learnable parameter the optimizer picks up; `AveragedModel(use_buffers=True)` averages it and the fc weight. Eval path deterministic, train≡eval. Flip-TTA valid (normalization is per-sample, flip-invariant over the averaged logits).
- bf16/channels_last: `F.normalize` is stable under autocast (compute the norm in fp32 if needed; small dim). Watch for τ drift to extreme values (clamp or init sensibly).
- Interaction with LS=0.2: a bounded-logit cosine head changes the achievable softmax peak; τ must be large enough that LS=0.2's target (0.82 on the true class) is reachable — INIT_TAU≈12–16 handles this; expose τ as learnable so the net tunes it.
- Smokes: (i) `HEAD=linear` bit-identical to baseline; (ii) num_params = baseline+1; (iii) finite fwd/bwd, `logit_scale.grad` non-None; (iv) cosine logits bounded by ±τ; (v) τ stays finite/positive over a few steps.

## Design — SAME-SESSION multi-cell (verdict keyed on cA)
- c0: `HEAD=linear` (baseline) — anchor + regression.
- cA (PRIMARY): `HEAD=cosine`, learnable τ (INIT_TAU=14) — determines the verdict.
- cB (diagnostic): `HEAD=cosine` with FIXED τ (no learnable scale) OR a different INIT_TAU — isolates whether the gain (if any) is from normalization vs the learnable temperature; informational only.

## Verification
- cA ≥ 96.48 AND > same-session c0 by >0.1pp; mandatory confirmation re-run on any apparent win.
- num_epochs ≈ 150 (throughput-free); ep25 sane (the head change shifts early logit scale — judge full-anneal best≈final); fully annealed; τ converged to a sane value (logged).
- Integrity: train.py-only; prepare.py unchanged; ≤1 eval/epoch; seed 42; summary best == per-epoch max; `HEAD=linear` ≡ baseline smoke. Background nvidia-smi sampling.
- ON A WIN: bake the cosine head as default.

## Hypothesis
A normalized cosine classifier with learnable temperature forces angular class separation and decouples confidence from feature magnitude, improving generalization and lifting best_test_acc ≥96.48 over the same-session control, throughput-free at ~150 epochs. If it ties, the unnormalized linear head + tuned `SCALE_OUT` already captures the available class signal and the output-geometry lever is exhausted.

## Effort: low-medium. Risk: (1) BN before the head already controls feature scale, so normalization may be largely redundant — likely the smallest-upside of plausible mechanisms (honest prior); (2) τ/LS interaction is delicate — a bad τ could degrade (mitigated by learnable τ + sane init); (3) bounded logits can slow early convergence — judged via ep25/anneal; (4) cosine heads help most on open-set/imbalanced tasks, less on balanced closed-set CIFAR — modest EV.

## Sources
EXP-012 04-analysis.md (loss-scalar saturation, geometry untouched); NormFace / cosine-softmax (Wang et al. 2017), CosFace (Wang et al. 2018); train.py:141-178.
