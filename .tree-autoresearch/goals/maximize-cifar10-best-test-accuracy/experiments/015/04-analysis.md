# Report EXP-015: Confidence-attenuating soft-target Poly-1
- **Created**: 2026-08-06

## Goal

Maximize CIFAR-10 `best_test_acc` under the fixed 300-second charged-training protocol. EXP-015 grew from parent/global-best EXP-011 at 95.61%; formal improvement required at least 95.71% with physical GPU 0, seed 42, `train.py`-only scope, one evaluation per epoch, and a complete bounded run.

## Idea & Hypothesis

Apply fixed negative Poly-1, `CE(q,p)-0.25*(1-q dot p)`, to every optimizer-driving hard or CutMix gradient. Hard-example gradient direction stays unchanged while its scale becomes `1-0.25*p_y`, attenuating confident examples by up to 25% and relatively preserving residual errors. To retain validated SAM geometry, the base perturbation pass stays plain CE and only the perturbed descent pass uses Poly-1. The hypothesis was that this low-overhead boundary weighting would preserve at least 25,300 updates and lift best accuracy to 95.71% with a final-16 EMA mean of at least 95.69%.

## Approach

Only `train.py` changed. A loss-audit object computes parent `F.cross_entropy` plus one FP32 softmax/gather Poly term, supports hard and area-corrected CutMix targets, and owns counters at actual loss call sites. Ordinary/CutMix paths use negative Poly-1; SAM uses CE ascent, exact RNG replay and BN suppression, then Poly descent before restoration and one Nesterov update. Model, data, optimizer, CutMix, drop path, SAM schedule/rho, EMA, evaluator, seed, and timing remain EXP-011. Evaluation progress and terminal train loss are outside charged work.

## Execution

The initial analytic smoke exposed only a test-precision issue: forcing FP32 probabilities on FP64 fixtures missed `1e-10` tolerance. The helper was corrected before GPU work to preserve FP64 fixtures while promoting BF16/FP16 production logits to FP32. Formula, dense/sparse CutMix, RNG, batch-256 CPU/GPU gradients, SAM, BN, EMA parity, success/exception restoration, and 610.284 MiB candidate-only memory checks passed.

The first and only complete preflight passed with median latency ratio 1.004224, maximum 1.004677, parent drift 0.001291, ratio dispersion 0.000452, and projected 25,689 steps. The sole metric run then exited 0 on physical GPU 0. Claude Opus independently returned PASS on every timing, counter, cadence, tail, and classification calculation.

## Results

- **Primary metric**: 95.34% (parent/global best: 95.61%; delta: -0.27 points, -0.28%; formal threshold shortfall: 0.37 points)
- **Observations**: The run completed 25,820 steps, 133 epochs/evaluations, 300.0 charged seconds, 438.8 total seconds, and used 1,222.4 MiB. CutMix applied 10,362/20,884 times; SAM applied 2,468/4,936 from progress 0.7500. EMA made 159 updates split 79/80, routed 107 live plus 26 EMA evaluations, and completed 26 exact restores with zero audit failures. Loss calls closed exactly: 12,990 ordinary Poly + 10,362 CutMix Poly + 2,468 Poly descents = 25,820 optimizer-driving Poly calls, plus 2,468 CE SAM ascents.
- **Analysis**: Full optimizer, SAM, and EMA dose rules out under-training or timing rejection as explanations for the miss. The final-16 EMA sequence averaged 95.234375%, ranged 95.15-95.34, ended at 95.16%, and declined from its early maximum; it was 0.258750 points below the parent tail mean. Best-minus-tail premium was similar to the parent (0.105625 vs 0.116875), so the deficit is stable-tail movement rather than different maximum-selection opportunity. Frozen evaluator CE loss also worsened from parent 0.1552 to 0.1645. The evidence rejects this exact negative-epsilon, CE-ascent/Poly-descent package, but cannot isolate epsilon from CutMix soft-target, SAM-objective mismatch, and EMA interactions or reject other PolyLoss operating points.
- **Key Learning**: Fixed epsilon -0.25 Poly-1 preserved full dose but lowered best accuracy 0.27 points and EMA tail mean 0.26 points.

## Verification

- **Conditions**: GPU/scope/integrity/budget/evaluation/dose checks passed. Primary accuracy failed (`95.34 <95.71`), and the scientific tail target failed (`95.234375 <95.69`).
- **Review Notes**: Exit 0; 133 evaluation lines equal 133 epochs and 107+26 sources; all loss, CutMix, SAM, EMA, restoration, coverage, and RNG counts reconcile; no error/nonfinite signature. Claude's final raw-result verdict was PASS, with classification `no-improvement` at 95.34.
- **Verdict**: no-improvement
- **Verdict Basis**: Trustworthy, fully dosed metric below the parent-relative threshold; no hard-constraint or integrity violation.

## Unexplored Avenues

- Positive epsilon would upweight confident examples and sharpen unequal CutMix targets; it is a materially different operating point, not supported or ruled out by this result.
- Matching Poly-1 on both SAM ascent and descent would remove the deliberate objective mismatch, but co-changes a validated SAM component and needs a new fixed rationale.
- Restricting Poly-1 to clean non-CutMix steps would avoid soft-target interaction, but introduces a phase choice and tests a much smaller dose.

## Next Steps

- **High confidence**: Treat EXP-011 as locally saturated for narrow output-loss edits and use navigation policy to consider a different successful base or a materially new representation mechanism.
- **Medium confidence**: Consider activation-anchored EMA bias correction only as a precise estimator study, acknowledging its likely sub-threshold effect size.
- **Low-medium confidence**: Revisit symmetry-aware or bottlenecked representations with a fixed H20 latency gate rather than another scalar calibration point.
