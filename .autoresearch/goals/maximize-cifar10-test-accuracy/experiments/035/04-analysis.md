# Report EXP-035: Weaker Alpha-0.1 Batch-Shared Mixup
- **Created**: 2026-07-27

## Goal

Raise fixed-seed CIFAR-10 `best_test_acc` from the accepted 94.32% baseline to
at least 94.42% within 300 counted training seconds and the 600-second wall
limit. This experiment tested whether the accepted deeper-plus-RandAugment
learner was slightly over-regularized by alpha-0.2 early mixup.

## Idea & Hypothesis

Change only symmetric batch-shared mixup concentration from alpha 0.2 to 0.1,
preserving its strict 65% cutoff and every accepted model, optimizer, schedule,
augmentation, seed, worker, and evaluator choice. Alpha 0.1 was the only
unmeasured one-constant weaker-side bracket after alpha 0.4 had failed on the
earlier learner. The hypothesis required at least 130 realized passes and a
best score of 94.42%, with final accuracy at least 94.32% and loss at most
0.2523 as non-decisive boundary-quality corroboration.

## Approach

The complete production diff was `MIXUP_ALPHA = 0.2 -> 0.1`. An ignored
evaluator-free harness imported `67c8e98:train.py` independently and proved
the exact one-line source scope, byte-equal initial model/construction RNG,
optimizer and schedule identity, 987,098 parameters, candidate self-replay,
batch-shared pixel/target alignment, bitwise hard-path identity, and strict
cutoff semantics.

The harness separately sampled 100,000 coefficients per concentration and ran
four counterbalanced complete-step H20 timing windows per arm and regime.
Accepted/candidate mixed trajectories were intentionally not aligned after the
first draw because concentration-dependent gamma rejection is part of the
fixed-seed treatment.

## Execution

Both gates passed without adjustment or retry. Alpha 0.1 measured mean
0.499486, variance 0.208358, central mass 12.077%, and endpoint mass 81.319%,
versus alpha 0.2 variance 0.178884, central mass 21.325%, and endpoint mass
67.436%. Complete-step timing projected 1.003557 retention and 133.48049
passes; all CVs were below 0.72% and the hard-path ratio was 0.999981.

The sole score exited 0. Mixup disabled at step 16,723/195.0 seconds and
RandAugment at the epoch-86 iterator boundary 47 steps later. The run produced
27 unique evaluations, completed 26,012 steps and 133.18144 passes, and had no
numerical, CUDA, worker, evaluator, or source-integrity error.

## Results

- **Primary metric**: 93.72% (baseline: 94.32%, delta: -0.60 points, -0.64%)
- **Observations**: Final accuracy equaled best at 93.72%, 0.50 points below the accepted 94.22% endpoint. Final loss worsened by 0.0247 to 0.2770. Realized exposure stayed in the protected regime and was 0.17408 passes above accepted, while counted/wall time and VRAM remained accepted-like at 300.0/342.8 seconds and 1,096.3 MiB.
- **Analysis**: The intervention achieved its intended distribution change without a compute or exposure penalty, but it moved many more early batches toward nearly clean/swapped endpoints and degraded both top-1 and loss. Normal exposure, a stable best-equals-final endpoint, and exact hard-tail semantics rule out insufficient training or throughput as the useful explanation. Together with alpha 0.4's earlier normal-exposure regression, this brackets the locally useful strength around alpha 0.2. The fixed-seed alpha-dependent CUDA trajectory means this does not estimate a continuous or seed-averaged optimum, but the 0.60-point accuracy and 0.0247-loss regressions provide no basis for an adjacent rescue.
- **Key Learning**: Batch-shared mixup is locally strength-calibrated near alpha 0.2; both weaker alpha 0.1 and stronger alpha 0.4 reduce normal-exposure accuracy and worsen loss.

## Verification

- **Conditions**: Run integrity passed; the primary 94.42% threshold and both endpoint corroboration conditions failed.
- **Review Notes**: Results are trustworthy. The one allowed production line changed, the frozen evaluator and 300-second budget were preserved, one idle H20 was used, exposure exceeded 130 passes, evaluation cadence was compliant, and all distribution/semantic/timing evidence was printed before its assertions.
- **Verdict**: no-improvement
- **Verdict Basis**: A valid completed result scored 0.60 points below baseline and 0.70 below the required improvement threshold, with worse endpoint loss.

## Unexplored Avenues

None identified within immediate mixup-strength tuning. Alpha 0.1 and 0.4 now
bracket accepted alpha 0.2, 50%/75% cutoffs have already failed around the 65%
window, and per-example coefficients have already lost useful batch coherence.
Adjacent alpha values, another seed, symmetrization, cutoff adjustment, or RNG
realignment would be post-result rescues rather than independently motivated
mechanisms.

## Next Steps

- **Low confidence**: Develop an exact low-cost classifier-boundary treatment, such as a normalized-logit or margin mechanism, only after deriving its optimization and hard-tail behavior.
- **Low confidence**: Explore a qualitatively different target prior only if training diagnostics provide a concrete confidence or calibration failure, rather than replacing proven mixup generically.
- **Low confidence**: Seek a compute-neutral internal transformation of the accepted extra stage-3 block that preserves full residual and high-resolution gradient adaptation.
