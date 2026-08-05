# Brainstorm EXP-022
**Created**: 2026-07-26

## Web Search & Literature Review

- **mixup: Beyond Empirical Risk Minimization** (`knowledge/papers/mixup.md`): Beta alpha controls the frequency and strength of interpolated examples; the accepted schedule leaves the weaker-than-0.2 side unmeasured.
- **RandAugment** (`knowledge/papers/randaugment.md`): a small operation count and shared magnitude can improve CIFAR generalization without policy search, and torchvision already supplies the transform.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): regularization timing matters; the accepted 65% mixup window and long clean tail should remain fixed when testing an orthogonal intervention.
- **EXP-021 local timing evidence** (`experiments/021/04-analysis.md`): every-step SAM costs 2.1523x a normal step, but sparse use can keep most accepted updates while sampling terminal sharpness gradients.

No network source was consulted; this offline loop uses the persistent peer-reviewed paper distillations and completed local experiments.

## Experimental History Review

- The accepted recipe is WRN-16-2 with batch-shared alpha-0.2 mixup until 65%, scoring 94.07% in 141.9 data passes. It remains the baseline after 19 rejected follow-ups.
- Mixup duration is bracketed around 65%: 50% scored 93.91 and 75% scored 93.82. Alpha 0.4 over-regularized at 93.57, while alpha 0.1 remains the only controlled strength-side gap.
- Added dropout, CutMix, altered weight decay, cosine-to-zero, BF16, EMA, and residual initialization all regressed or stayed below the 0.10-point acceptance margin. Any new regularizer must avoid compounding softness throughout the accepted mixup window.
- Low-resolution capacity and conditional attention were the only near-positive architecture axes, reaching 94.11-94.16, but simplifications of stage-3 attention failed repeatedly. Exact architecture replays are closed.
- EXP-021 established that every-step final-10% SAM is semantically implementable but retains only 89.67% exposure, just below its preregistered feasibility floor. A sparse cadence is materially cheaper but needs a fixed, non-adaptive rule.
- The limiting gap is fixed-budget generalization at a noise-scale acceptance boundary, not raw training exposure: faster BF16 and fixed-MAC redistribution increased passes but reduced accuracy. Remaining candidates must improve decision boundaries without sacrificing the accepted clean-refinement trajectory.

## Collected Ideas

## Combinations

## Candidate Ideas

### Weaker Alpha-0.1 Mixup
**Summary**: Change only `MIXUP_ALPHA` from 0.2 to 0.1 while retaining the accepted 65% cutoff, batch-shared coefficient, permutation behavior, schedule, and model.

**What it targets**: Fine calibration of target/input softness at the now-bracketed duration. Beta(0.1,0.1) reduces strongly interpolated batches while preserving stochastic batch-level regularization.

**Reasoning**: Alpha 0.4 clearly over-regularized and the lower side is the sole unmeasured mixup-strength direction. The intervention is maximally controlled and preserves throughput, so it can determine whether the accepted alpha sits above the local optimum.

**Sources**: `knowledge/papers/mixup.md`; EXP-002, EXP-005, EXP-015, and EXP-020; `03-experiment-learnings.md` mixup-duration entry.

**Estimated Effort**: low

**Risk Assessment**: Endpoint-heavy Beta(0.1,0.1) may under-regularize, consistent with the 50% duration regression, and alpha response need not be monotonic. Expected upside is small even if the direction is correct.

### Mild RandAugment Before the Accepted Pipeline
**Summary**: Add torchvision `RandAugment(num_ops=1, magnitude=5)` to training images before random crop, flip, tensor conversion, and the unchanged alpha-0.2 mixup schedule. The low policy strength is fixed in advance and uses no new dependency.

**What it targets**: The persistent train-to-test generalization gap using label-preserving input diversity rather than additional target softness or model-side masking. It leaves the accepted mixup duration and hard-label tail unchanged.

**Reasoning**: RandAugment reports strong CIFAR results with a compact policy space, while this model currently uses only crop and flip. One mild operation offers an orthogonal invariance signal and negligible GPU work; unlike CutMix or dropout, it does not introduce mixed spatial labels or erase residual features.

**Sources**: `knowledge/papers/randaugment.md`; EXP-003 and EXP-006 failure mechanisms; accepted EXP-002 recipe.

**Estimated Effort**: low

**Risk Assessment**: CPU transform overhead can reduce fixed-time exposure, PIL operations can be overly destructive even at magnitude 5, and the added augmentation may still compound early mixup regularization. A loader-throughput preflight must show at least 95% retention before scoring.

### Sparse Final-Window SAM
**Summary**: Apply rho-0.05 non-adaptive SAM on every other optimizer step only during the final 10% of counted training time. All other steps remain exact accepted SGD; the already validated helper semantics preserve one BatchNorm update and restore perturbed parameters exactly.

**What it targets**: The terminal solution-geometry gap implicated by the 94.07% plateau, while directly addressing EXP-021's measured exposure failure. With half of final-window steps using SAM, the measured 2.1523x step cost predicts about 94.6% rather than 89.7% whole-run exposure retention.

**Reasoning**: EXP-021 showed the mechanism can be implemented correctly but that dense cadence was too expensive. A deterministic alternating cadence samples sharpness-aware gradients throughout the same convergence window, cannot adapt to evaluation outcomes, and preserves one ordinary terminal update for every SAM update. It spends the available cost envelope while remaining above the fixed feasibility floor.

**Sources**: `experiments/021/04-analysis.md`; `03-experiment-learnings.md` low-importance SAM entry.

**Estimated Effort**: medium

**Risk Assessment**: The alternating cadence is a compute-derived compromise rather than a literature-calibrated optimum, and sparse SAM gradients may still be too weak to affect the basin. Reintroducing the complex helper increases implementation risk, so semantic and timing preflights remain mandatory.

## Review

The blind reviewer selected sparse final-window SAM at 7/10 evidence and 7/10 impact, but found the proposed one-in-four cadence unnecessarily weak. I adopt its every-other-step refinement: EXP-021's measured 2.1523x SAM-step cost projects 94.55% whole-run retention, comfortably above the fixed 90% floor while doubling sharpness-aware gradient density. I also adopt hard semantic and timing gates. RandAugment remains a future tail-only option because throughout-training use conflicts with repeated additive-regularization failures; alpha 0.1 is a low-upside closure probe. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md` and its cadence refinement. Sparse final-window SAM uniquely repairs a measured prior failure while targeting terminal solution geometry; the other candidates either compound a recurring failure mode or have insufficient upside.

## Chosen Idea
**Selected**: Alternating Final-Ten-Percent SAM

**Why this idea**:
Use rho-0.05 SAM on alternating optimizer steps only after 90% counted progress. This preserves the accepted trajectory through its high-learning-rate regime, samples sharpness gradients throughout convergence, and quantitatively fits the fixed-time envelope according to EXP-021's local measurements.

**Hypothesis**:
Alternating rho-0.05 SAM during the final 10% of counted time, with exact perturbation restoration and one persistent BatchNorm update, will retain at least 90% projected fixed-time exposure and raise fixed-seed `best_test_acc` from 94.07% to at least 94.17% by improving terminal solution geometry.
