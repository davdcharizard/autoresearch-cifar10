# Brainstorm EXP-035
**Created**: 2026-07-26

## Web Search & Literature Review

- **mixup: Beyond Empirical Risk Minimization** (`knowledge/papers/mixup.md`): convex image/target interpolation is the strongest demonstrated low-cost regularizer in this project, and alpha controls endpoint concentration.
- **When Does Label Smoothing Help?** (`knowledge/papers/label-smoothing.md`): mild uniform target smoothing can reduce overconfidence, but stacking or replacing soft-target methods requires calibration.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): early-only regularization followed by clean refinement supports preserving the accepted 65%/35% temporal structure.

No network or remote source was consulted. This offline quick pass used the existing knowledge base and developed experiment history.

## Experimental History Review

- EXP027 remains the 94.32% baseline. Preserve its `(2,2,3)` depth, early worker-safe N1/M5 RandAugment, batch-shared alpha-0.2 mixup through 65%, full FP32 gradients, schedule, decay, seed, and hard tail unless the experiment directly tests one of those elements.
- EXP034 closes exact batch 512 after only a 6.10% image-rate gain; EXP029 closed batch 128. Batch scaling, late averaging, feature masking, SAM, attention, padding, channels-last, BF16, freezing, schedule/decay changes, and nearby capacity reallocations are closed or infeasible in their tested forms.
- Alpha 0.4, per-example alpha-0.2 coefficients, and 50%/75% mixup cutoffs all regressed. Alpha 0.1 is the remaining clean weaker-side bracket on the accepted deeper-plus-RandAugment composition, but there is no observed symptom that alpha 0.2 is too strong.
- The model nearly interpolates the hard-label tail while generalization and top-1 boundaries remain limiting. A target-distribution change has near-zero counted cost, but replacing the successful mixup mechanism with label smoothing would trade away project-specific evidence for generic literature support.

## Collected Ideas

## Combinations

## Candidate Ideas

### Weaker Alpha-0.1 Batch-Shared Mixup
**Summary**: Change only `MIXUP_ALPHA=0.2` to `0.1`, retaining one shared CUDA scalar per batch, the strict 65% cutoff, model, early RandAugment, optimizer, LR curve, seed, and exact hard tail. This is the final one-shot strength bracket rather than a monotonic extrapolation from the alpha-0.4 failure.

**What it targets**: Generalization and boundary quality at source-identical graph shape and near-identical exposure, testing whether added depth plus early image invariance needs less interpolation severity.

**Reasoning**: Alpha 0.1 raises Beta variance from 0.1786 to 0.2083 and endpoint mass from 67.34% to 81.28% while preserving mean 0.5 and batch-level coefficient coherence. It has exceptional causal isolation and a complete fail-closed proposal, although every neighboring mixup perturbation is negative. See `experiments/033/proposals/idea-01.md`.

**Sources**: `knowledge/papers/mixup.md`; EXP002, EXP004, EXP005, EXP015, EXP020, EXP027; `experiments/033/proposals/idea-01.md`.

**Estimated Effort**: low

**Risk Assessment**: Endpoint-heavy batches may under-regularize. Concentration-dependent Beta rejection changes the later fixed-seed CUDA coefficient/permutation trajectory by design, so one run cannot estimate a seed-averaged treatment effect.

### Early Label Smoothing Instead of Mixup
**Summary**: Remove input interpolation and pairing, train with cross-entropy label smoothing `epsilon=0.05` only through the accepted 65% window, then retain the exact hard-label tail. Preserve early RandAugment, model, optimizer, schedule, seed, and data pipeline.

**What it targets**: The same near-zero-train-loss generalization gap through target entropy without mixed images or batch-shared coefficient noise.

**Reasoning**: Mild label smoothing has literature support for overconfident classifiers and preserves the accepted early-regularization/late-refinement structure. It is distinct from alpha tuning, but it discards EXP002's largest demonstrated gain and has no local evidence on the deeper-plus-RandAugment learner.

**Sources**: `knowledge/papers/label-smoothing.md`; `knowledge/papers/time-matters-regularization.md`; EXP002, EXP005, EXP015, EXP027.

**Estimated Effort**: low

**Risk Assessment**: Uniform class smoothing may be less informative than example-pair targets, and removing input interpolation is a larger mechanism change than its low code cost suggests. Exact epsilon is weakly evidenced, so a miss would be difficult to generalize beyond this point.

## Review

The offline adversarial reviewer selected alpha 0.1 only as a low-confidence closure experiment. I adopted its significant constraints: the hypothesis cannot infer monotonicity from alpha 0.4; the 81.28% endpoint mass creates real under-regularization risk; concentration-dependent CUDA trajectory divergence is intrinsic; and a valid normal-exposure miss closes immediate strength tuning without alpha/cutoff/RNG rescue. Label smoothing was rejected because it abandons the strongest local causal mechanism without a confidence diagnosis or justified epsilon. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the reviewer's pick. Alpha 0.1 scored 2.5/5 for evidence and 2/5 for impact versus label smoothing's 1.5/5 and 2.5/5. The chosen idea has lower speculative ceiling but much stronger attribution and a complete preregistered distribution, RNG, exposure, score, and family-closure contract.

## Chosen Idea
**Selected**: Weaker Alpha-0.1 Batch-Shared Mixup

**Why this idea**:
Change only the symmetric Beta concentration from 0.2 to 0.1 while preserving the accepted batch-shared coefficient, 65% cutoff, deeper-plus-early-RandAugment learner, optimizer, schedule, seed, and hard tail. This cleanly resolves the only unmeasured one-constant strength bracket without discarding proven input/target interpolation or stacking another regularizer.

**Hypothesis**:
If alpha 0.2 is slightly too strong only in the accepted deeper-plus-early-RandAugment composition, then alpha 0.1 will retain at least 130 passes and raise fixed-seed `best_test_acc` from 94.32% to at least 94.42%, with `final_test_acc >=94.32%` and `final_test_loss <=0.2523` as corroboration. A valid normal-exposure score below 94.42% closes adjacent batch-shared mixup-strength tuning.
