# Brainstorm EXP-033
**Created**: 2026-07-26

## Web Search & Literature Review

- **mixup: Beyond Empirical Risk Minimization** (`knowledge/papers/mixup.md`): symmetric Beta interpolation is the strongest demonstrated low-cost local generalizer; concentration changes endpoint mass without adding model compute.
- **When, Where and Why to Average Weights?** (`knowledge/papers/weight-averaging.md`): a carefully selected short averaging window can improve generalization cheaply, but local EXP013 shows that a long lagging whole-state EMA is the wrong implementation here.
- **When Does Label Smoothing Help?** (`knowledge/papers/label-smoothing.md`): mild target smoothing can address overconfidence, although stacking soft-target methods without calibration is risky in the accepted mixup learner.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): width and moderate depth can use accelerator compute effectively; local profiling, not static memory or MACs, must validate a new batch operating point.

No network or remote source was consulted. This offline pass used the existing per-goal knowledge base and experiment artifacts.

## Experimental History Review

- EXP027 remains the 94.32% baseline: `(2,2,3)` plus worker-safe early N1/M5 RandAugment turned two standalone near-misses into a +0.25-point improvement while retaining 133.007 passes.
- Compute is bound in forward/backward, but memory and wall time are not. Exposure-only reallocations repeatedly regressed: BF16, fixed-MAC depth redistribution, and late prefix freezing all delivered 159-172 passes without preserving accuracy.
- The accepted alpha-0.2 mixup duration is bracketed by 50% and 75% failures; alpha 0.4 and per-example coefficients failed. Alpha 0.1 is the only clean unmeasured side of batch-shared strength on the accepted deeper-plus-RandAugment learner.
- Exact early feature masking is closed after two normal-exposure regressions. Full and simplified stage-3 SE, late SAM, EMA-0.999, zero residual endpoints, rank bottlenecks, decay removal, cosine-to-zero, channels-last, batch 128, and reflection padding are also closed or infeasible in their tested forms.
- EXP013's late whole-state EMA improved intermediate loss but lagged terminal top-1, leaving a distinct short finite parameter average as an untested variance-control mechanism. Batch 512 remains the untested memory-rich counterpart to infeasible batch 128 but needs a large image-rate gate because it halves optimizer and mixup decisions.
- The limiting gap is generalization/boundary placement at near-zero tail training loss. A useful experiment must improve accuracy per backward pass or change generalization at near-zero counted cost while preserving full high-resolution adaptation and the accepted early-invariance/depth interaction.

## Collected Ideas

- **Weaker batch-shared mixup** — change only symmetric Beta alpha from 0.2 to 0.1 through the accepted 65% cutoff. It targets possible over-regularization from composing mixup, RandAugment, and extra depth while preserving batch-level coherence and all model compute; the stronger-side alpha-0.4 failure does not determine the weaker side.
- **Batch 512 with linear LR scaling** — double batch size and both LR endpoints while image-equivalently halving the safety step cap. It attacks forward/backward efficiency using the H20's 98.9% memory headroom, but must earn the loss of optimizer, BN, and batch-shared-mixup decisions through a strict measured image-rate gain.
- **Short finite parameter averaging** — average a small predetermined set of late live parameter snapshots, keep BN buffers from a specified live endpoint, and evaluate the averaged state only at the final partial epoch. It targets sparse-evaluation and SGD boundary variance without the long horizon or continuously lagging buffers that hurt EXP013.
- **Early mild label smoothing** — replace or complement early mixup targets with a small fixed epsilon and return to hard labels at 65%. It targets overconfidence at negligible cost, but overlaps mixup's soft-target mechanism and lacks a clean sign under the already regularized accepted composition.
- **Normalized classifier head** — L2-normalize penultimate features and class weights with a learned or fixed scale. It directly changes angular class boundaries at low parameter cost, but adds per-step normalization and substantially changes optimization geometry without local evidence.
- **Smooth activation substitution** — replace residual ReLUs with SiLU or GELU to preserve small-gradient information. This is an algorithmic representation change that could improve function quality, but it adds counted elementwise work in the most expensive high-resolution stages and abandons the validated preactivation recipe.
- **Remove mixup under early RandAugment** — test whether the accepted deeper learner now needs only image-space invariance. This simplification eliminates soft-label interpolation and its GPU work, but EXP002 established mixup as the largest standalone gain and no local evidence suggests complete redundancy.
- **Tensor-side crop/flip augmentation** — move basic geometry onto batched GPU tensors to reduce worker variance and enable richer boundary modes. This moonshot could unify input transformations with counted execution, but it changes RNG, interpolation, counted cost, and data ordering too broadly for a clean one-loop attribution.

## Combinations

- **Batch 512 + weaker mixup**: larger batches make each shared coefficient cover twice as many samples, so alpha 0.1 could reduce whole-batch interpolation severity while the larger batch improves image rate. The cross may balance regularization and hardware utilization better than either alone, but it entangles two uncertain operating-point changes and should follow standalone evidence.
- **Short averaging + normalized head**: angular classification could make parameter averaging more geometrically coherent near the terminal solution. The combination has more boundary-control upside than either alone, but no local evidence supports normalized logits and the state semantics would be difficult to attribute.
- **Weaker mixup + short averaging**: alpha 0.1 could give cleaner late basin entry while a finite average reduces terminal SGD variance. Both are near-zero-cost generalization controls, but a combined success could not identify whether weakening interpolation or smoothing the trajectory caused the gain.

## Candidate Ideas

### Batch 512 With a Fully Scaled LR Curve
**Summary**: Set batch 512, LR `0.4 -> 0.004`, and an image-equivalent 32,000-step safety cap while preserving all accepted training logic. Score only if balanced complete-body timing demonstrates at least 1.10x image rate and 146.308 projected passes.

**What it targets**: The 98% forward/backward counted-time bottleneck and the H20's large unused memory, seeking better convolution utilization without dropping model capacity or high-resolution gradients.

**Reasoning**: This is the unexplored large-batch counterpart to EXP029. Linear LR scaling approximately preserves cumulative first-order displacement per example, but the operating point intentionally changes optimizer, BN, epoch, and batch-shared mixup decision frequency. Local exposure failures justify the unusually high throughput gate. See `proposals/idea-02.md`.

**Sources**: `02-system-understanding.md`; `project-notes/project-insights.md`; EXP009, EXP016, EXP028, EXP029; `proposals/idea-02.md`.

**Estimated Effort**: medium

**Risk Assessment**: A passing speed gate still leaves half as many optimization and mixup decisions, altered BN statistics, and coarser hard-tail refinement. Larger batch changes worker assignment and dropped examples, so exact post-iterator trajectory identity is neither possible nor a useful gate.

### Weaker Alpha-0.1 Batch-Shared Mixup
**Summary**: Change only `MIXUP_ALPHA = 0.2` to `0.1`, preserving the batch-shared scalar, 65% cutoff, early RandAugment, model, optimizer, schedule, seed, and hard tail. This is the remaining one-shot strength bracket, not a monotonic extrapolation from alpha 0.4.

**What it targets**: Generalization and boundary quality in the accepted triple-regularized learner, testing whether its added depth plus early image invariance needs less interpolation while retaining identical GPU graph and near-identical exposure.

**Reasoning**: Alpha 0.1 raises endpoint concentration, making most mixed batches closer to clean examples without removing the interpolation prior or changing batch-level coefficient coherence. It has the cleanest attribution and closes an explicit history gap, although every neighboring mixup perturbation has been negative. See `proposals/idea-01.md`.

**Sources**: `knowledge/papers/mixup.md`; EXP002, EXP004, EXP005, EXP015, EXP020, EXP027; `proposals/idea-01.md`.

**Estimated Effort**: low

**Risk Assessment**: The accepted alpha 0.2 may already be locally calibrated; alpha 0.1 places roughly 81% of draws near endpoints and may under-regularize. Alpha-dependent Beta rejection also intentionally changes the later CUDA RNG trajectory, so the fixed-seed result cannot isolate coefficient magnitude from its deterministic draw sequence.

### Short Predetermined Parameter Averaging
**Summary**: Maintain a fixed short set of late parameter snapshots and evaluate their arithmetic mean at the final partial epoch using preregistered BN-buffer semantics, while leaving all live training and earlier evaluations unchanged. The exact window and state policy must be fixed before scoring.

**What it targets**: Terminal SGD and sparse-evaluation boundary variance, seeking a better late-basin representative without adding backward work or repeating EXP013's long-horizon lagging whole-state EMA.

**Reasoning**: The accepted run's best and final accuracies differ by 0.10 points, while EXP013 showed that averaging can reduce loss even though its 0.999 whole-state EMA lagged top-1. A short finite average may retain terminal responsiveness and avoid continuously averaging BN buffers. See `proposals/idea-03.md`.

**Sources**: `knowledge/papers/weight-averaging.md`; EXP013, EXP027; `proposals/idea-03.md`.

**Estimated Effort**: medium

**Risk Assessment**: Parameter interpolation across moving BN statistics can produce a mismatched evaluation state, and selecting a window from observed results would be metric leakage. Snapshot copies also add counted or excluded overhead depending on placement and require strict accounting to avoid changing live training exposure.

## Review

The offline adversarial reviewer selected short predetermined parameter averaging. It judged the treatment materially distinct from EXP013 and best aligned with the boundary-quality limiter while preserving all accepted updates. I adopted its significant cautions: the accepted best-final gap motivates but does not diagnose stochastic variance; terminal live BN buffers are an unresolved statistical approximation; and preflight must exercise the exact snapshot timing, counted clone cost, state integrity, exception-safe restoration, and single terminal evaluation sequence. A normal-exposure endpoint regression will implicate parameter-space curvature or BN mismatch without authorizing a window or recalibration rescue. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the reviewer's pick. Parameter averaging scored 3.5/5 for evidence and 3/5 for impact, ahead of batch 512's weaker accuracy alignment and alpha 0.1's uniformly negative local neighborhood. The chosen treatment keeps all live training and earlier evaluations identical, targets the final boundary representative at near-zero cost, and has an explicit mechanism criterion at the averaged endpoint. The critique does not establish that terminal variance exists; the one-shot experiment is the falsifiable test.

## Chosen Idea
**Selected**: Short Predetermined Parameter Averaging

**Why this idea**:
Uniformly average trainable parameters from the first post-update states whose pre-step times are at least 95% and 97.5% of budget plus the terminal live state, and use that view only for the existing final evaluation while retaining terminal live BN buffers. This directly changes the terminal boundary representative without altering the accepted optimizer trajectory, data policy, batch statistics, or earlier evaluation opportunities. Its three-point last-5% window and parameter-only state are materially different from EXP013's lagging 65%-start whole-state EMA.

**Hypothesis**:
If a small part of the accepted terminal top-1 gap comes from late iterate variance, then the exact uniform `[95%, 97.5%, 100%]` parameter average with terminal live BN buffers will retain at least 131.67 passes and produce `final_test_acc >=94.42%`, thereby raising fixed-seed `best_test_acc` from 94.32% to at least 94.42%. A normal-exposure averaged endpoint below 94.42% falsifies this exact mechanism; improved loss alone cannot rescue it.
