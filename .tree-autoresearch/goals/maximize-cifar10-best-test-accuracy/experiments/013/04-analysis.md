# Report EXP-013: Fixed-scale-40 cosine-normalized classifier
- **Created**: 2026-08-06

## Goal

Maximize CIFAR-10 `best_test_acc` under the fixed 300-second charged-training protocol, with higher better. EXP-013 grew from EXP-011 at `95.61%`, which was also the global best; formal improvement required at least `95.71%` while preserving GPU-0, scope, budget, seed, and evaluation-cadence constraints.

## Idea & Hypothesis

Replace only the affine classifier geometry with FP32 normalized features and normalized classifier rows scaled by the preregistered constant 40. The preregistered rationale cited cosine-classifier literature suggesting that this operating point could improve angular class separation at negligible cost. The hypothesis was that better-conditioned angular margins would lift EXP-011's EMA tail enough to clear `95.71%` without reducing optimizer exposure.

## Approach

The parent `nn.Linear(256,10,bias=True)` construction and initialization order were preserved. Its initialized-zero bias was frozen, retained in optimizer and EMA state, and excluded from SAM; the forward became `40 * F.linear(F.normalize(features), F.normalize(fc.weight))` in FP32 with epsilon `1e-6`. No data, loss, optimizer, CutMix, SAM, EMA, schedule, backbone, seed, budget, or evaluator setting changed.

Setup, every-evaluation, EMA-shadow, and terminal audits measured classifier row norms, pairwise row cosines, bias, online/EMA distance, parameter counts, and ownership. Claude's implementation review required deferred audit failures so the complete terminal summary would still print and distinct geometry for EMA-sourced evaluations; both changes were diagnostic only.

## Execution

CPU and GPU-0 smokes passed formula, bound, scaling invariance, initialization/RNG parity, CutMix identity, BF16 integration, gradients, SAM replay, EMA cadence, and exact restoration. The first preflight attempt stopped before any complete measurement because a vector-valued finite predicate was used as a Python Boolean; reducing it with `.all().item()` was the one allowed harness correction.

The first complete accuracy-blind preflight passed: parent drift `0.00994`, paired-ratio dispersion `0.00256`, median latency ratio `1.00958`, projected `25553` steps, and projected `452.19s` total. Claude Opus independently authorized one metric launch. The only metric run then exited `0` on physical GPU 0 and printed a complete summary; Claude's raw-result review initially blocked one incorrect log pointer, then returned `PASS` after correction.

## Results

- **Primary metric**: `95.11%` (parent: `95.61%`, delta: `-0.50` points, `-0.52%`; global best: `95.61%`)
- **Execution metrics**: final accuracy/loss `95.07%/0.1765`; charged/total/startup time `300.0s/457.2s/1.6s`; peak allocation `1221.4 MiB`; `132` epochs and `25598` steps.
- **Dose**: CutMix `10275/20704`; SAM `2447/4894` from progress `0.7500`; EMA `158` updates split exactly `79/79` ordinary/SAM, with 26 swaps/restores and zero audit failures. This cleared the preregistered `25300`-step and 155-update floors, though exposure was about `0.8%` below the parent's `25798` steps.
- **Tail stability**: the final 16 EMA accuracies averaged `95.073750%`, ranged only `95.01-95.11`, and spanned progress `0.857810-1.000000`. The plateau was `0.419375` points below EXP-011's `95.493125%` tail mean and failed the `95.64%` mechanism target.
- **Classifier behavior**: row-norm mean grew from `1.43738306` initially to `3.26836252` online and `3.30340576` in EMA. Mean off-diagonal row cosine moved from `-0.00618019` to `0.61419338/0.61962318`, so classifier directions became strongly positively correlated. Bias remained exactly zero; normalized online/EMA distance was `0.02613891`.
- **Analysis**: the intervention achieved its intended normalized angular parameterization at full preregistered dose, but it did not produce well-separated classifier directions: mean pairwise row cosine reached `0.614`, versus `-0.111` for a maximally separated 10-class simplex, though no parent-classifier geometry was measured for direct comparison. It settled on a tight, materially lower EMA plateau; best accuracy exceeded the final-16 mean by only `0.03625` points, and final loss worsened from the parent's `0.1552` to `0.1765`. In this single fixed-seed comparison against the historical parent, the complete dose, ownership, restoration, and RNG audits make the fixed-scale cosine package the only intended intervention explaining the degradation. The one unintended difference was `0.8%` lower step exposure, consistent with measured overhead and too small to plausibly explain the full `0.50`-point gap given EXP-010's null accuracy result at `9.3%` more steps. One run per arm cannot establish statistical significance, isolate the value 40 from the implemented cosine package, or support a general non-transfer claim.
- **Key Learning**: Fixed-scale-40 cosine normalization preserved full training dose but lowered best accuracy by `0.50` points and the EMA tail by `0.42` points.

## Verification

- **Conditions**: GPU/scope/integrity/budget/evaluation/dose checks passed; formal accuracy and tail-mechanism targets failed.
- **Review Notes**: exit `0`; `132` evaluation lines equal `132` epochs and `106+26` live/EMA sources; no traceback, CUDA, OOM, audit, RuntimeError, NaN, or Inf signature. Claude's final raw-result verdict was `PASS`.
- **Verdict**: no-improvement
- **Verdict Basis**: trustworthy `95.11%` result below the required `95.71%`; the node is a failed leaf under EXP-011 at commit `d15e138`.

## Unexplored Avenues

- A cosine-family classifier with a differently justified scale or learned temperature is not ruled out; this experiment tested only scale 40 once, so it provides no evidence about behavior at other operating points and no unbiased basis for choosing one.
- Partial normalization or an explicitly decorrelating angular objective could behave differently, but would be a materially different package requiring its own preregistered evidence and RNG/throughput audit.

## Next Steps

- **Shorter-lag charged-time EMA (low confidence)**: EXP-013 produced no evidence about EMA lag, and its online/EMA distance was measured under a changed classifier geometry, so attempt this only with a preregistered mechanistic rationale rather than as horizon tuning.
- **Bounded soft-target Poly-1 (medium confidence)**: test a coherent hard/CutMix loss modification with negligible compute after preregistering one coefficient from gradient-scale evidence.
- **Memory-rich fused state changes (medium confidence)**: exploit abundant H20 memory while preserving at least 25k optimizer steps and avoiding extra full forwards.
