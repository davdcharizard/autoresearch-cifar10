# Brainstorm EXP-067
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- (no new external fetch) BN momentum and BN eps are standard, well-understood `nn.BatchNorm2d` hyperparameters; Lookahead (Zhang et al., NeurIPS 2019) is a standard optimizer wrapper. No high-signal external source adds beyond the project's own 66-experiment record at this point. This loop probes the last genuinely-untested feasible knobs.

## Experimental History Review
- **Current best: 96.45 (EXP-054)** = k=4 WideResNet-20 + RandomApply(AugMix w3, p=0.5) + GPU Cutout16 + cosine peak0.2/warmup0.05/Nesterov/LS0.1/WD1e-4 + compile(reduce-overhead), ~91 ep, dt 8ms. 67 experiments, 8 improvements.
- **The plateau is EXHAUSTIVELY mapped** (project-insights High). Decisive closures: augmentation from BOTH delivery paths (CPU free-but-wall-limited to the w3/p=0.5=96.45 optimum; GPU unlimited-coverage-but-epoch-limited to ~95.6; coverage=50% is a TRUE interior optimum per EXP-055/057); capacity ×4 directions; LR peak/shape/warmup; optimizer family/grad-dynamics(GC)/objective(SAM, PolyLoss)/AdamW; EMA/SWA weight-averaging; normalization-as-regularizer; eval-BN recalib; head; batch; activation; label smoothing (both sides, EXP-023/065); throughput→epochs from BOTH single-shape-floor (EXP-040/045/046) AND multi-shape-penalty (EXP-066, last loop). The project-insights High entry states verbatim: "96.45 appears at/near the k=4/300s ceiling."
- **Genuinely UNTESTED knobs remaining** (all low-ceiling, never tuned): BN momentum (default 0.1); BN eps (default 1e-5). These are normalization SUB-knobs that the "normalization-as-regularizer" closure (which tested BN-as-regularizer mechanisms, not the running-stat estimator hyperparameters) did not cover. Plus Lookahead — an optimizer WRAPPER mechanically distinct from the gradient-transform / objective / family changes already tested, though close to the closed EMA/SWA weight-averaging family.
- **Approach-specific avoid**: full augmentation coverage (closed, EXP-056/057/059); any width/depth (closed); any 2nd input shape under compile (EXP-066 penalty).

## Candidate Ideas

### 1. BN momentum reduction (0.1 → 0.05) — lower-variance eval running statistics
**Summary**: Set `momentum=0.05` on all four `BatchNorm2d` constructor sites (BasicBlock bn1/bn2 + shortcut BN + stem bn1). Single static-arg change, all else byte-identical to EXP-054.
**Reasoning**: Under heavy AugMix the per-batch BN statistics are noisy; a longer EMA window (momentum 0.05 vs 0.1) halves the running-stat update rate, lowering the eval-time estimation variance over the same augmented operating distribution. This is mechanistically DISTINCT from EXP-061's clean-data BN recalibration (which CHANGED the stat distribution to clean images and regressed) — here the operating distribution is unchanged, only the estimator's effective window lengthens. Compute-/throughput-neutral, single-graph (static kwarg → cudagraph-safe), zero wall risk (no eval-side or per-step additions).
**Sources**: train.py BN sites (L71/75/83/103); EXP-061 (BN-stat operating point, distinct); brainstorm-065/066 Idea 2 (carried).
**Estimated Effort**: Trivial (momentum kwarg ×4). Params unchanged.
**Risk Assessment**: Low, low-evidence. With cosine-to-0 the final epochs are near-frozen-weight, so default-momentum running stats are ALREADY stable; a longer window folds in slightly-staler higher-LR batches → most-likely near-noise null or mild regression. No scope/wall/throughput risk.

### 2. BN eps increase (1e-5 → 1e-3) — soft low-variance-channel down-weighting
**Summary**: Set `eps=1e-3` on all `BatchNorm2d` sites. Single static-arg change.
**Reasoning**: Larger eps shrinks the normalized output of low-variance channels (BN divides by sqrt(var+eps)), mildly down-weighting less-informative channels — a soft implicit regularizer, untested.
**Sources**: train.py BN sites; brainstorm-065/066 Idea 3 (carried).
**Estimated Effort**: Trivial. Compute-neutral, single-graph, wall-safe.
**Risk Assessment**: Low, very-low-evidence. On well-activated k=4 channels eps 1e-5 vs 1e-3 is negligible for most channels → near-certain exact null.

### 3. Lookahead optimizer wrapper (k=5, α=0.5) around Nesterov SGD
**Summary**: Wrap the existing SGD+Nesterov in a Lookahead meta-optimizer: keep fast weights stepping normally; every k=5 steps interpolate a slow-weight copy toward the fast weights by α=0.5 and reset fast→slow. Implement inline in train.py (no new dep — it is ~15 lines of tensor ops on `optimizer.param_groups`).
**Reasoning**: Lookahead reduces optimization variance via a fast/slow weight split, often giving small CIFAR gains by stabilizing the trajectory — and it adds ~zero compute (the slow-weight interpolation is a cheap eager tensor op every 5th step, OUTSIDE the compiled forward → cudagraph-safe). It is mechanically distinct from the tested gradient-transform (GC), objective (SAM/PolyLoss), and family (AdamW) optimizer changes.
**Sources**: train.py optimizer setup + training loop; Zhang et al. NeurIPS 2019.
**Estimated Effort**: Low (inline slow-weight buffer + periodic interpolation in the loop).
**Risk Assessment**: Moderate-low. Mechanistically CLOSE to the closed EMA/SWA weight-averaging family (EXP-006/019/020, all null/near-miss) — Lookahead's slow weights are a form of trajectory averaging, so the precedent leans null. Some implementation risk (correct fast/slow bookkeeping with the time-fraction LR), but contained to train.py. No wall risk.

## Idea Evaluation
- **Evidence strength**: all three are low-evidence probes on an exhaustively-mapped plateau (the honest state at experiment 67). Ideas 1/2 are genuinely-untested knobs with the cleanest implementation and zero wall/throughput/cudagraph risk; Idea 3 has a marginally more interesting "new mechanism" story but is undercut by the closed EMA/SWA precedent (weight-space averaging already failed here) AND carries implementation risk.
- **Mechanism clarity**: Idea 1 clear (longer EMA window → lower eval-stat variance) but the cosine-to-0 tail already stabilizes stats. Idea 2 real-but-negligible-magnitude. Idea 3 clear but proximate to a closed family.
- **Expected impact**: all near-noise. None has a strong reason to clear +0.1pp; this is plateau-mapping, not a likely breakthrough.
- **Risk profile**: Ideas 1/2 safest (trivial static kwargs, fail gracefully to no-improvement, zero wall risk — important given the recipe's 3 wall breaches). Idea 3 riskier (implementation + closed-family precedent).
- **Feasibility**: Ideas 1/2 trivial; Idea 3 low-moderate.
- **Conclusion**: Lead with **Idea 1 (BN momentum 0.1→0.05)** — the cleanest genuinely-untested, wall-safe, single-graph knob, mapping the BN running-stat-window axis with a clear (if weakly-evidenced) mechanism. BN-eps is the trivial fallback; Lookahead is deferred (closed-family precedent + implementation risk make it lower-EV than a clean static-kwarg probe). Honest expectation: near-noise null, run per NEVER-STOP to close the last untested feasible knobs.

## Chosen Idea
**Selected**: BN momentum reduction (0.1 → 0.05) on all BatchNorm2d sites, all else byte-identical to EXP-054.

**Why this idea**: After 66 experiments every major lever is decisively closed (augmentation both paths, capacity, LR, optimizer, EMA/SWA, normalization-as-regularizer, head, batch, activation, label smoothing, throughput→epochs both directions). BN momentum is one of only two genuinely-untested feasible knobs (with BN eps), it has a clean mechanism (longer EMA window → lower-variance eval running statistics over the noisy AugMix operating distribution), and it is the SAFEST remaining probe — a static kwarg with zero wall, throughput, or cudagraph risk (critical given the recipe's 3 recorded wall breaches and the EXP-066 multi-graph penalty). It maps the BN-stat-window axis cleanly whether it helps or not.

**Hypothesis**: Lowering BN momentum to 0.05 yields lower-variance eval-time running statistics and raises best_test_acc to ≥ 96.55 (baseline 96.45 + 0.1pp). Given the cosine-to-0 tail already drives near-frozen-weight, stable running stats, the most likely outcome is a within-noise null or mild regression that closes the BN-momentum knob — but it is genuinely untested, trivial, and wall-safe.
