# Brainstorm EXP-068
**Created**: 2026-06-10
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- (no new external fetch) Lookahead (Zhang, Lucas, Ba, Hinton — "Lookahead Optimizer: k steps forward, 1 step back", NeurIPS 2019) is a standard, well-understood meta-optimizer: maintain fast weights (stepped by the inner optimizer) and slow weights; every k inner steps, pull slow toward fast by α and reset fast→slow. Reported small-but-consistent CIFAR gains and variance reduction. BN eps / SGD-momentum coefficient are standard hyperparameters. No high-signal source adds beyond the project's own 67-experiment record.

## Experimental History Review
- **Current best: 96.45 (EXP-054)** = k=4 WideResNet-20 + RandomApply(AugMix w3, p=0.5) + GPU Cutout16 + cosine peak0.2/warmup0.05/Nesterov m0.9/LS0.1/WD1e-4 + compile(reduce-overhead), ~91 ep, dt 8ms. 68 experiments, 8 improvements.
- **The plateau is EXHAUSTIVELY mapped** (project-insights High; verbatim "96.45 appears at/near the k=4/300s ceiling"). Closures now include: augmentation (both delivery paths, coverage=50% a TRUE interior optimum); capacity ×4 directions; LR peak/shape/warmup; optimizer family/grad-dynamics(GC)/objective(SAM, PolyLoss)/AdamW/grad-clip; EMA/SWA weight-averaging; normalization-as-regularizer + eval-BN recalib + **BN momentum (EXP-067, last loop)**; **residual-branch scaling DOF (zero-init γ EXP-026, LayerScale EXP-051 — "depth-driven, inert on a 9-block net")**; head; batch; activation; label smoothing both sides; throughput→epochs both directions (single-shape-floor + multi-shape-penalty EXP-066).
- **Best recent near-misses** (all gradient/tail polish, improve loss not top-1): EXP-064 grad-clip 96.34, EXP-063 cooldown-on-AugMix 96.31, EXP-060 AutoAugment 96.22.
- **Genuinely UNTESTED feasible knobs remaining** (all low-ceiling): Lookahead (optimizer WRAPPER — mechanically distinct from the tested gradient-transform / objective / family changes, AND distinct from closed EMA/SWA in that slow weights feed BACK into the training trajectory, not just the eval point); BN eps (default 1e-5); SGD-momentum coefficient (0.9, never swept). Weight-averaging family (EXP-006/019/020) is closed — a headwind for Lookahead.

## Candidate Ideas

### 1. Lookahead optimizer wrapper (k=5, α=0.5) around the existing Nesterov SGD
**Summary**: Wrap the current SGD+Nesterov in Lookahead. Keep fast weights stepping normally each iteration; maintain a slow-weight copy of all params; every k=5 steps set `slow += α·(fast − slow)` (α=0.5) and copy `fast ← slow`. Eval uses the fast weights (current model handle) at the moment of evaluation, as today. Implement inline in train.py (~15 lines, no new dep): a dict of slow-weight tensors + a step-counter gate in the training loop after `optimizer.step()`.
**Reasoning**: The binding constraint is convergence/epochs (project-insights High). Lookahead reduces optimization variance and stabilizes the trajectory by periodically pulling the fast weights back toward a slow exponential-ish average, which can improve effective convergence per epoch at ~zero compute (the slow-weight interpolation is a cheap eager tensor op every 5th step, OUTSIDE the compiled forward → cudagraph-safe, like the EXP-064 grad-clip that ran clean). It is mechanically distinct from the tested optimizer changes (it is a meta-wrapper, not a gradient transform/objective/family swap) AND from the closed EMA/SWA eval-time averaging (Lookahead's slow weights re-enter TRAINING — the fast weights reset to slow every k steps — changing the optimization path, not just the eval point).
**Sources**: train.py optimizer setup (~L200) + training loop (`optimizer.step()` ~L246); Zhang et al. NeurIPS 2019; EXP-064 (eager per-step op between backward/step ran cudagraph-safe).
**Estimated Effort**: Low-moderate (inline slow-weight buffer dict + periodic interpolation; careful fast/slow bookkeeping under the time-fraction LR — LR drives the fast steps; slow update is LR-independent).
**Risk Assessment**: Moderate-low. Headwind: the weight-averaging family (EMA EXP-006, SWA EXP-019/020) is closed/near-miss here, and Lookahead's slow weights are a form of trajectory averaging → precedent leans null. Some implementation risk (correct bookkeeping; eval-at-fast-weights consistency). No wall/throughput/scope risk (eager op, train.py only, no new dep, params unchanged). Fails gracefully to no-improvement.

### 2. BN eps increase (1e-5 → 1e-3)
**Summary**: Set `eps=1e-3` on all four `nn.BatchNorm2d` sites. Single static-arg change.
**Reasoning**: Larger eps shrinks the normalized output of low-variance channels (divide by sqrt(var+eps)), a soft implicit down-weighting of less-informative channels — untested. The last clean static-knob on the BN-estimator axis (BN momentum closed EXP-067).
**Sources**: train.py BN sites (L71/75/83/103); brainstorm-067 Idea 2 (carried).
**Estimated Effort**: Trivial. Compute-neutral, single-graph, wall-safe.
**Risk Assessment**: Low, very-low-evidence. On well-activated k=4 channels eps 1e-5 vs 1e-3 is negligible for most channels → near-certain exact null.

### 3. SGD Nesterov momentum coefficient sweep (0.9 → 0.95)
**Summary**: Raise `MOMENTUM` 0.9 → 0.95 (more acceleration / longer gradient memory). Single scalar.
**Reasoning**: Higher momentum accelerates convergence in low-curvature directions — potentially valuable on an epoch-bound net — at zero compute. The momentum coefficient itself was never swept (only the optimizer FAMILY/grad-dynamics were).
**Sources**: train.py MOMENTUM (L25); standard SGD.
**Estimated Effort**: Trivial.
**Risk Assessment**: Low-moderate. 0.9 is a robustly-near-optimal default; 0.95 with PEAK_LR 0.2 raises the effective step and could mildly destabilize the warmup/early phase → near-null or small regression. Interacts with the tuned LR schedule (confound risk).

## Idea Evaluation
- **Evidence strength**: all three are low-evidence probes on an exhaustively-mapped plateau. Idea 1 (Lookahead) has the most concrete external grounding (a published method with reported CIFAR gains) AND the most genuinely-distinct mechanism (trajectory-level, not eval-level, averaging — the one untested optimizer class). Ideas 2/3 are trivial near-null scalars.
- **Mechanism clarity**: Idea 1 clear and distinct (fast/slow trajectory stabilization, slow weights re-enter training). Idea 2 real-but-negligible-magnitude. Idea 3 clear but confounded by the tuned LR schedule and contraindicated by the robustness of m=0.9.
- **Expected impact**: all near-noise on top-1, but Idea 1 has the only non-trivial (if still small) ceiling and the highest learning value (probes a distinct mechanism rather than re-confirming a scalar optimum).
- **Risk profile**: Ideas 2/3 lowest-risk (trivial), Idea 1 moderate (implementation) but still fails gracefully and has no wall/scope/dep risk.
- **Feasibility**: Ideas 2/3 trivial; Idea 1 low-moderate.
- **Conclusion**: Lead with **Idea 1 (Lookahead)** — per the "think harder / try a genuinely distinct mechanism" mandate, it is the most defensible real attempt: a published, mechanism-distinct optimizer wrapper targeting the binding constraint (convergence) at zero compute, untested here, that fails gracefully. BN-eps and momentum-sweep are trivial near-null fallbacks for later loops. Honest expectation: near-noise null (the closed weight-averaging precedent is a real headwind), run per NEVER-STOP.

## Chosen Idea
**Selected**: Lookahead optimizer wrapper (k=5, α=0.5) around the existing Nesterov SGD, implemented inline in train.py, all else byte-identical to EXP-054.

**Why this idea**: After 67 experiments every scalar/schedule/augmentation/capacity/normalization/init/optimizer-family lever is closed, and the last two loops were near-certain-null micro-probes. Lookahead is the most defensible remaining "real attempt": a published method with a genuinely distinct mechanism (fast/slow trajectory averaging that re-enters training, unlike the closed EMA/SWA eval-time averaging and unlike the tested gradient-transform/objective/family changes), it targets the binding constraint (convergence/epochs) at ~zero compute, it is cudagraph-safe and wall-safe (eager periodic op, like the clean EXP-064 grad-clip), and it fails gracefully. It honors the directive to try a distinct mechanism over re-running trivial scalar nulls.

**Hypothesis**: Wrapping Nesterov SGD in Lookahead (k=5, α=0.5) stabilizes the optimization trajectory and improves effective per-epoch convergence, raising best_test_acc to ≥ 96.55 (baseline 96.45 + 0.1pp). Given the closed EMA/SWA weight-averaging precedent on this net, the most likely outcome is a within-noise null or near-miss that maps the last untested optimizer-mechanism class — but Lookahead's trajectory-level (not eval-level) action is a genuine, untested distinction and the probe is compute-/wall-safe.
