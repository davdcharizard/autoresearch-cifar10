# Brainstorm EXP-069
**Created**: 2026-06-10
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- AugMix (Hendrycks et al., ICLR 2020) — knowledge: torchvision `transforms.AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0, all_ops=True, ...)`. The `alpha` hyperparameter is the concentration of BOTH probability distributions AugMix samples: the **Dirichlet(alpha,…,alpha)** weights that convex-combine the `mixture_width` augmentation chains, AND the **Beta(alpha, alpha)** weight `m` that mixes the combined-augmented image back with the ORIGINAL clean image (`out = m·clean + (1−m)·chain_mix`). Key property: alpha is a **mean-preserving variance knob** — at any alpha the Dirichlet/Beta means are unchanged (uniform chain weights; m mean 0.5), only the *variance* changes. alpha=1 ⇒ Beta(1,1)=Uniform[0,1] (high-variance m, often near 0 or 1) and Dirichlet(1,1,1)=uniform-simplex (often one chain dominates). alpha>1 ⇒ both distributions concentrate toward their means (consistent ~50/50 clean-mix; all 3 chains contribute ~equally every image). alpha<1 ⇒ both become more bimodal/peaked (images either near-clean OR dominated by a single chain). The AugMix paper credits its clean-corruption-robustness gains to the *multi-chain Dirichlet mixing + clean convex-mix* STRUCTURE — which higher alpha makes more faithful per-image.

## Experimental History Review

Current best = **EXP-054 = 96.45** (`RandomApply([AugMix() w3,d-1], p=0.5)` + GPU Cutout16). Bar = 96.55. **14 consecutive no-improvements since EXP-054.**

What worked (the ONLY lever that ever lifted top-1, 3× confirmed — project-insights High, line 295): augmentation **chain-COUNT diversity** — Cutout→TrivialAugment (EXP-012 +0.22)→AugMix w2,d1 (EXP-052 +0.12)→full-AugMix-w3-on-50%-subset (EXP-054 +0.11). Cumulative 96.00→96.45.

Augmentation sub-levers explicitly CLOSED (do-not-retry):
- **Magnitude/severity** 3→6 (EXP-053, interior-optimal) — line 139-141
- **Chain-count WIDTH** >3 (EXP-055 w4, regressed) — line 142-144
- **Coverage** <50% (EXP-055 p0.35) AND =100% (EXP-057 GPU full) — 50% is a TRUE interior optimum — line 148-150
- **GPU delivery** path (EXP-056/057/059, epoch-disadvantaged) — line 151-153
- **Policy family** (TA/RA/AutoAugment all ≈96.2 < AugMix; EXP-012/014/060) — line 66-68
- **Label-mixing** (Mixup/CutMix, underfit at budget; EXP-011/018) — line 79-85
- **Cooldown** (EXP-033/034/035/063, never clears bar) — line 182-185
- **Border-quality / occlusion-pattern** (EXP-037/048) — line 160-162, 106

Non-augmentation axes ALL closed: optimizer-family (AdamW EXP-043), grad-dynamics (GC EXP-031, clip EXP-064), objective (SAM EXP-036, PolyLoss EXP-041), schedule (peak-LR EXP-016/017, warmup EXP-062, SGDR EXP-029, cooldown), capacity (k=5/6 EXP-004/009, depth EXP-044, shallow-wide EXP-058, fat-head EXP-038), normalization (GhostBN EXP-047, BN-momentum EXP-067, clean-BN EXP-061), weight-averaging (EMA/SWA/Lookahead EXP-006/019/020/068), head (multi-scale EXP-032, cosine EXP-039), regularizers (WD EXP-005, LS EXP-023/065, dropout EXP-022, LayerScale EXP-051), batch (64/256 EXP-050/025), init/micro-arch (pre-act EXP-015, zero-gamma EXP-026, ResNet-D EXP-027, SE EXP-008, deep-sup EXP-042, BlurPool EXP-024).

**Genuinely UNTESTED gaps** (not in any closed list): the AugMix INTERNAL distribution parameters — **`alpha`** (Dirichlet/Beta mixing concentration) and **`chain_depth`** (ops composed per chain). The closures mapped chain-COUNT (width) and MAGNITUDE (severity), but never the mixing-weight DISTRIBUTION (alpha) or compositional DEPTH (chain_depth). Line 297 explicitly lists "p-sweep (0.4/0.6)" as literally-untried, though coverage is otherwise declared interior-optimal.

## Candidate Ideas

### 1. AugMix mixing-concentration alpha 1.0 → 2.0
**Summary**: Change the single AugMix constructor arg: `transforms.AugMix()` → `transforms.AugMix(alpha=2.0)`. All else byte-identical to EXP-054 (w3, chain_depth=-1, severity=3, p=0.5 RandomApply, GPU Cutout16, full recipe). This concentrates the Dirichlet chain-weights and the Beta clean-mix toward their means: every augmented image becomes a consistent ~50/50 clean-mix that genuinely blends all three chains (instead of the alpha=1 regime where one chain often dominates and `m` is frequently near 0 or 1). It amplifies the exact "multi-chain Dirichlet mixing + clean convex-mix" STRUCTURE that project-insights line 68/295 credits for AugMix's win, at **zero op-count change → wall-neutral** (the binding constraint at this recipe; EXP-054 is 593s/7s-margin).

**Reasoning**: Augmentation chain-COUNT diversity is the ONLY documented top-1 lever (3× confirmed, High importance). alpha is the one remaining *wall-safe* sub-knob of that lever never mapped — width (count) and severity (magnitude) are closed, but the mixing-weight distribution is orthogonal to both. Because alpha is **mean-preserving** (average augmentation strength is identical to alpha=1), it cannot over- or under-regularize on average — it only changes the *shape* of the per-image mix, so it sidesteps the coverage/strength failure modes that sank EXP-053/055/057. Higher alpha makes the Dirichlet less likely to collapse to single-chain → more faithful 3-chain blending per image → more of the credited structure, while Beta(2,2) keeping `m`≈0.5 preserves the full 50% effective coverage (no drop toward the EXP-055/057 over-/under-regularization cliffs).

**Sources**: knowledge: torchvision AugMix signature; project-insights line 68 ("AugMix wins via STRUCTURE — multi-chain Dirichlet mixing + clean convex-mix"), line 295 (diversity = only lever), line 297 (untried internals); EXP-052/053/054/055/057 reports.

**Estimated Effort**: low (one constructor kwarg; single-variable; ~590s run).

**Risk Assessment**: Most-likely outcome is a within-noise null (mean-preserving variance change is subtle; the augmentation peak is well-characterized). Lower-variance mixing could mildly *under*-augment (averaging 3 chains partially cancels distortions → softer net image), behaving like a gentle severity reduction → small regression. Worst case ≈ −0.3pp (scalar-knob band). Wall-safe (no op-count change), throughput-neutral (CPU dataloader change, dt unaffected, params unchanged) — clean no-caveat run expected.

### 2. AugMix coverage p 0.5 → 0.6 (the literally-"untried p-sweep")
**Summary**: Raise the RandomApply probability `RandomApply([AugMix()], p=0.5)` → `p=0.6`: full 3-chain AugMix on ~60% of images instead of 50%. The one coverage cell between the tested 50% (EXP-054, best) and 100% (EXP-057, over-regularized).

**Reasoning**: Line 297 explicitly flags "p-sweep (0.4/0.6)" as untried. If 50% is only a *local* plateau rather than a sharp optimum, 60% might edge higher.

**Sources**: project-insights line 297; EXP-054 (p0.5), EXP-055 (p0.35 ✗), EXP-057 (p1.0 ✗).

**Estimated Effort**: low (one literal change).

**Risk Assessment**: **Doubly contraindicated.** (a) Wall: AugMix CPU cost scales ~linearly with coverage (p1.0≈792s, p0.5≈593s ⇒ slope ~390s/unit-p); p0.6 adds ~+39s → ~632s, a near-certain >600s wall breach (worse than EXP-066's 617s). (b) Evidence: coverage→100% over-regularizes (EXP-057 −0.81pp), and 50% is declared a TRUE interior optimum (EXP-055/057) — 60% moves toward the failing direction. Reject in favor of Idea 1 (wall-safe, evidence-aligned).

### 3. AugMix chain_depth -1 → 3 (fixed deep 3-op chains)
**Summary**: Fix the per-chain composition depth `transforms.AugMix(chain_depth=3)` (default −1 = random 1–3 per chain). Every chain becomes a 3-op composition → richer compositional diversity per augmented image.

**Reasoning**: chain_depth (compositional DEPTH) is distinct from width (chain COUNT, closed) and severity (per-op MAGNITUDE, closed) — a genuinely untested diversity sub-axis. Deeper chains = more distinct augmented appearances = the winning diversity lever.

**Sources**: torchvision AugMix signature; project-insights line 139-144 (width/magnitude closed, depth not mapped).

**Estimated Effort**: low (one kwarg).

**Risk Assessment**: **Wall-infeasible.** Fixing depth −1→3 raises mean ops/chain from ~2 to 3 (+50% ops) on the 50% augmented subset; AugMix is the binding wall bottleneck (lines 32-33), so +50% aug ops → est. +40–60s → ~640s, a clear >600s breach. Depth-DOWN (fix to 1) is wall-cheap but reduces toward single-op TrivialAugment (96.22, worse). Reject on wall grounds in favor of Idea 1.

## Idea Evaluation

All three stay on the only productive lever (augmentation diversity), which is correct given every non-aug axis is closed and the goal-learnings' "pivot off augmentation" guidance has nowhere open to pivot TO. The decisive discriminator is the **600s wall**, the recurring breach failure (EXP-061/065/066) on this 7s-margin recipe.

- **Idea 2 (p0.6)** and **Idea 3 (chain_depth=3)** both INCREASE AugMix CPU work → near-certain wall breach (~632s / ~640s), and both push toward directions with *negative* prior evidence (coverage>50% over-regularizes EXP-057; depth-up is a magnitude-like strength increase the severity closure EXP-053 argues against). A breach with a trustworthy metric is classified no-improvement (informative, not invalid) per EXP-061/065/066 — but choosing a probe that is *expected* to breach AND is evidence-contraindicated is poor EV.
- **Idea 1 (alpha=2.0)** is the only **wall-neutral** option (no op-count change — alpha just re-samples mixing weights), it is **mean-preserving** (cannot over/under-regularize on average, sidestepping the strength/coverage failure modes), it is **single-variable**, and it amplifies the *specific* multi-chain-Dirichlet-mixing structure that project-insights explicitly credits for AugMix's win. Mechanism clarity and risk profile both strongly favor it.

Evidence strength is comparable across all three (all are untested internals of the documented winning lever), but Idea 1 dominates on feasibility (wall-safe), mechanism alignment (amplifies the credited structure), and safety (mean-preserving, no-caveat clean run). Honest expected impact for ALL three is modest (the augmentation peak is well-characterized; most-likely a within-noise null) — but Idea 1 is the highest-EV remaining wall-safe probe on the only axis that has ever moved this metric.

## Chosen Idea
**Selected**: Idea 1 — AugMix mixing-concentration alpha 1.0 → 2.0

**Why this idea**:
It is the single genuinely-untested, **wall-safe**, single-variable knob on augmentation diversity — the only lever documented (3×, High importance) to lift top-1 on this saturated recipe. The width/magnitude/coverage closures never touched the Dirichlet/Beta mixing-concentration `alpha`. Raising alpha to 2.0 concentrates the chain-weights and clean-mix toward their means, making every augmented image a faithful ~50/50 blend of all three chains — amplifying the exact "multi-chain Dirichlet mixing" STRUCTURE that project-insights line 68 credits for AugMix's win — at **zero wall cost** (alpha changes only the sampling of mixing weights, not the op count), so it avoids the recurring >600s breach that contraindicates Ideas 2/3. Because alpha is mean-preserving, it cannot over- or under-regularize on average, sidestepping the strength/coverage failure modes (EXP-053/055/057). It is the highest-EV wall-safe move that remains.

**Hypothesis**:
Concentrating the AugMix mixing distributions (alpha 1.0→2.0) so that every augmented image consistently blends all three augmentation chains with the clean image (rather than the alpha=1 regime where one chain often dominates and the clean-mix weight is frequently extreme) will increase the *effective* multi-chain diversity the model sees per image without changing average augmentation strength or coverage, raising best_test_acc to ≥ 96.55. The stated most-likely alternative: a within-noise null (96.2–96.45), since the augmentation peak is well-characterized and alpha is mean-preserving — in which case the AugMix mixing-distribution sub-axis is closed and the augmentation lever is exhausted on its internal-distribution dimension too.
