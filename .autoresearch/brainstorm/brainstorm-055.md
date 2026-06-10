# Brainstorm EXP-055
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- **AugMix (Hendrycks et al., ICLR 2020)** (knowledge — torchvision `AugMix` reference): the canonical ablation sweeps `mixture_width` (number of parallel augmentation chains that are convex-combined). The paper's default/sweet-spot is **width=3**; gains are reported to *saturate* past 3 on clean accuracy, with most of the benefit on corruption robustness. Implication: pushing width 3→4 may be interior-optimal/null on clean CIFAR-10 top-1, the same shape as the EXP-053 severity null. Treated as a *moderate-prior-against* signal, not a disqualifier — our regime differs (subset application via RandomApply + GPU Cutout + 300s budget, not full-coverage AugMix).
- No new external search this loop — the lever (augmentation chain-count diversity) and its torchvision knobs are already well-characterized in goal-learnings; ideation draws on accumulated project history.

## Experimental History Review
- **The single validated lever (now 3× confirmed)**: augmentation chain-COUNT / DIVERSITY is the only thing that lifts top-1 here. EXP-012 TrivialAugmentWide +0.22 → EXP-052 AugMix(w2,d1) +0.12 → EXP-054 RandomApply(full AugMix w3, p=0.5) +0.11. Cumulative 96.00→96.45 (+0.45pp). Magnitude/severity is interior-optimal/null (EXP-053). [exp-index rows 012/052/053/054; goal-learnings § Patterns High-Importance]
- **Current best = 96.45** (EXP-054, commit 86161d9): `RandomApply([AugMix()], p=0.5)` — full default 3-chain AugMix on ~50% of images, the rest get only RandomCrop+Flip(+GPU Cutout). final_test_loss 0.1968 (loss AND top-1 both improved — signature of a real, if small, gain).
- **Two governing walls**: (a) the **600s wall-clock** is the binding constraint for CPU-heavy augmentation — budget gates on Σdt (300s GPU-step), so CPU aug doesn't cut epochs but balloons wall. Uniform full AugMix (w3) ≈ 792s → infeasible; RandomApply(p) is the lever to stay under 600s. (b) **polish-vs-top1 wall**: compute-neutral changes move loss not top-1 at fixed capacity.
- **Wall variance**: RandomApply's stochastic per-batch cost (variable # augmented samples) widens wall variance — EXP-054 projected 535s but came in at 593s (7s margin). Protocol finding: project the wall CONSERVATIVELY for stochastic aug. [goal-learnings § Protocol Findings]
- **Gaps not yet tried**: (1) per-augmented-image *richness* beyond w3 (more chains) — untested; (2) the p coverage dial at fixed w3 (only p=0.5 tried; p=0.4/0.6 unmapped); (3) replication of the 96.45 to confirm above the ±0.25pp noise band; (4) all-image coverage with a light else-branch (infeasible at full cost, EXP-054 unexplored avenue); (5) throughput recovery / GPU-side AugMix to afford uniform rich chains (high implementation risk).
- **Honesty flag**: the last three gains are each within the ±0.25pp run-to-run noise band; aggregate +0.45pp over three steps is the stronger evidence. Replication is increasingly valuable but cannot itself advance the metric.

## Candidate Ideas

### 1. Richer per-image chains on a subset — RandomApply([AugMix(mixture_width=4)], p≈0.4)
**Summary**: Replace the winning `RandomApply([AugMix()], p=0.5)` (w3 on ~50%) with `RandomApply([AugMix(mixture_width=4, chain_depth=-1)], p=0.4)` — deliver a RICHER 4-chain AugMix mixture to ~40% of images. This is the only untested direction of the validated chain-count lever: EXP-052/054 raised chain *count* via mixture_width 2→3 and via richer-on-subset; this raises it 3→4. Lower p (0.4 vs 0.5) pays for the heavier per-image op (8 ops avg vs 6) so the average dataloader cost stays under the wall. Probed dataloader cost (8 workers, this session): w4 p=0.4 = **12.2 ms/batch → est ~557s wall**; w4 p=0.35 = 11.4 ms → ~528s (safe fallback). dt (GPU step) stays 8ms — Σdt budget and epoch count unaffected. One-line change in `train_tf`; model/optimizer/schedule/seed/batch/compile unchanged; no new deps (torchvision-native).

**Reasoning**: Chain-count diversity is the *only* lever that has moved top-1 here, confirmed 3×. Width 3→4 is its direct, untested extension. The AugMix paper suggests clean-accuracy gains saturate past width 3 (prior-against), but our regime is materially different — AugMix is applied to a *subset* (not uniform), stacked with GPU Cutout, under a 300s compute budget where the model is mildly under-trained — so the saturation point need not match the paper's full-coverage setting. Graceful failure (within-noise null at worst).

**Sources**: AugMix ICLR 2020 (width ablation); exp-index rows 052/054; goal-learnings § Patterns (chain-count lever), § Protocol Findings (wall variance); this session's dataloader probe.

**Estimated Effort**: low (one-line transform change + a wall-feasibility gate already routine).

**Risk Assessment**: (a) **Most likely null** — paper-saturation prior says w4 may not beat w3 (interior-optimal, like the EXP-053 severity null). (b) **Confounded** — width↑ and coverage↓ (p 0.5→0.4) co-move; a null can't cleanly attribute. (c) Wall: w4 p=0.4 probed ~557s but RandomApply variance can add ~40-60s → could approach 600s; mitigated by an early real-load gate with a p=0.35 (~528s) fallback. Worst case: within-noise no-improvement, code reverts.

### 2. Map the coverage dial at fixed w3 — RandomApply([AugMix()], p=0.4 vs the p=0.5 baseline)
**Summary**: Hold the validated w3 config and test a single clean variable — coverage p. EXP-054 used p=0.5 (tight 593s wall); p=0.4 (probed ~517s, comfortable margin) trades 10% coverage for ~75s wall headroom. Single-variable, clean attribution, safest wall.

**Reasoning**: The coverage-vs-richness tradeoff is unmapped (EXP-054 unexplored avenue). If p=0.4 matches p=0.5's top-1, it's the more robust config (more wall margin under the 600s constraint); the beneficial direction (higher p=0.6) is wall-infeasible (~700s). Cleanest diagnostic of the lever.

**Sources**: EXP-054 report § Next Steps ("Map the p tradeoff"); this session's probe (p=0.4 ~517s).

**Estimated Effort**: low.

**Risk Assessment**: **Unlikely to BEAT 96.45** — lower coverage than the winner; the informative-but-beneficial direction (higher p) is wall-blocked. Best case a tie at safer wall (robustness win, not a metric win). Mostly diagnostic — poor fit for a NEVER-STOP loop whose objective is to advance the metric.

### 3. Replicate EXP-054 (confirm 96.45 above noise + wall reproducibility)
**Summary**: Re-run the exact winning config to check the 96.45 holds (three stacked near-noise gains) and that the 593s wall reproduces under the tight 600s margin.

**Reasoning**: Aggregate +0.45pp over three near-noise steps warrants a confirmation before further investment; checks the tight-wall reproducibility flagged in EXP-054.

**Sources**: EXP-054 report § Next Steps (replication, medium).

**Estimated Effort**: low.

**Risk Assessment**: **Cannot advance the metric by construction** (same config, seed 42 → near-deterministic modulo throughput-jitter epoch-count variance). Confirms foundation but burns a loop without a chance at the bar. Defer until a richness/coverage probe also stalls.

## Idea Evaluation
The goal of a NEVER-STOP loop is to advance the metric, which down-weights the two diagnostic candidates (2 and 3) that cannot, by construction, clear the bar: Candidate 2 tests a coverage direction (lower p) that is *a priori* unlikely to beat the winner (the beneficial higher-p direction is wall-blocked), and Candidate 3 is a pure replication. Both are worth doing eventually (robustness / confidence), but neither has a real shot at 96.55 this loop.

Candidate 1 is the only option with a credible path to *beating* 96.45: it pushes the single 3×-validated lever (chain-count diversity) into its one untested region (width > 3). Its evidence is the strongest available — three confirmations that *count* is the lever — tempered by a moderate paper-prior that clean-accuracy gains saturate past width 3. That prior is real but regime-specific (full-coverage AugMix), and our subset+Cutout+budget regime differs enough that the test is genuinely informative either way: a win extends the lever; a null tightens the "count saturates at 3 here too" boundary (complementing the EXP-053 magnitude null), which is itself a useful closing result. The mechanism is clear (more parallel chains → richer per-image mixture → more diversity), it's feasible and wall-gated with a probed fallback, and it fails gracefully. Expected impact is modest (likely near-noise, possibly null) but it is the highest-EV *metric-advancing* move on the board. The confound (width↑/coverage↓) is acceptable for a within-lever probe and documented for analysis.

## Chosen Idea
**Selected**: Candidate 1 — Richer per-image chains on a subset: `RandomApply([AugMix(mixture_width=4, chain_depth=-1)], p=0.4)`.

**Why this idea**: It is the only untested direction of the only lever that has ever moved top-1 here (chain-count diversity, confirmed 3×), and the only candidate with a real chance at the 96.55 bar. It is low-effort, wall-feasible (probed ~557s with a ~528s p=0.35 fallback), Σdt/epoch-neutral, and fails gracefully to a within-noise null that still sharpens the lever's boundary. The diagnostic alternatives (p-map, replication) cannot advance the metric and are deferred.

**Hypothesis**: Increasing AugMix mixture_width 3→4 on a ~40% subset (vs w3 on 50%) exposes training to a richer per-image chain mixture along the validated diversity lever. Prediction: dt steady ~8ms (Σdt budget and epoch count ≈ EXP-054's 91), wall < 600s; IF richer 4-chain diversity regularizes better than 3-chain at the cost of slightly lower coverage, best_test_acc ≥ 96.55 (bar = baseline 96.45 + 0.1). Null/saturation outcome (within ±0.25pp of 96.45) would mirror the EXP-053 magnitude null and bound the chain-count lever at width 3 in this regime.
