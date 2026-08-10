# EXP-006 Adversarial Idea Review

**Reviewer role**: independent skeptic. **Baseline**: 92.30% (EXP-004, commit `11f8469`). **Threshold**: ≥92.40% (+0.10 pts = 10 test images). **Budget**: 300 s counted training, 600 s wall, single H20, seed 42, no reroll, `train.py`-only.

Candidates (labels from proposals dir):
- **idea-01** — Same-width full-preactivation ResNet-20 (architecture / representation-flow lever)
- **idea-02** — RandAugment magnitude 7→8 (augmentation-strength lever)
- **idea-03** — Plateau-only fixed 16×16 Cutout replacing RandAugment (augmentation-type lever)

I read all three proposals, the brainstorm, goal definition, learnings, results TSV, and `train.py`.

---

## Prioritized feedback (most important first)

### 1. idea-01's own cited source argues against a positive result at this depth — the strongest single weakness in the field
The mechanistic case rests on He et al. (Identity Mappings), but that paper's gains are demonstrated at ResNet-110/164/1001, and the proposal itself quotes the authors' caveat that "post-addition truncation is less severe at shallower depth." ResNet-20 has nine residual units. This is a textbook **gap between the claimed mechanism and the evidence**: the citation actively predicts a near-null effect at depth 20, and reproductions generally show preact ≈ post-act ResNet-20 within noise. The honest expected outcome is a result inside single-seed noise of 92.30 → likely no-improvement.
- **Refinement**: don't oversell the primary hypothesis. Either (a) re-frame the win condition around the *interaction* — the specific claim that clean identity paths help *under the strong-view plateau* (the only novel, untested part, not covered by the shallow-depth caveat) — and pre-register the plateau-phase train-loss/checkpoint trajectory as mechanism evidence; or (b) demote this idea until a lever with higher expected effect is exhausted. Parameter parity (verified: BN redistributes 192+352+704+128 = 1,376, total stays 269,722) and throughput parity are genuine strengths, but clean attribution of a null is low-value for the metric.

### 2. idea-03 replaces a *proven* regularizer on a bet its own sources do not support head-to-head
EXP-004 established RandAugment N1/M7 is worth +0.47 here. idea-03 discards it entirely for Cutout. The cited papers (DeVries & Taylor; Zhong et al.) support that Cutout is a strong CIFAR regularizer **additive to standard crop/flip over long (≥200-epoch) schedules** — they do **not** compare Cutout against RandAugment, and they do not run under a 300 s / ~80-plateau-epoch budget. So the central claim "occlusion is a *better* strong-view family than N1/M7 for this model/budget" has no direct evidentiary support; the proposal concedes this in its "Primary Evidence" section. This is the highest-ceiling idea but also the one most exposed to regression.
- **Refinement**: acceptable as a deliberately controlled category test *if* framed honestly — but tighten the a-priori justification for the specific strength. `p=1.0` + 16×16 masks **25% of every image on every plateau step** for the whole 240 s high-LR phase, materially more aggressive relative to the *shortened* budget than canonical Cutout's long schedule. Under-fitting is a real risk the proposal underweights. Pre-register the plateau train-loss EMA and last strong-phase checkpoint as an under/over-fit diagnostic (the proposal lists this only as an "intermediate signal," not a decision input). If you cannot tune within the experiment (correct, per protocol), justify 16×16 vs a smaller mask up front rather than defaulting to the canonical value from a longer-schedule paper.

### 3. idea-02's ceiling barely clears threshold and its direction is unknown — attribution strength ≠ metric upside
The paired-RNG argument is **correct and verified**: torchvision RandAugment uses a fixed `magnitude` (no extra RNG draw), so op identity/sign draws stay aligned between M7 and M8 within a worker stream — this genuinely is the most trustworthy marginal measurement of the three. But the proposal's own table shows ~4/14 operations are unchanged between bins and the rest shift by tiny amounts (rotation +1°, translation +0.48 px). Expected effect is near-zero and **the sign is unargued**: EXP-004 proved M7 > nothing, *not* that stronger helps. The dropped "M6" candidate in the brainstorm concedes M7 could equally be too strong. A clean measurement of a null does not advance `best_test_acc`.
- **Refinement**: give an actual directional argument (why M8 > M7 specifically, not just "M7 may not be at the boundary," which is symmetric with "M7 may already be past it"). Without one, this is a coin-flip at threshold. Also: the preflight gate (140 b/s) is inexplicably *below* EXP-004's observed 165 b/s and idea-03's 160 b/s — M8 is the same cost as M7, so 140 is a loose, unmotivated floor.

### 4. Single-seed noise vs a 10-image threshold undercuts every non-idea-02 marginal pass
The goal's own learnings note effect sizes are "entangled with augmentation RNG draws under the fixed single-seed protocol." idea-01 (architecture) and idea-03 (full augmentation swap) both fully de-correlate the RNG/optimization trajectory from EXP-004, so a landing at exactly 92.40 could be noise. Inherent to the no-reroll hard constraint and **not disqualifying**, but all three should state plainly that a marginal pass (92.40–92.45) is not a reliable improvement claim. idea-02 is the only candidate partially insulated from this — its real but narrow virtue.

### 5. No fatal / scope / already-failed violations found
None of the three retries the EXP-005 failure (75% switch) — all hold the strong/LR boundary at 80%, consistent with the learnings. All are `train.py`-only, preserve evaluator/seed/budget, and are feasible within 600 s. No reward-hacking or seed-hacking. idea-01's param-parity math is correct; idea-03's `inplace=True` safety argument is correct (ToTensor allocates a fresh per-sample tensor, so mutation cannot corrupt the shared CIFAR array or persistent workers) and its `value=0`→mean-fill claim is correct given `std=(1,1,1)` at `train.py:149`. idea-03 must rename `randaugment_enabled` in **both** predicates (`train.py:250-254` and `:281`), not just one.

---

## Scored verdict

Scale 1–10 per axis.

| Idea | Evidence & reasoning | Potential impact | One-line reason |
|---|---|---|---|
| **idea-03 Cutout** | 6 | 8 | Mechanism (part-distributed occlusion) is well-established on CIFAR and composes with EXP-004's proven strong-explore/weak-refine structure; highest ceiling (~92.70), but "beats RandAugment here" is unsupported and it forfeits a proven win. |
| **idea-01 Preact** | 5 | 5 | Cleanest attribution (exact param/throughput parity) and the only untested *lever class*, but its own citation predicts a near-null at depth 20 — low probability of moving top-1. |
| **idea-02 M8** | 6 | 3 | Best-attributed (verified paired-RNG) and lowest-risk, but expected effect ~0 with unargued sign and a ceiling that barely reaches threshold. |

### Pick: **idea-03 — Plateau-Only 16×16 Cutout**

**Why it wins.** On the rubric's two axes together, idea-03 offers the best product of a *sound, source-backed mechanism* and a *high ceiling*. Cutout is one of the most effective single regularizers for small CIFAR models, and crucially it does not fight EXP-004's real innovation — it plugs a genuinely different invariance (occlusion / part-distributed evidence) into the same validated strong-explore→weak-refine→BN-resettle scaffold. That is exactly the orthogonal, high-upside bet the goal rewards over a safe incremental one.

It wins over **idea-01** because idea-01's mechanistic argument is undercut by its own cited evidence at ResNet-20 depth — its most likely outcome is a within-noise null, so its clean attribution buys a low-value measurement. It wins over **idea-02** because idea-02's upside is too small to matter: even a perfectly-attributed one-bin change is expected to land at or just under threshold with an unknown sign, and the goal is to maximize accuracy, not to measure a marginal augmentation effect cleanly.

**Required refinements before planning (idea-03):**
1. **Reframe the win condition and register the under-fit diagnostic as a decision input, not a footnote.** Pre-commit to logging the plateau train-loss EMA and the last strong-phase checkpoint; a materially higher plateau loss than EXP-004's trajectory with no test gain is the expected failure mode of 25%-every-image occlusion under a 300 s budget — treat it as diagnostic, not incidental.
2. **Justify 16×16 / p=1.0 a-priori against the shortened schedule.** The canonical value comes from ≥200-epoch training; state explicitly why full-strength occlusion for the entire ~80-epoch plateau is expected to help rather than starve fitting here, or accept it as the single pre-registered strength with a documented under-fit risk (no in-experiment tuning, per protocol).
3. **State the honest comparison.** The verdict tests "Cutout ≥ RandAugment for this model/budget," a claim with no direct source support; a no-improvement result is *informative* (RandAugment's broad invariances win) and must not be read as Cutout failing in general.
4. **Mechanical correctness.** Rename `randaugment_enabled`→`cutout_enabled` in both the epoch-break predicate (`train.py:250-254`) and the switch block (`:281`); keep the `crop→flip→ToTensor→Normalize→RandomErasing` order exactly (erasing must follow normalize so `value=0` = mean-fill); confirm the switch log label change and one clean forkserver teardown.
5. **Single-seed caveat.** Record that a marginal pass (92.40–92.45) is not a reliable improvement under the no-reroll protocol.

**Note for the deciding agent:** idea-01 is the reasonable fallback *if* priority shifts from "chase the highest ceiling" to "open a new orthogonal lever class with airtight attribution" — but only after its hypothesis is re-scoped around the plateau interaction rather than generic preactivation, since the generic version is predicted-null at this depth. idea-02 should not be run as a standalone experiment: its expected upside does not justify a full run; fold magnitude direction into a future sweep instead.
