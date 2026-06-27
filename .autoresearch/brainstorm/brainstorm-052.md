# Brainstorm EXP-052
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new external fetches; sources are the project record + model knowledge:

- **reports/exp-report-046.md (the anchor)**: anti-aliased shortcut downsampling (F.avg_pool2d replacing the pad shortcut's [::2,::2] strided slice) read **96.65 = mean + 0.5σ** — the HIGHEST read among all toll-free changes in the project's history — at perfectly clean signatures (dt family, params unchanged at 4,286,026? — verify: report says signature-clean; params actually unchanged since avg_pool has no params), family test_loss, branch (ii) absorbed-null. The external anchor was Zhang 2019 "Making Convolutional Networks Shift-Invariant Again" (BlurPool): anti-aliasing before subsampling reliably adds +0.5–1.0pp on ImageNet classification.
- **Statistical resolution limit (project's own arithmetic)**: σ ≈ 0.16 (EXP-027); a single draw cannot distinguish a true effect of 0 from +0.2 (both produce 96.65 with ordinary probability). The detection law (effect ≥ +0.3 for one-draw detection) is exactly why a true +0.1–0.2 BlurPool-class effect would have registered as an absorbed null. A pre-registered TWO-run replicate decided on the MEAN halves the noise (σ_mean ≈ 0.113): under H0 (no effect), P(mean ≥ 96.81) ≈ 1.6% — an honest test, NOT max-statistic scatter farming (the decision statistic is the mean; the max is never used).
- **Selection-effect honesty**: re-testing the best of ~45 nulls is post-hoc selection, which lowers the PRIOR (some +0.5σ reads are expected under pure noise) but does not invalidate the TEST (false-positive rate stays 1.6%). This is recorded so the analysis cannot quietly inflate the prior.

## Experimental History Review

State after 52 indexed experiments: baseline 96.71 @ 1990397, bar ≥ 96.81; mean ≈ 96.57, σ ≈ 0.16; 45 consecutive non-improvements. Frontier after EXP-051:

- **Every constructible single-mechanism class is measured-closed**: recipe constants bracketed/flat (049 completed the audit); LOSS AXIS fully closed both directions (050 margin −2.4σ with improved CE; 051 confidence-weighting −7.8σ with degraded CE — plain mean CE+LS sits between two measured cliffs); structural triangulated closed (030/046/047); throughput exhausted (048, 99.3% kernel math); gradient noise at measured maximum (count-5 law); regularizer stack peaked (009); weight/function averaging closed (011/032/043); init, BN constants, activations, optimizer all closed.
- **The ONLY positive-direction unresolved datum**: EXP-046's 96.65 (+0.5σ) single draw. Every other read of the last 20 loops is at/below mean. The absorbed-null classification was correct protocol for n=1 — but n=1 cannot resolve a +0.1–0.2 true effect, and BlurPool's external record (+0.5–1.0 on ImageNet) makes a small surviving effect physically plausible even under absorption (it removes aliasing artifacts — information corruption — rather than adding invariance pressure that aug duplicates).
- **This session's sweep of alternatives** (recorded so they are not re-derived): warmup-region real-data training = timer manipulation, FORBIDDEN; per-channel input std normalization ≈ absorbed by bn1 (linear rescale before BN); gradient clipping = inactive in stable training (weak mechanism); cosine-shape micro-variants = heat-distribution axis with cosine at measured optimum vs linear; batch < 512 = noise+throughput double-priced; TA-Wide→TA dose-down = inside the measured ±0.17 flat region; stem 5×5 = structural class closed; SGDW decoupling = momentum-smoothing micro-difference, sub-σ by construction.
- Protocol carry-overs: composite gates (26ms), step ledger, trajectory numerics, detection arithmetic.

## Candidate Ideas

### 1. Pre-registered replicate-pair resolution of EXP-046 (anti-aliased shortcut; n=2, decision on the MEAN)
**Summary**: Re-apply EXP-046's exact change (shortcut `[::2,::2]` strided slice → `F.avg_pool2d(x, 2)` at both stage transitions; byte-identical to the EXP-046 diff). Run TWO byte-identical gated composite runs. Pre-registered decision statistic: the MEAN of the two best_test_acc reads. Improvement iff mean ≥ 96.81 (then commit; the TSV metric records the mean). The max of the pair is never used for any decision.

**Reasoning**: This is the only candidate left holding a positive-direction measurement. Mechanism (unchanged from 046, externally validated): the strided-slice shortcut subsamples without low-pass filtering, aliasing high-frequency content into the residual stream at both stage transitions; avg_pool removes that corruption. Absorption explains why a +0.5–1.0 ImageNet-scale effect could shrink to +0.1–0.2 here (aug already teaches some shift-robustness) — but a +0.1–0.2 survivor is exactly what n=1 CANNOT distinguish from zero and what the 96.65 read weakly suggests. Test honesty: mean-of-2 with pre-registered threshold has a 1.6% false-positive rate under H0; this is sharper, not looser, than the standard single-run protocol (≈6.7% under H0 at the same bar). Selection-effect discipline: the prior is modest BECAUSE this is the best-of-many-nulls; the brainstorm records ~20–30% as the honest prior of a true +0.1–0.2 effect (BlurPool external record + clean mechanism vs noise expectation of some +0.5σ reads among 45 nulls).

**Sources**: reports/exp-report-046.md (diff, signatures, branch registration); goal-learnings EXP-046 entry; knowledge: Zhang 2019 BlurPool (cited in brainstorm-046); EXP-027 σ estimate.

**Estimated Effort**: low — re-apply a known diff (from exp-report-046/plan-046), re-run the known sanity script pattern, two sequential gated runs (~17 min wall).

**Risk Assessment**: Failure modes all graceful: (a) mean ≤ 96.60 → 046 confirmed null at n=3 total draws; permanently closed (further sampling would be variance mining); (b) mean ∈ [96.61, 96.80] → weak-positive but sub-bar: recorded as unresolved-and-closed (at σ=0.16, resolving +0.1 vs 0 needs n≈20 — not affordable; the bar decides); (c) mean ≥ 96.81 → improvement, commit. Justification for retrying a Failed-Approaches entry (count 1, Low importance): the entry's verdict was correct AT n=1; this is a measurement-resolution follow-up with a strictly more stringent decision statistic, not a re-roll.

### 2. Gradient clipping (global-norm 5.0) — runner-up
**Summary**: `torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)` before optimizer.step().

**Reasoning (and why not the lead)**: The only never-probed optimizer-side knob. But the mechanism is conditional: clipping only acts when gradients spike, and this training shows zero instability (no divergences in 40+ clean runs; loss curves smooth). At threshold 5.0 it would likely activate on <1% of steps — a sub-σ perturbation by construction. At aggressive thresholds (1.0) it becomes a heat reduction — the heat axis is closed flat-below (049), pricing that direction at zero-to-negative. No published anchor for clipping gains on stable CIFAR CNN training.

**Sources**: train.py loop structure; goal-learnings heat entries (EXP-010/014/049).

**Estimated Effort**: trivial.

**Risk Assessment**: Almost certainly an expensive coin-flip on noise; dominated by Idea 1's measured datum.

### 3. Real-data training during compile warmup — documented rejection (FORBIDDEN)
**Summary**: The 3 warmup iterations run forwards/backwards on random noise without optimizer.step(); stepping on real batches there would add free uncharged training.

**Reasoning (and why rejected)**: This is timer manipulation — moving training work outside the charged window violates the goal's charged-time semantics exactly as EXP-006's design notes anticipated ("No optimizer.step() -> weights unchanged"). Any gain would be reward hacking, classifiable invalid. Recorded so it is never re-derived.

**Sources**: goal hard constraints (timer semantics untouchable); train.py warmup comment L183–184.

**Estimated Effort**: N/A — rejected.

**Risk Assessment**: Integrity-failing by construction.

## Idea Evaluation

- **Evidence strength**: Idea 1 holds the project's only positive-direction measurement plus a strong external anchor (BlurPool). Idea 2 has no anchor and a self-limiting mechanism. Idea 3 is forbidden.
- **Mechanism clarity**: Idea 1's mechanism (aliasing removal at stage transitions) is exact, externally replicated, and orthogonal to augmentation's input-space action — the one absorbed-null where the absorption argument itself predicted only PARTIAL absorption. Idea 2's mechanism is vacuous on stable training.
- **Expected impact**: honest numbers — P(improvement) ≈ prior(0.2–0.3) × P(mean ≥ 96.81 | Δ=+0.15) ≈ 0.25 × 0.29 ≈ 7%, vs ≤2% for any law-priced alternative; and EVERY branch is terminal (null → closed at n=3; weak → closed by resolution limit; pass → new baseline). After 45 consecutive nulls, terminal information per run is the right currency.
- **Risk profile**: known diff, known signatures, graceful failure, no new code surface beyond a validated change.
- **Feasibility**: lowest of any candidate — a re-application plus one extra run.

Idea 1 dominates. Idea 2 recorded as the last unprobed knob (weak); Idea 3 recorded as forbidden.

## Chosen Idea
**Selected**: Idea 1 — Replicate-pair resolution of the anti-aliased shortcut (n=2, mean-decision, threshold 96.81)

**Why this idea**:
It is the only remaining candidate backed by a positive-direction measurement, carries the strongest external anchor of any absorbed-null (BlurPool), and its test design is strictly more stringent than the standard protocol (mean-of-2 at the same bar: 1.6% false-positive vs 6.7%). All three branches close permanently — this either finds the last small true effect on the board or retires it with adequate statistics.

**Hypothesis**:
The anti-aliased shortcut carries a true +0.1–0.2 effect partially surviving augmentation absorption (aliasing removal is information-repair, not invariance duplication): the mean of two fresh byte-identical runs ≥ 96.81 if the true effect is ≥ +0.2; pre-registered branches: (i) mean ≥ 96.81 → improvement, commit the shortcut change; (ii) mean ∈ [96.61, 96.80] → weak-positive unresolved at affordable n; 046 closed permanently (resolution limit: distinguishing +0.1 from 0 needs n≈20); (iii) mean ≤ 96.60 → confirmed null at n=3 total; closed permanently; (iv) GATE_KILL/contention on either run → infra relaunch per standard screens (the change was already measured signature-clean in EXP-046).
