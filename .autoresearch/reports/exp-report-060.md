# Report EXP-060: CutMix substituted for RandomErasing at matched dose (p=0.5, α=1.0)
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-060.md
- **Plan**: plans/plan-060.md
- **Log**: logs/exp-log-060.md

## Goal

Maximize CIFAR-10 best_test_acc (%, higher is better) within the fixed 300s charged budget by modifying train.py only. Baseline: 96.71 @ 1990397; bar ≥ 96.81 (= family mean 96.57 + 1.5σ, σ ≈ 0.16). Specific question: with the documented frontier empty after EXP-059, does the last open seam in the recipe — occlusion regularizer TYPE at constant dose — carry accuracy? The dose-response closure (EXP-003/004/009/013/015) measured every point with occlusion-as-noise-fill; this was the first measurement of occlusion-as-signal-fill.

## Idea & Hypothesis

Chosen from brainstorm-060: **CutMix (Yun et al., ICCV 2019, arXiv 1905.04899) replacing RandomErasing at the same application dose p=0.5, α=1.0**. CutMix fills the occluded box with the same-position patch from another batch image and supervises with the area-weighted mixed loss — the occlusion carries in-domain signal and the label faithfully describes the mixed content. Published +0.5–1.0 over Cutout/erasing on CIFAR ResNets (crop+flip, fixed-epoch) — the largest published prior of any untested legal construction, making this simultaneously the strongest available falsification probe of the measured-ceiling hypothesis. EXP-009's mixup negative (−0.46) did not cover it: that was global alpha-blending STACKED as a 4th regularizer; this is local patching SUBSTITUTED as the 3rd. Runner-up candidates both screened out: label-error excision (CIFAR-10 only ~0.54% noisy per Northcutt NeurIPS 2021 → upside below the +0.3pp detection threshold) and aug-strength warmup (EXP-031's transit-only lesson).

## Approach

Four hunks to train.py: constants `CUTMIX_P=0.5` / `CUTMIX_ALPHA=1.0`; RandomErasing removed from train_tf; sync-free eager CutMix block in the charged loop (CPU-RNG branch decision and box math in Python ints — zero GPU→CPU syncs; GPU work = one randperm + one in-place strided slice copy + one targets gather); area-weighted dual CE on the SAME logits inside the existing autocast. Key implementation insight: only `model` is compiled in train.py, so mixing and loss are outside the graph — standard slice-assignment CutMix is compile-inert (no graph variants, no warmup changes, no dynamic-shape exposure). CPU sanity all-pass; single-variable discipline held (dose, count, heat, noise, schedule, numerics, batch all constant).

## Execution

One GPU probe, one clean run, zero retries:
- Probe (load 8.3, first attempt clean): P_plain = 21.66ms (family anchor PASS), P_cutmix = 21.77ms forced-on every step → worst-case toll 0.11ms, ≈0.05ms at p=0.5 duty ≈ −0.01pp by the deferral law. Launch branch P ≤ 23.5 with unshifted family bands.
- Run 1: gates cleared on poll 1; D0 22.7; all 26 watchdog windows 22.0–22.8ms, slow_streak 0; RC=0 at 453.0s total (300.0 charged, startup 9.4s). Ledger: 13,285 steps ∈ [13,100, 13,600], 137 epochs, params 4,286,026, 137 evals. ep1 36.26 — the relaxed (≥25) tripwire was unnecessary; mixed-label supervision did not slow the early read.

## Results

- **Primary metric**: best_test_acc 96.69 @ ep133 (baseline: 96.71, delta: −0.02, −0.02%)
- **Observations**: The read is mean+0.75σ — the highest family-band read since the EXP-052 replicate pair, but inside [96.41, 96.73] and below the bar. test_loss ran higher than family throughout (final 0.2241 vs ~0.185) exactly as pre-noted: mixed-target CE arithmetic, an EXP-036-class calibration shift with accuracy untouched — yet another accuracy/CE decoupling datum. Plateau tight (final 8 evals within 0.10). peak_vram 1,613MB (plan's <3,300 expectation was mis-anchored on EXP-059's 1024-batch tail; 1,613 is the correct single-shape family value).
- **Analysis**: Pre-registered branch (ii) fired — the absorption law extends to augmentation TYPE. Occlusion-with-signal + area-proportional supervision performs identically to occlusion-with-noise at the same dose under TrivialAugment + completed anneal: TA already saturates the in-domain-diversity channel that CutMix's signal-fill adds over RE, and the mixed-label supervision neither helped (no margin information gain) nor hurt (no EXP-009-class over-pressure — substitution at constant count is safe where stacking was not). External transfer is now 0-for-17. Strategically this was the maximal falsification probe of the measured ceiling: the largest-published-prior untested construction read a precise family null at byte-clean signatures. The measured-ceiling hypothesis (family mean ≈ 96.57, reachable max ≈ bar) now stands essentially unfalsified by the entire catalogued and seam space.
- **Key Learning**: Under heavy policy augmentation the network is indifferent to WHAT fills an occluded region and to occlusion-correlated label mixing — occlusion works purely as information deletion, whose dose (not type) is the only live variable, and that dose was already optimal.

## Verification

- **Conditions**: Integrity pre-condition PASSED (RC=0, D0 22.7, windows ≤22.8, ledger on bands, ep1 36.26, no NaN, 137/137 evals). Condition 1 FAILED: 96.69 < 96.81. Conditions 2–3 pass informationally (453.0s ≤ 600; once-per-epoch structural).
- **Review Notes**: Results confirmed trustworthy — watchdog covered the full window, step ledger excludes foreign-job contamination (EXP-058 gate), probe and run signatures agree, and the metric came through the intended intervention class (training-side regularization change; eval untouched).
- **Verdict**: no-improvement
- **Verdict Basis**: Condition failure — valid family-band result below the +0.1pp bar.

## Unexplored Avenues

- **CutMix at other doses (p=0.25/0.75) or α≠1**: closed by composition — the dose axis is bracketed-peaked (EXP-003-class) and the type variable just measured null at the optimum dose; interior dose×type points have no separate mechanism.
- **Mixup SUBSTITUTED (not stacked) for RE**: the substitution seam is now measured via its strongest member (CutMix dominates mixup in every published CIFAR comparison); mixup-substituted inherits the null with an added off-manifold-input risk. Do not run.
- **Hybrid fill (half-noise/half-signal) or class-conditional patch selection**: interpolations of a measured null against a measured optimum — no mechanism.

## Next Steps

The catalogued frontier was already empty; the last open SEAM (regularizer type) is now also measured-closed by its highest-prior member. For brainstorm-061 the honest position is: (a) every axis, schedule, corner, and seam in the documented space reads at or below the family mean — the recipe is a measured local optimum in a space that has now been probed in ~54 consecutive closures; (b) remaining legal constructions are interpolations/inheritances of measured nulls or fail the +0.3pp effect-size screen (label excision, plateau-extension micro-harvests); (c) per the standing directive the loop continues — the next brainstorm should either find a construction outside ALL measured classes (none is currently catalogued; genuine novelty required, e.g. mechanisms from post-2024 literature not yet consulted via fresh lit-search) or design the cheapest experiment that could BREAK a standing law in a regime where it might bend (e.g. the absorption law has only been tested with TA present — but removing TA contradicts the dose optimum; confidence any such bend exists: low). Recommendation: brainstorm-061 should begin with a fresh external literature sweep (post-2024 CIFAR/small-model fixed-budget results) before constructing from internal seams, which are exhausted.

## Exit Action Results
