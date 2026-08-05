# Brainstorm EXP-009
**Created**: 2026-06-28

<!-- Goal/metric/constraints in goals/maximize-cifar10-test-accuracy/01-definition.md; baseline 96.38% (EXP-008) in 04-results.tsv. Bar = ≥96.48% AND clearly above the ~0.1pp noise floor. -->

## Web Search & Literature Review

- **Strong auto-augmentation needs LONG schedules; under-fits on short ones** (Raschka aug comparison; RandAugment arXiv:1909.13719; TrivialAugment Müller&Hutter 2021): RandAugment/AutoAugment/TrivialAugmentWide are validated at 900–2000 CIFAR epochs; "smaller models benefit from weaker augmentation strength; excessive strength on a short schedule → underfitting." Our budget fits ~150 epochs and EXP-008's tail was already mildly under-annealed → **heavier augmentation is the WRONG lever now** (EXP-008's cutout12+light-erasing likely hit the sweet spot). Redirects to levers that don't further slow convergence.
- **Muon optimizer (Keller Jordan; airbench94_muon, arXiv:2404.00498 lineage)** (https://github.com/KellerJordan/cifar10-airbench): Newton-Schulz-orthogonalized momentum on ≥2D weights; the lever behind the NEWEST fast-CIFAR records (beats plain-SGD airbench at matched compute). Converges faster + reaches better minima. NS quintic coeffs (3.4445,−4.7750,2.0315), bf16, 5 iters. LR is ~order-of-magnitude smaller than SGD (≈0.02 canonical, vs airbench's 0.24 for its 10-epoch sprint) and must be retuned.
- **Bag of Tricks "No bias decay" (He et al. CVPR 2019, arXiv:1812.01187)**: exclude BN γ/β + biases from weight decay; ~0.1–0.3pp on conv nets, throughput-free.

## Experimental History Review

- **What's been tried**: 001 DavidNet+one-cycle (95.22) → 002 EMA+flip-TTA (95.72) → 003 ZCA whitening conv (95.87) → 004 ReZero block @8×8 (96.00) → 005 2nd ReZero @4×4 (95.90, no-imp, under-anneal) → 006 multi-crop TTA (95.93, no-imp, sub-noise) → 007 widen 256→384 (95.85, no-imp, under-anneal) → 008 **cutout12 + light RandomErasing (96.38, current best, +0.38pp)**.
- **What worked**: capacity @8×8 (004), eval-side EMA+TTA (002), whitening (003), and decisively **throughput-free stronger augmentation (008, +0.38pp)** — the largest lever since EXP-001.
- **Validated strategic frame** (project-insights High): the net is **regularization-bound with a ~4× epoch surplus**; throughput-FREE levers that raise the per-epoch difficulty/ceiling WITHOUT cutting steps are the productive class. Capacity adds (005/007) under-anneal and lose. Eval-side TTA near-exhausted (006).
- **NEW signal from EXP-008**: the test-acc tail was **still mildly rising at ep150** (96.32→96.38, best==final) → the harder-augmented net is slightly under-annealed in 150 epochs. Levers that further slow convergence (more aug, capacity) risk worsening this; levers that don't (penalty redistribution, EMA/schedule reshape, better optimizer) are favored.
- **Untried gaps**: decoupled weight decay (BN/α); EMA-horizon / schedule retune for the harder problem; a better optimizer (Muon). The augmentation axis is now near its short-budget ceiling (per the literature).

## Collected Ideas

1. (Literature / Bag-of-Tricks) **Decoupled weight decay** — wd=0 on BN γ/β + ReZero α. Throughput-free, doesn't slow convergence, orthogonal to EXP-008.
2. (Experimental-history / EXP-008 observation) **Shorten the EMA horizon** (EMA_DECAY 0.998→0.995) — the scored model IS the EMA, which lags the still-rising tail; a shorter horizon reads a later (better) point. Near-zero blast radius.
3. (Algorithm / moonshot) **Muon optimizer** — Newton-Schulz-orthogonalized momentum on conv/fc weights; the lever behind airbench's newest records. Highest ceiling, highest risk (LR retune, ~50 LOC, likely 2 loops).
4. (Schedule) Lower PEAK_LR (0.4→0.3) or PCT_START (0.15→0.10) or cosine decay — reshape the LR-vs-time curve to anneal the harder problem better. (Folded into idea-02's analysis; LR knobs rejected there in favor of the EMA, which sits directly on the metric.)
5. (Orthogonal, deprioritized by search) **Heavier augmentation** (RandAugment/TrivialAugmentWide, cutout→14) — search shows strong auto-aug UNDER-FITS at ~150 epochs; EXP-008 likely hit the sweet spot. Not developed.
6. (Capacity, deprioritized) Milder widen 256→320 on the raised ceiling — but it SLOWS convergence, worsening the EXP-008 under-anneal; high regression risk. Not developed.
7. (Simplification) Reduce label smoothing 0.2→0.1 — sweep-ish single knob; lower priority than the above.

## Combinations

- **#1 + #2 (decoupled WD + faster EMA)**: both throughput-free, orthogonal (penalty redistribution vs eval-average horizon); could stack to clear the bar together, but muddies single-variable attribution — keep separate for now.
- **#3 + #2 (Muon + faster EMA)**: Muon fully anneals the tail, making the EMA-horizon question moot or complementary; deferred until Muon is stable.
- **#3 + #1 (Muon + decoupled WD)**: Muon's writeups pair it with decoupled wd; a natural future compose once Muon lands.

## Candidate Ideas

### 1. Shorten the EMA horizon (EMA_DECAY 0.998 → 0.995)
**Summary**: Change one constant (`train.py:28`, `EMA_DECAY 0.998→0.995`), all else byte-identical to EXP-008. The scored model each epoch is the weight EMA (`AveragedModel`, evaluated in place of the raw iterate), a low-pass filter of the iterate with half-life ~3.6 epochs at 0.998 (≈96.5 steps/epoch). On EXP-008's still-rising, near-converged tail, that 3.6-epoch lag biases the evaluated average toward worse, higher-LR weights; shortening to 0.995 (half-life ~1.4 ep) re-centers it nearer the better final iterate, recovering the discarded tail. (proposals/idea-02.md)

**What it targets**: The EXP-008-observed **slight under-anneal of the harder-augmented net** — but attacked at the object that sits directly on the metric (the EMA), not the raw-iterate LR schedule. Throughput-free, zero convergence-slowdown.

**Reasoning**: Mechanistically the cleanest of the candidates: the metric is provably computed on `ema_model` (`train.py:345`), the EMA decay was a first-principles guess from EXP-002 (six experiments before the harder EXP-008 tail), and the step/epoch arithmetic quantifies the 3.6-ep vs 1.4-ep lag. Near-zero blast radius (one constant), clean falsification (tail at/above EXP-008 = bias-dominated as bet; below = variance-dominated).

**Sources**: proposals/idea-02.md; EXP-008 analysis §Results obs.3 (still-rising tail) + Unexplored #5; EXP-002 analysis (EMA 0.998 a first guess, progress-scheduled decay untried); `03-experiment-learnings.md` (low-LR-tail pattern).

**Estimated Effort**: low (one-constant change, one run).

**Risk Assessment**: Magnitude is intrinsically small (~+0.05–0.15pp; the tail rises only +0.06pp over 3 epochs) → plausibly sub-noise (EXP-006 fate). Core unfalsified assumption: the tail is rising-but-low-noise (bias-dominated), so a faster EMA reduces bias more than it adds variance; if the step-level iterate is noisier than the epoch prints suggest, 0.995 under-denoises and could LOSE. Worst case a small regression, not catastrophic.

### 2. Decoupled weight decay (no-decay on BN γ/β + ReZero α)
**Summary**: Split SGD into two param groups — wd=5e-4 on conv/fc weight matrices (4-D/2-D), wd=0 on the 1-D params (BN γ/β + the ReZero α scalar). One ~10-line edit at the optimizer; LR-schedule loop already writes LR to all groups. All else byte-identical to EXP-008. (proposals/idea-01.md)

**What it targets**: The regularization ceiling — redistributes the L2 penalty off params where it's spurious (BN scale; the α gate it actively fights) onto the conv/fc weights where it controls real complexity. Throughput-free AND no convergence slowdown (so it doesn't worsen the under-anneal).

**Reasoning**: Standard, evidence-backed (Bag of Tricks; fastai/timm defaults), zero downside (can't under-anneal; worst case ≈ baseline). The EXP-008 runner-up; explicitly named in EXP-008 Next Steps #2.

**Sources**: proposals/idea-01.md; arXiv:1812.01187; EXP-004 (α is live, α.grad≈0.0179); EXP-008 analysis Next Steps #2.

**Estimated Effort**: low (one ~10-line edit + final-α print, one run).

**Risk Assessment**: Self-assessed **<35% to clearly clear 96.48** — most likely sub-noise on this already-heavily-regularized recipe (LS 0.2 + cutout12 + RandomErasing + EMA); only 0.07% of params un-decayed; the α benefit is weak (SGD decay ∝ α, tiny unless α grows). Appeal is zero-downside, not ceiling.

### 3. Muon optimizer (Newton-Schulz orthogonalized momentum)
**Summary**: Replace SGD-Nesterov on the ≥2D weights (all conv + fc) with a from-scratch torch-only **Muon** (momentum update orthogonalized by a 5-iter Newton-Schulz quintic before application), keeping an SGD fallback group for the 1-D params (BN γ/β, α). ~50 LOC in train.py; requires retuning the Muon-group LR (≈0.02, vs SGD's 0.4 — orthogonal ~unit-RMS updates) and a separate fallback LR (≈0.2). Schedule/EMA integration unchanged. (proposals/idea-03.md)

**What it targets**: Both EXP-008's under-annealed tail (faster convergence completes the anneal in-budget) AND the minimum quality (orthogonalized = spectral-norm steepest descent → better-generalizing minima). The lever behind airbench's records ABOVE our current point.

**Reasoning**: Highest ceiling — it's *the* optimizer that set the fast-CIFAR records on this exact wide-shallow whitened-ResNet family. Throughput cost is genuinely small (<2% step FLOPs; ≤2–3 fewer epochs), unlike the failed capacity adds. Mechanism well-matched to the diagnosis.

**Sources**: proposals/idea-03.md; airbench94_muon (github.com/KellerJordan/cifar10-airbench); arXiv:2404.00498; `03-experiment-learnings.md` (low-LR-tail, under-anneal).

**Estimated Effort**: medium–high (new optimizer class + NS function + dual-optimizer wiring + shared-schedule edit; ~50 LOC). Realistically may need **2 loops** (stabilize+coarse-tune LR, then fine-tune).

**Risk Assessment**: Highest variance. The Muon-group LR is the load-bearing assumption and can't be swept (one fixed-seed run); 0.02 could be off 2–3× → first run may land below 96.38. NS numerics under bf16, Muon×BN/EMA interaction, conv-reshape/transpose bugs are correctness risks (mitigated by a singular-value smoke test + divergence guard). First run is "as much a calibration as a bid to win" — success judged on trajectory diagnostics too.

## Review

Cross-model adversarial review by Codex (`01-idea-review.md`, full text). Verdict scores (/10, Evidence&Reasoning / Potential-Impact): **idea-01 decoupled-WD 7/3**, **idea-02 EMA-0.995 7/4**, **idea-03 Muon 6.5/9**. **Pick: idea-03 (Muon).**

Reviewer's decisive argument: ideas 01 and 02 are *honest null-band probes* — both authors place expected gains inside the ~0.1pp noise band (WD +0.02–0.08pp, EMA best-case +0.05–0.15pp), so **under one-shot fixed-seed evaluation, even a real positive effect cannot be proven above the noise floor**. Muon is the only finalist with a credible path to a *clearly-above-noise* win over 96.38%. No hard-constraint violation found in any of the three (all fit in train.py, frozen evaluator preserved, one eval/epoch, no new deps).

Key concerns to carry into planning (idea-03):
- **[FATAL-if-ignored] Weight-decay semantics**: the idea-03 sketch adds `wd*p` to the gradient *before* Newton-Schulz orthogonalization/normalization, which distorts/erases the radial L2 penalty — i.e. it silently changes regularization in a regularization-bound recipe, confounding the "optimizer-only" attribution. **Fix: apply weight decay DECOUPLED (outside the orthogonalized update, `p ← p − lr·wd·p` applied separately), AdamW-style**, OR explicitly pre-register the run as "Muon + changed-WD-semantics." Plan MUST adopt decoupled WD.
- **LR is the load-bearing unverified assumption** (0.02 may be off 2–3×); treat the first run partly as calibration, judge on early-trajectory diagnostics, but only spend the official run once the code path is clean (smoke tests pass).
- **Throughput optimism**: many small bf16 matmuls/step add launch overhead beyond the FLOP estimate; since the schedule is time-based, slower steps → fewer epochs → possible tail-quality loss. **Verify `num_epochs` stays ~142–150; if it drops materially the run is throughput-confounded**, not a verdict on Muon.
- EMA interaction: metric is on `ema_model`; keep EMA integration byte-unchanged so Muon is the only variable.

## Idea Evaluation

- **idea-03 Muon — LEAD.** Only candidate whose upside (impact 9/10) clears the noise floor with headroom; it's *the* optimizer behind the fast-CIFAR records on this exact whitened wide-ResNet family, and its mechanism (faster convergence + better-conditioned minima) matches both the diagnosed regularization-bound state AND the EXP-008 under-anneal. Risk is real but bounded by smoke tests + a divergence guard, and a fixed-seed run can still produce an above-noise signal — which the other two structurally cannot. Worth the ~2-loop investment.
- **idea-02 EMA-0.995 — deferred rider.** Cleanest mechanism and smallest blast radius, but reviewer's point 3 is decisive: EXP-008's rising tail is observed on the *already-smoothed* EMA, so it is NOT direct evidence the raw iterate beats the EMA — a shorter horizon may cut denoising as much as lag. Honest expected magnitude sub-noise. Hold as a near-free follow-up to compose onto a Muon win (or to retry as a tail-only EMA schedule).
- **idea-01 decoupled-WD — deferred rider.** Sound and zero-downside but ~sub-noise alone (only 5,505 of 7.78M params change decay; ReZero-α decay force tiny). Best value is *as a component of Muon's decoupled-WD group* (the reviewer's required fix literally folds idea-01's partition into idea-03) — so it is not wasted, it becomes part of the Muon recipe.

## Chosen Idea

**Idea-03 — Muon optimizer (Newton-Schulz orthogonalized momentum) on the ≥2D weights, with a decoupled SGD-style group for 1D params.** This is the only finalist that can yield a provably-above-noise improvement under one-shot fixed-seed evaluation, it is the lever behind the fast-CIFAR records above our current operating point, and its convergence-acceleration mechanism directly targets the EXP-008 under-anneal while its better-conditioned minima target the regularization-bound ceiling.

**Hypothesis**: Replacing SGD-Nesterov with Muon (5-iter Newton-Schulz orthogonalized momentum, bf16) on all conv + fc weight tensors — with **decoupled weight decay applied outside the orthogonalized update** (per the review fix, absorbing idea-01's BN/α partition) and a separate SGD group for BN γ/β + ReZero α — will reach `best_test_acc ≥ 96.48%` (clearly above the ~0.1pp noise floor) within the 300s budget, while keeping `num_epochs` in the 142–150 band (throughput-uncompromised).

**Load-bearing assumptions / falsification**: (1) The Muon-group peak LR (~0.02, retuned from SGD's 0.4 because orthogonalized updates are ~unit-RMS) is within ~2× of optimal — if mis-set, the first run lands below 96.38 and serves as LR calibration (early-trajectory diagnostics: ep25/ep50 acc vs EXP-008's 92.31/93.75 tell us if LR is hot/cold). (2) Newton-Schulz is numerically stable under bf16 for these conv-reshaped matrices — guarded by a singular-value smoke test (post-NS singular values ≈1) and a divergence abort. (3) Throughput holds (num_epochs ~142–150); a material drop confounds the verdict. **Clean falsification**: `best_test_acc ≤ 96.38` with num_epochs in-band and LR in a sane range ⇒ Muon does not beat tuned SGD-Nesterov on this recipe at this budget.

**Likely 2 loops**: EXP-009 = implement + stabilize + coarse LR (one official run, possibly a calibration); a follow-up = fine-tune the Muon/SGD LR split and optionally compose the EMA-horizon rider.

**Implementation refinement during planning (see `02-plan.md`)**: the core hypothesis above is unchanged, but the *implementation* of idea-03 was grounded in the canonical reference `airbench94_muon.py` (read during planning) and the idea-review's WD-correctness concern, yielding three deviations from the proposal sketch: (1) **conv-only** Muon (`ndim==4`), fc stays on SGD (airbench routes the non-BN'd head to SGD); (2) **no weight decay on the Muon group** — airbench **weight re-normalization** `p←p·√out/‖p‖` replaces L2 (fully resolves the review's "coupled-WD-through-orthogonalization" concern by removing WD from the orthogonalized path, leaning on BN scale-invariance — idea-01's argument taken to its conclusion); (3) **PEAK_LR_MUON=0.24** (airbench's proven value for this net family) instead of the guessed 0.02 — the 12× gap is the weight-renorm-vs-update-scale convention difference, and 0.24 is the better-grounded start (review concern #1). NS steps 5→3 (airbench; lower throughput risk). These make the run a test of the *airbench Muon package as a coherent unit* rather than an isolated orthogonalization swap.
