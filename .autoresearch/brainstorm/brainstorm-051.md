# Brainstorm EXP-051
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new external fetches; sources are the project record + model knowledge:

- **EXP-050's root-cause hypothesis (the direct anchor)**: reports/exp-report-050.md § Results — uniform margin pressure lost −2.4σ because "under TrivialAugment+RandomErasing, a meaningful fraction of training samples carry effectively-wrong labels; uniform margin pressure forces the network to push THOSE samples' wrong-label gaps past 5.26 too." This is currently an INFERENCE, not a measurement. It is directly testable from the opposite side: suppress destroyed-view gradients instead of amplifying them. If suppression helps, the hypothesis is confirmed AND the metric moves; if it nulls, the destroyed-view reading of EXP-050 weakens.
- **Generalized Cross Entropy (Zhang & Sabuncu, NeurIPS 2018 — model knowledge)**: GCE loss (1−p^q)/q interpolates CE (q→0) and noise-robust MAE (q=1); its logit gradient is exactly p^q × the CE gradient, i.e., a confidence-derived per-sample weight that vanishes for p→0 (wrong-label) samples. Canonical q = 0.7 improves accuracy under 20–40% synthetic label noise on CIFAR-10 by multiple points and is ~neutral at 0% noise. The detached-weight form (w_i = p_i^q × CE+LS) preserves the project's LS semantics exactly while reproducing GCE's gradient geometry.
- **Small-loss/self-paced literature (co-teaching, MentorNet — model knowledge)**: down-weighting high-loss samples is the standard, repeatedly-validated mechanism for learning under label noise. Heavy-aug views with destroyed semantics ARE label noise (the label no longer matches the visible content); TrivialAugmentWide includes severe ops (solarize, posterize, large translations) and RandomErasing covers up to 40% of the image at p=0.5.
- **Absorption-law check**: augmentation cannot supply this mechanism — augmentation CREATES the destroyed views; the loss currently treats them as ground truth. Filtering them removes systematically wrong-direction gradients (bias, not SGD variance — distinct from the gradient-noise law's batch/LR axis).

## Experimental History Review

State after 51 indexed experiments: baseline 96.71 @ 1990397, bar ≥ 96.81; mean ≈ 96.57, σ ≈ 0.16; 44 consecutive non-improvements. Frontier after EXP-050:

- **Loss-geometry: closed for static/uniform forms** (EXP-050: margin m=0.75 → 96.19 = mean−2.4σ while test_loss IMPROVED 0.185→0.150). The CE/accuracy hard-decoupling datum: CE-improving interventions are at best accuracy-neutral (EXP-011/032), at worst negative (050). Any selective variant bears a SIGN-FLIP burden — it must argue why selectivity reverses the sign, not attenuates it.
- **Sign-flip argument for this loop's lead**: EXP-050's damage mechanism was pushing destroyed-label samples HARDER (uniform margin: full pressure at p→0). Confidence weighting w = p^q does the exact OPPOSITE at the failure point — gradient → 0 for p→0 samples. It is not a smaller dose of 050's mechanism; it is its complement: 050 amplified the wrong-gradient population, 051 suppresses it. Genuine boundary samples (p ≈ 0.3–0.7) keep most of their gradient (0.5^0.7 ≈ 0.62 vs easy 0.95^0.7 ≈ 0.96 — mild relative tilt).
- **All other classes**: recipe constants bracketed/flat (049 closed the audit); structural closed (046/047/030 triangulation); throughput exhausted (048); noise optimum (011/022/023/024/035); regularizer stack peaked (009 — note: this candidate is NOT a 4th regularizer; it removes corrupted signal rather than adding invariance pressure); weight averaging closed (011/032); tail lightening closed ×2 (025/033 — note: this candidate is NOT phase-keyed; it acts whenever confidence is low, mostly mid/late on destroyed views only).
- Protocol carry-overs: composite gates (26ms), step ledger, trajectory-criterion numerics, replicate band 96.70–96.80, detection arithmetic (true effect ≥ +0.3 for one-draw detection).

## Candidate Ideas

### 1. Confidence-weighted CE — GCE-style detached weight w = p_true^0.7, batch-mean-normalized (aug-noise filtering)
**Summary**: Per-sample CE+LS with `reduction='none'`; weight each sample by `w_i = p_i^0.7` where `p_i = softmax(outputs)[i, y_i]` DETACHED; normalize weights to mean 1 per batch (detached); loss = mean(w_i × ce_i). One new constant `GCE_Q = 0.7`. Warmup loss mirrored. Eval untouched.

**Reasoning**: Tests EXP-050's destroyed-view hypothesis from the suppression side with the canonical noise-robust form. Mechanism: TA+RE generates views whose visible content no longer matches the label; their gradients point AWAY from the correct decision boundary all training long; p^0.7 weighting sends exactly those gradients (p→0) toward zero while leaving genuine-sample gradients nearly intact. Mean-1 normalization keeps integrated gradient magnitude ≈ family (heat-neutral — the heat axis is closed and must not be confounded). Costs: softmax+gather+pow on 512×10, fused, ≪ 0.1ms — signature-clean. Why it is not a "4th regularizer" (EXP-009): mixup ADDED invariance pressure; this REMOVES corrupted signal — opposite information direction. Why not tail-lightening (025/033): not phase-keyed; acts only on low-confidence views at any time. Failure shapes: (a) destroyed-view fraction too small to matter → mean-band null, hypothesis weakened; (b) hard-but-genuine samples (true boundary cases) also get suppressed enough to hurt → below-band negative, closing the selective branch too.

**Sources**: reports/exp-report-050.md (root cause + Unexplored Avenues); goal-learnings EXP-050 entry (sign-flip burden articulated); model knowledge: Zhang & Sabuncu 2018 (GCE, q=0.7), co-teaching/MentorNet small-loss principle.

**Estimated Effort**: trivial-plus — one constant + ~4 loss-site lines (detach discipline), CPU sanity for weight values and m=0/q=0 identity.

**Risk Assessment**: Graceful failure at byte-identical signatures. Branches: (i) ≥96.81 → improvement AND destroyed-view hypothesis confirmed; (ii) mean band 96.42–96.72 → selective filtering absorbed; destroyed-view hypothesis unsupported; per-sample loss class fully closed (uniform AND selective measured); (iii) <96.42 → suppression also negative (hard-genuine samples load-bearing); class closed from both sides; (iv) infra screens.

### 2. Boundary band-pass weight w = 4·p(1−p) (uncertainty emphasis) — runner-up
**Summary**: Same detached-weight scaffold, weight maximal at p=0.5, suppressing both easy (p→1) and destroyed (p→0) samples.

**Reasoning (and why not the lead)**: Also suppresses destroyed views, but ADDITIONALLY emphasizes boundary samples — which is 050's discredited direction (boundary emphasis turned CE gains into accuracy losses). Its outcome would confound two mechanisms (noise filtering + boundary emphasis) that 050/051 are trying to separate; and CE's native gradient already ∝ (1−p), so the easy-side suppression is largely redundant. Idea 1 is the clean single-mechanism probe; run this only if Idea 1 lands branch (i) and the question becomes dosing.

**Sources**: brainstorm-050 Idea 2; exp-report-050 Unexplored Avenues; focal-loss caveat (Lin et al. 2017).

**Estimated Effort**: trivial-plus.

**Risk Assessment**: Confounded attribution; carries the measured-negative boundary-emphasis component. Dominated by Idea 1.

### 3. Class-balanced batch composition (stratified sampler) — documented rejection
**Summary**: Replace shuffle=True with a per-class stratified batch sampler (51–52 per class per batch).

**Reasoning (and why rejected)**: The only other unmeasured data-side axis, but its mechanism is gradient-VARIANCE reduction (removing class-count fluctuation in batches), and the gradient-noise law (EXP-011/022/023/024/035, bracketed both implicitly and explicitly) places the recipe AT the measured noise maximum — variance-reducing interventions have lost every time (batch 1024 twice, momentum trades twice, SAM). Predicted negative by a count-5 law; recorded so it is not re-derived.

**Sources**: goal-learnings GRADIENT-NOISE LAW entry; EXP-012/022 batch-scaling entries.

**Estimated Effort**: moderate (custom sampler with persistent workers) — and rejected regardless.

**Risk Assessment**: Law-priced negative expectation; also touches loader plumbing (worker interaction risk). Rejected.

## Idea Evaluation

- **Evidence strength**: Idea 1 has the strongest combination available on the remaining frontier: a published, repeatedly-replicated mechanism (GCE/small-loss under label noise) PLUS an internal, freshly-measured pointer (050's failure localizes to exactly the population this weight suppresses). Idea 2 shares the scaffold but mixes in a measured-negative component. Idea 3 is priced by a count-5 law.
- **Mechanism clarity**: Idea 1's is exact: w = p^q reproduces GCE's gradient geometry; destroyed views (p→0) lose their vote; the LS target semantics are preserved by the detached-weight form. The sign-flip burden from EXP-050 is explicitly discharged: this is the complement of the failed mechanism, not a dose reduction.
- **Expected impact**: honest sizing — GCE's published gains are large at 20–40% injected noise and ~neutral at 0%; the aug-induced destroyed-view rate here is unknown (plausibly 2–10% of views). The realistic best case is small-positive; both null and negative branches carry high information (they resolve the destroyed-view hypothesis and close the per-sample-loss class entirely).
- **Risk profile**: byte-identical signatures, zero throughput cost, graceful failure, no heat confound (mean-1 normalization).
- **Feasibility**: ~4 lines + constant; the detach discipline is the only care point and CPU sanity covers it.

Idea 1 dominates. Idea 2 reserved for a post-success dosing question; Idea 3 recorded as law-priced.

## Chosen Idea
**Selected**: Idea 1 — Confidence-weighted CE (detached w = p_true^0.7, mean-normalized; GCE_Q = 0.7)

**Why this idea**:
It is the strongest-evidenced candidate left on the frontier: a canonical noise-robust form (GCE q=0.7) aimed at the precise failure population EXP-050 just identified, with the sign-flip burden discharged by construction (it suppresses what 050 amplified). Every branch is informative: improvement confirms the destroyed-view hypothesis; a null closes the hypothesis AND the selective per-sample class; a negative closes the class from both sides. Zero cost, no heat confound, byte-identical signatures.

**Hypothesis**:
Weighting per-sample CE+LS by detached p_true^0.7 (batch-mean-normalized) filters augmentation-destroyed views' wrong-direction gradients, improving converged boundary placement: best_test_acc ≥ 96.81 if destroyed-view gradient corruption is a binding limitation. Pre-registered branches: (i) best ≥ 96.81 → improvement, destroyed-view hypothesis confirmed (replicate pair first if 96.70–96.80); (ii) mean band 96.42–96.72 → filtering absorbed; destroyed-view hypothesis unsupported; per-sample loss class closed (uniform 050 + selective 051); (iii) < 96.42 → suppression negative (hard-genuine samples load-bearing); class closed both sides; (iv) GATE_KILL/contention → infra (a fused softmax-gather-pow on 512×10 cannot alter dt; standard screens).
