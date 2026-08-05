# Brainstorm EXP-023
**Created**: 2026-06-30

<!-- Ideation only. Goal/metric/constraints live in 01-definition.md; baseline in 04-results.tsv. -->

## Web Search & Literature Review
<!-- Grounded in internal CIFAR-10 literature + the per-goal knowledge base; no live web search this loop. -->

- **Self-distillation / Born-Again Networks** (Furlanello et al. 2018; Zhang et al. "Be Your Own Teacher" 2019): a student trained against a teacher of the SAME capacity (or its own EMA) gains ~0.3–0.9pp on CIFAR via "dark knowledge" — input-dependent class-similarity structure in soft targets — which static label smoothing cannot supply. Free teacher = the temporal-ensemble EMA we already maintain.
- **Mean Teacher** (Tarvainen & Valpola, NeurIPS 2017): the EMA-of-weights model is a better predictor than the raw student; a consistency loss `‖f_student − f_EMA‖²` (or KL) pulls the student toward the temporally-ensembled teacher → improved generalization. Ramp the consistency weight up over training (the EMA teacher is near-random early). We already compute the EMA model for eval — so the teacher is FREE except one extra forward.
- **PolyLoss** (Leng et al., ICLR 2022) / **Focal loss**: reshape the CE loss via the (1−p_t) polynomial expansion — re-weights the per-example gradient toward hard/uncertain examples. Throughput-free loss swap; a different target/loss mechanism than LS.
- **Knowledge-distillation soft targets vs label smoothing** (Müller et al. 2019, "When does LS help?"): LS and KD both soften targets but LS ERASES inter-class similarity structure that KD PRESERVES — evidence the two are NOT redundant, so KD on top of LS can still add signal.
- **Goal-saturation check**: best single-model CIFAR-10 without extra data is ~97.3% (WRN-28-10+AutoAugment, 200+ ep) up to ~98.5% (PyramidNet+ShakeDrop+AA, 300–1800 ep). At ~150 ep/300s the practical ceiling is ~96.5–97 → the goal is NOT saturated; ~0.5–1pp headroom remains IF a learning-signal lever lands. Knowledge base: `references/fast-cifar10-recipes.md`, `references/wide-resnet.md` (EXP-022 backbone null).

## Experimental History Review
<!-- Sources: 04-results.tsv, project-insights.md, 03-experiment-learnings.md (all freshly updated post-EXP-022). -->

- **Current best / baseline**: EXP-008 = **96.38%** (commit 07c3760). Recipe: whitened ResNet-9/DavidNet + EMA(0.998) + tail flip-TTA + SGD one-cycle (peak 0.4, LS 0.2, wd 5e-4) + Cutout12+RandomErasing + bf16/channels_last, batch 512, ~150 ep.
- **What worked**: EXP-001 DavidNet+one-cycle (+3.65), EXP-002 EMA+TTA (+0.50), EXP-003 whitening (+0.15), EXP-004 one ReZero block (+0.13), EXP-008 stronger aug (+0.38). All but EXP-004 are recipe/eval-side, backbone-agnostic.
- **17 STRAIGHT NULLS (EXP-006→022)** mapping the space flat: within-DavidNet capacity-width/depth (007/014/021), optimizer/Muon (009/010), all 3 input-aug mechanisms occlusion/mixing/transform (011/015), reg-scalars wd+LS (012), SAM (013), throughput/epochs (014), BN-noise (016/017), downsampling (018), channel-attention (019), schedule-shape (020), **and a wholesale Wide ResNet backbone (022)**.
- **Decisive EXP-022 finding**: a pre-act Wide ResNet backbone TIES DavidNet at full anneal across two sizes (96.31/96.34 vs c0 96.32) → the ~96.4 ceiling is **backbone-family-INDEPENDENT**. Two best-evidenced fast-CIFAR families (DavidNet + WRN) both top out ~96.4 in 300s → the limiter is the **recipe/data/learning-signal regime**, NOT topology.
- **What hasn't been tried**: the **loss / learning-signal / target** axis. Every prior experiment kept plain CrossEntropy+LS0.2 as the training signal. Never tried: distillation/soft-dynamic-targets, consistency regularization, or loss-function reshaping. This is the one major lane untouched after 22 experiments — and the diagnosis now points squarely at it.
- **Hard assets**: (a) EMA(0.998) model already maintained every step → a FREE temporal-ensemble teacher; (b) banked torch.compile +12% (THRICE-validated, EXP-014/021/022) to fund any extra per-step forward; (c) #1 failure mode = under-anneal (`num_steps` first-class diagnostic; gate ≥ 12610 = 130 ep); (d) CPU-side changes are epoch-free (budget excludes loader wait) but GPU per-step cost under-anneals.

## Collected Ideas
<!-- 5-10 one-liners across buckets; ≥1 imports an outside technique. -->

- **(Lit/loss, lead)** **EMA self-distillation**: add `α·KL(student ‖ stop-grad EMA-teacher)` to the CE loss, teacher = the EMA model we already compute (eval-mode forward, no grad), consistency weight ramped up with `progress`. Dark-knowledge soft targets the LS0.2 static smoothing cannot supply.
- **(Lit/loss)** **PolyLoss / focal reshaping**: replace CE+LS with PolyLoss (`CE + ε(1−p_t)`) — throughput-free per-example gradient re-weighting toward hard examples.
- **(Lit/data-loss)** **Weak-strong consistency (FixMatch/UDA-style)**: enforce KL-consistency between a weakly-aug and strongly-aug view of each image — a new consistency-regularization signal distinct from plain aug.
- **(Orthogonal/eval)** **EMA+raw logit ensemble at eval**: average EMA and raw-model logits in the single per-epoch eval (still ≤1 eval) — cheap eval-side ensemble.
- **(History recombine)** **Tail-only self-distillation**: gate KD to the low-LR tail (`progress ≥ 0.6`) where the EMA teacher is strong and most accuracy lands — bounds the extra-forward cost to protect anneal.
- **(Moonshot)** **Two-teacher / second slow-EMA distillation**: a 2nd slower EMA (0.9995) as an additional teacher for a richer ensemble target.
- **(Simplification)** **Pure self-distillation REPLACING LS** (α·KL + (1−α)·CE, LS→0): test whether dynamic KD targets are strictly better than static LS rather than additive.
- **(Loss)** **Logit normalization / temperature-scaled CE** (e.g. LogitNorm, Wei 2022): control logit-norm growth to improve calibration/generalization — a loss-side regularizer.

## Combinations
<!-- A+B crosses; one line each on why the cross beats its parts. -->

- **EMA self-distillation + the EMA we already maintain**: the teacher is FREE (already computed for eval) — KD adds only ONE forward, making a normally-expensive distillation setup cheap enough to fit the 300s budget, and reuses a component already proven to help (EXP-002).
- **Self-distillation + tail-only gate**: ramping/gating KD to `progress ≥ 0.6` both (a) avoids the near-random early EMA teacher poisoning the student and (b) confines the extra-forward cost to ~40% of steps → protects the anneal gate (the EXP-013/021 under-anneal lesson applied preemptively).
- **Self-distillation REPLACING part of LS (lower LS + KD)**: since KD and LS are both soft-target regularizers but KD carries inter-class structure LS erases (Müller 2019), trading some static LS for dynamic KD targets may beat either alone without over-regularizing.
- **PolyLoss + self-distillation**: hard-example reshaping (PolyLoss) on the CE term + dark-knowledge on the KD term attack generalization from two different loss angles; low marginal cost since PolyLoss is throughput-free.

## Candidate Ideas

### 1. EMA self-distillation (dark-knowledge soft targets from the free temporal-ensemble teacher)
**Summary**: Augment the training loss with a knowledge-distillation term whose teacher is the **EMA model already maintained** for evaluation. Each step: the raw student does its usual fwd+bwd on the CE+LS loss; additionally, the EMA model produces (in eval mode, under `no_grad`, on the same input batch) soft target logits, and we add `α · T² · KL(softmax(student/T) ‖ softmax(teacher/T))` to the loss. The consistency weight `α` is **ramped with `progress`** (0 early when the EMA teacher is near-random, rising once `progress ≥ EMA_WARMUP_FRAC`), and a temperature `T≈4` softens the targets. To protect the anneal budget, the teacher forward is **torch.compile-funded** and KD may be **gated to `progress ≥ 0.5–0.6`** (tail-only) where the teacher is strong and most accuracy lands. Wrapped in the unchanged recipe (whitening, EMA, TTA, one-cycle, aug); run same-session vs a control with KD off (α=0), plus a confirmation pair if a winner.

**What it targets**: the diagnosed limiter — the **learning-signal/loss regime** that EXP-022 implicated when a wholesale backbone swap tied (ceiling is recipe/data-bound, not topology). KD injects input-dependent inter-class similarity structure ("dark knowledge") into the targets, which static LS 0.2 provably cannot supply (Müller 2019) — a genuinely new regularization mechanism on the one axis (training target) untouched across 22 experiments.

**Reasoning**: Self-distillation / born-again nets add ~0.3–0.9pp on CIFAR over strong baselines, and the mean-teacher EMA is a better predictor than the raw student (EXP-002 already exploits this at eval). The teacher is essentially FREE (already computed) — the only new cost is one extra forward, fundable by the thrice-validated compile headroom and bounded by a tail-only gate. KD targets generalization directly, which is exactly what a generalization-ceiling-bound net needs.

**Sources**: Furlanello 2018 (born-again); Tarvainen & Valpola 2017 (mean teacher); Müller 2019 (LS vs KD non-redundancy); Hinton 2015 (KD/temperature); EXP-002 (EMA-teacher helps at eval); project-insights ceiling bullet (EXP-022).

**Estimated Effort**: medium — a KD loss term + EMA-teacher forward + α-ramp/gate + compile of the teacher forward; recipe otherwise unchanged. Needs a throughput smoke (under-anneal risk).

**Risk Assessment**: (a) **Under-anneal** — the extra teacher forward cuts epochs; mitigate with compile + tail-only KD gate + `num_steps ≥ 12610` gate (pre-smoke like EXP-022). (b) **Redundancy with LS0.2** — KD may add little on an already soft-target-regularized net near its ceiling (the EXP-011/015 "2nd regularizer ties" pattern); mitigate by also testing KD with reduced LS. (c) **Teacher quality** — early EMA is weak; the progress-ramp/gate handles this. Worst case: ties at ~96.4 → strengthens the "loss axis also saturated" evidence.

### 2. PolyLoss / loss-function reshaping (throughput-free)
**Summary**: Replace the CE+LS objective with **PolyLoss** — `L = CE + ε·(1 − p_t)` (the leading dropped polynomial term of the CE Taylor expansion), which re-weights the gradient toward low-confidence (hard) examples; sweep `ε ∈ {−1, +1, +2}` (negative ε down-weights hard examples, positive up-weights). Optionally compose with the existing LS. Completely throughput-free (a scalar loss change), so zero under-anneal risk; same-session vs CE+LS control.

**What it targets**: the loss/learning-signal axis — changes which examples dominate the gradient, potentially extracting more from the hard tail of CIFAR-10 that a uniform CE+LS treats flatly.

**Reasoning**: PolyLoss reports +0.2–0.5pp on ImageNet/CIFAR over tuned CE with a single hyperparameter; it is the cheapest possible probe of the untouched loss axis. Being throughput-free, it cannot under-anneal — the safest way to test whether the loss function is a live lever at all.

**Sources**: Leng et al. ICLR 2022 (PolyLoss); Lin et al. 2017 (focal loss).

**Estimated Effort**: low — a few-line loss change + a small ε sweep, throughput-free.

**Risk Assessment**: Likely marginal — on balanced, clean CIFAR-10 with LS0.2 already tuned, per-example reshaping may tie (the regularization-scalar axis EXP-012 was largely closed). Low cost makes even a null cheap, but the upside ceiling is lower than KD's. Best as a cheap secondary, not the lead.

### 3. Weak-strong consistency regularization (FixMatch/UDA-style)
**Summary**: For each image, form a **weakly-augmented** view (crop+flip only) and the existing **strongly-augmented** view (Cutout+RandomErasing); add a consistency loss `KL(strong-view logits ‖ stop-grad weak-view logits)` so the strong view is pulled toward the (cleaner) weak-view prediction. A new consistency-regularization signal distinct from the data-augmentation content itself. Needs two forwards/step (weak + strong) → compile-funded, tail-gated; same-session vs control.

**What it targets**: the learning-signal axis via input-consistency — encourages augmentation-invariant representations beyond what plain aug supplies, a regularizer on the function's smoothness.

**Reasoning**: Weak-strong consistency is the core of FixMatch/UDA semi-supervised gains and improves fully-supervised generalization too. Mechanistically distinct from self-distillation (consistency across views vs distillation from a temporal-ensemble teacher).

**Sources**: Sohn et al. 2020 (FixMatch); Xie et al. 2020 (UDA).

**Estimated Effort**: medium-high — two forwards/step (heavier than KD's one extra forward) → higher under-anneal risk.

**Risk Assessment**: Two forwards/step is costly → strong under-anneal risk (worse than idea 1's single extra forward). Consistency may also be redundant with the strong aug already in the recipe. Dominated by idea 1 on the cost/evidence tradeoff; hold as fallback.

## Review
<!-- Reviewer's top concerns + resolutions; full review in 01-idea-review.md. -->

Codex (cross-model) picked **idea 1 (EMA self-distillation)**: evidence 8/10, impact 8/10 — the only candidate with a genuinely-new learning signal, clearly ahead of idea 2 (6/5, "safe but probably small, overlaps EXP-012 scalar tuning") and idea 3 (4/4, "most redundant — semi-supervised consistency doesn't transfer to fully-supervised CIFAR with strong labels + saturated aug"). Concerns folded into the chosen idea / plan:

1. **KL DIRECTION FIX (real correctness bug, not a detail)**: the brainstorm wrote `KL(student ‖ teacher)` (reverse KL). Standard KD uses the **detached teacher probabilities as the TARGET**: `L_KD = T² · KL(softmax(teacher/T) ‖ softmax(student/T))` = cross-entropy of the student against soft teacher labels. **Resolution**: implement forward-KL with `teacher = softmax(EMA_logits/T).detach()` as the target distribution; verify with a smoke (KD loss → 0 when student==teacher).
2. **Cost optimism — KD forward is GPU time, not free**: all-step KD risks the SAM/capacity under-anneal trap. **Resolution**: tail-only KD gate (`progress ≥ ~0.5`) + torch.compile the teacher forward + a mandatory throughput smoke requiring `num_steps ≥ 12610` (130 ep) before the official run, exactly as EXP-022.
3. **Redundancy with LS 0.2 (over-softening)**: LS0.2 + KD may double-soften. **Resolution**: include a **reduced-LS arm** (e.g. LS 0.1 + KD, and/or LS→0 + KD) as a pre-registered cell, since KD carries inter-class structure LS erases (Müller 2019) — test additive vs replacement.
4. **Multiple-comparison reward hacking** (sweeping α/T/gate against best_test_acc near the 0.1pp floor): **Resolution**: pre-register a SMALL grid (≤3–4 KD cells), always compare against a **same-session α=0 control**, confirm any apparent winner on a 2nd pair, and keep **≤1 eval/epoch** (eval the EMA only — never raw+EMA separately per epoch).

## Idea Evaluation
<!-- How the verdict lands; scored critique in 01-idea-review.md. -->

Adopt the reviewer's pick (idea 1, EMA self-distillation) without override — it is the only candidate attacking the freshly-diagnosed loss/learning-signal limiter (EXP-022 showed the ceiling is backbone-family-independent) with a genuinely different target source, and the only one with credible >0.1pp upside. Idea 2 (PolyLoss) is retained as a cheap throughput-free secondary probe; idea 3 (consistency) is held as a fallback (dominated on cost/redundancy). See `01-idea-review.md`.

## Chosen Idea
**Selected**: EMA self-distillation — dark-knowledge soft targets from the already-maintained EMA teacher, tail-gated and compile-funded.

**Why this idea**:
After 17 nulls and the EXP-022 finding that the ~96.4 ceiling survives a wholesale backbone-family change, the limiter is the recipe/data/learning-signal regime — and the training target (plain CE+LS0.2) is the single major axis untouched across 22 experiments. KD from the EMA teacher injects input-dependent inter-class similarity ("dark knowledge") that static label smoothing provably cannot supply (Müller 2019), directly targeting generalization. The teacher is essentially free (already computed for eval, EXP-002), so the only new cost is one extra forward — fundable by the thrice-validated compile headroom and bounded by a tail-only gate to protect the anneal budget. Codex scored it the clear highest-EV bet (8/8).

**Hypothesis**:
Adding a tail-gated EMA self-distillation term (`L = (1−α)·CE_LS + α·T²·KL(softmax(teacher/T) ‖ softmax(student/T))`, teacher = detached EMA logits, α ramped after `progress ≥ ~0.5`, T≈4) — compile-funded so the official run anneals at `num_steps ≥ 12610` — will lift `best_test_acc` to ≥ 96.48 and beat a same-session α=0 control by more than the ~0.1–0.2pp noise floor, replicated on a confirmation pair, because the dynamic dark-knowledge targets supply generalization signal the static CE+LS recipe lacks. If instead KD ties the control at full anneal (as a redundant-with-LS regularizer), the loss-target axis joins the closed lanes and the ~96.4 ceiling is confirmed robust to the learning-signal regime too — strong evidence the goal is near its 300s practical ceiling.
