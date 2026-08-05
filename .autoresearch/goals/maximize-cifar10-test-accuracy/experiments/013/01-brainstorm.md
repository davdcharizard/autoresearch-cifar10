# Brainstorm EXP-013 (Thorough)
**Created**: 2026-06-29

<!-- Goal/metric/constraints: goals/maximize-cifar10-test-accuracy/01-definition.md. Baseline 96.38 (EXP-008, commit 07c3760), bar ≥96.48, in 04-results.tsv. Full proposals: experiments/013/proposals/idea-01..04.md. -->

## Web Search & Literature Review

- **SAM — Foret et al., ICLR 2021 (arXiv:2010.01412)** (`proposals/idea-04.md` §5): ascent to `w+e_w`, `e_w=ρ·g/‖g‖`, then descent with the perturbed-point gradient → flat minima, smaller generalization gap. ρ=0.05 canonical CIFAR-10 default (0.05–0.1 range; overestimating ρ less harmful than under). +0.3–1.0pp on CIFAR ResNets *at matched epochs*. Caveats: 2× fwd-bwd (throughput), numerically touchy under mixed precision (mitigate: compute e_w/grad-norm on fp32 master params, autocast only the forward), BN stats should update on the 1st pass only. Periodic SAM-k/LookSAM degrade accuracy → prefer a temporal tail-only split.
- **fast-CIFAR lineage** (`knowledge/references/fast-cifar10-recipes.md`): one-cycle triangular; "Bag of Tricks" (He et al. CVPR 2019) lists cosine annealing as a standard ingredient; SGDR (Loshchilov & Hutter) is the cosine-decay origin. airbench/DavidNet at batch 512 use peak LR in the ~0.4–1.0 band.
- **Implicit-reg / LR-scaling** (`proposals/idea-03.md` §5, not a finalist): SGD gradient-noise scale g∝LR·N/B (Smith & Le 2018; Goyal et al. 2017) — higher peak LR ⇒ more high-LR-phase noise ⇒ flatter minima. Directional, magnitude net-specific.
- No new `/lit-search` this loop — candidate levers grounded in prior brainstorms + the knowledge base; SAM hyperparameters pulled via the idea-04 subagent's web search.

## Experimental History Review

Current best **96.38 (EXP-008)**: DavidNet/ResNet-9 + frozen ZCA whitening + ReZero(256)@layer2 + EMA(0.998) + flip-TTA, SGD-Nesterov lr 0.4/wd 5e-4/LS 0.2, time-based triangular one-cycle (PCT_START 0.15), Cutout(12)+RandomErasing, ~150 ep/300s @ ~26.5k img/s.

- **Long plateau**: 6 of the last 7 experiments no-improvement (009 Muon diverged 94.11; 010 Muon-sweep 96.33 ties SGD; 011 CutMix 96.40 +0.02 noise; 012 WD-shaping/LS 96.29 ties/degrades).
- **Five axes now exhausted**: optimizer (Muon ties SGD, 009/010); eval-side TTA (006); input-space aug saturating (008 won +0.38 as the FIRST strong aug, 011 CutMix as a 2nd only ties); regularization-scalar allocation (012 WD-shaping ties, clean LS 0.1 degrades −0.23pp; α instrumentation showed the ReZero gate is NOT accuracy-limiting at any magnitude); large capacity steps under-anneal (005 deepen 4×4 → 131 ep; 007 widen 256→384 → 94 ep, best==final).
- **Untried levers with ceiling plausibly above the ~0.1pp noise floor**: (a) MILD capacity at the proven 8×8 stage (256→**320**, the pre-registered EXP-007 follow-up at ~1.25× vs the failed 1.5×); (b) SCHEDULE SHAPE (cosine decay / earlier peak — set EXP-001, never revisited, throughput-free); (c) loss-geometry SAM (the one axis the EXP-011 learnings entry explicitly names as the next-different-mechanism; high upside but 2× step cost → under-anneal risk).
- **Protocol**: every comparison needs a SAME-SESSION baseline cell (fixed seed still varies ±0.1pp with host-throughput epoch jitter); `num_epochs` is the first-read diagnostic on any throughput-costing change (≥142 clean, 135–141 mild, <110 under-anneal/abort).

## Diagnose What Limits the Objective

The net is **regularization-bound near its generalization ceiling at 300s**, with a large epoch surplus (~150 ep vs airbench's 37 for ~96%). The throughput-FREE regularization sub-levers that exploit that surplus have now saturated one after another — input-aug (011), weight-decay allocation + label smoothing (012) — each landing within the ~0.1pp noise floor; the optimizer (010) and eval-TTA (006) axes are likewise tied/exhausted. Two qualitatively different kinds of headroom remain. **(1) Move the ceiling via a different generalization mechanism that is NOT a scalar reallocation of the existing regularizers** — loss-geometry (SAM, flat minima) is the canonical such lever and is the one the learnings explicitly flag. **(2) Test whether the net is mildly CAPACITY-bound at the proven 8×8 stage** — EXP-004 (the last win) added capacity there; EXP-007 failed on magnitude (epoch cost), not location, so a milder step is genuinely untested. A third, cheapest probe is the **schedule shape** (the last untouched throughput-free training-side knob), whose ceiling is honestly modest (≤~0.15pp) but whose downside is nil (cannot under-anneal). The highest-value move combines a genuinely-different mechanism with a real chance of clearing noise — which points at SAM (high ceiling, names the live axis) and mild capacity (high ceiling, pre-registered), with the cosine reshape as the low-risk floor.

## Collected Ideas

- **Mild capacity widen layer2 256→320** (8×8 stage; the pre-registered EXP-007 follow-up). [experimental-history]
- **SAM (sharpness-aware minimization), tail-only** to control the 2× cost. [literature / outside-field; loss-geometry]
- **One-cycle cosine decay reshape + earlier peak (PCT_START 0.15→0.10)**. [orthogonal lever: schedule shape]
- **Peak-LR as implicit regularizer (0.4→0.5/0.6)**. [orthogonal lever: SGD-noise; cut — near a 1-param sweep, sub-noise prior, instability tail-risk]
- **Mixup/CutMix as REPLACEMENT for an occlusion aug** (not addition; per EXP-011 insight). [simplification/recombine; cut — input-space saturating]
- **Self-distillation from the EMA teacher** (KD loss vs EMA logits). [algorithm; cut — extra forward = throughput cost, complex]
- **Stochastic depth / dropout on the wide layers**. [regularization; cut — ReZero already a soft-depth gate, redundant-risk]

## Combinations

- **Mild capacity (256→320) + cosine reshape**: capacity that costs epochs needs the tail to anneal; cosine's late-tail steepening could let the added 8×8 capacity settle better in the reduced epoch budget. Deferred — confounds two variables and the capacity under-anneal risk dominates a single read; pursue capacity clean first, fold cosine in as a free rider on a future capacity win.
- **SAM only in the low-LR tail (idea-04's primary)**: itself a temporal A+B (plain SGD high-LR ∥ SAM tail) — spends the 2× cost exactly where accuracy concentrates (EXP-001) and basin-selection is meaningful, keeping global epochs ≥~120.

## Candidate Ideas

### 1. Mild capacity step — widen layer2 / 8×8 stage 256→320
**Summary**: Widen the proven layer2/8×8 stage from 256→320 channels (~1.25×), a ~3-line edit to `ResNet9.__init__` (`conv_bn(128,320)`, `GatedResidual(320)`, and the layer3 stem `conv_bn(320,512)`); all hyperparameters held. The pre-registered EXP-007 follow-up after 256→384 (1.5×) under-annealed. num_params ~7.78M→~8.82M (+1.03M, ~47% of EXP-007's +2.21M). Full proposal `proposals/idea-01.md`.

**What it targets**: a possible MILD capacity bound at the 8×8 stage — the only stage where adding capacity has ever won here (EXP-004 ReZero, +0.13pp). A different axis (capacity, not regularization) with a higher ceiling than the saturated scalar levers, IF the net isn't already capacity-saturated.

**Reasoning**: EXP-007's loss was magnitude not location — it cut epochs 150→94 (best==final, still climbing). The FLOP delta of 256→320 at the two c²-dominated 8×8 convs is ~45% of the 256→384 increment, predicting ~122–128 epochs — better-positioned but in the ambiguous band. EXP-004 is the existence proof that 8×8 capacity can win even at reduced epochs (won at 142 vs 174). 8×8 convs run at full ~26k img/s (no 4×4 cuDNN penalty), so the epoch cost is FLOP-driven and predictable.

**Sources**: `proposals/idea-01.md`; EXP-007 (`experiments/007/04-analysis.md`, the direct parent, pre-registers "try 256→320"); EXP-004 (8×8 capacity win); learnings under-anneal entry (count 2) + noise-floor entry.

**Estimated Effort**: low-medium (~3 integer literals + 1 run + same-session baseline cell).

**Risk Assessment**: dominant risk is under-anneal (recurring failure, count 2) — predicted ~122–128 ep sits in the ambiguous middle; host contention could push it toward the <110 cliff. Secondary: EXP-012's α-evidence (gate not accuracy-limiting) hints the 8×8 stage may already be partially capacity-saturated, so even a clean run could tie. Pre-registered decision gate: read num_epochs FIRST (<110 abort/under-anneal; 110–135 check best==final; ≥135 accuracy is the verdict); win needs ≥96.48 AND >same-session baseline +0.10pp. A second under-anneal datapoint would promote width to a High-importance do-not-retry.

### 2. SAM (sharpness-aware minimization), tail-only — MOONSHOT
**Summary**: Hand-implement SAM in the training step (no new deps): a 1st fwd-bwd at `w`, an ascent to `w+e_w` (`e_w=ρ·g/‖g‖`, ρ=0.05), a 2nd fwd-bwd at the perturbed point, restore `w`, then `optimizer.step()` with the perturbed gradient → flat-minima selection. To control the 2× cost, apply SAM ONLY in the low-LR tail (`progress ≥ SAM_TAIL_FRAC≈0.35`), plain SGD before → epochs ≈ `150·(1−f/2)` ≈ 124. BN running stats updated on the 1st pass only (momentum=0 on the perturbed pass) so the EMA's averaged buffers aren't double-counted. Full proposal `proposals/idea-04.md`.

**What it targets**: the generalization ceiling on a regularization-bound net via loss-GEOMETRY — orthogonal to every saturated lever (it changes which minimum the same SGD trajectory selects, not the loss function, data, weight penalty, or preconditioner). This is the exact axis the EXP-011 learnings entry names: "Future regularization should target a DIFFERENT mechanism … or loss-geometry (SAM)."

**Reasoning**: published SAM gains on CIFAR ResNets are +0.3–1.0pp (3–10× the noise floor) at matched epochs. Tail-only spends the 2× cost where this recipe's accuracy concentrates (EXP-001) and where basin-selection is meaningful, keeping epochs ≥~120 (above the ~110 under-anneal line). The bet: capturing even ~0.2pp of the flat-minima gain while losing ~26 epochs (150→124, fewer than EXP-004's winning 142) is a plausible net positive.

**Sources**: `proposals/idea-04.md`; Foret et al. ICLR 2021 (arXiv:2010.01412); davda54/sam (BN-stats tip); EXP-001 (tail pattern), EXP-002 (EMA buffers); learnings (SAM named as the next-different-mechanism; under-anneal count 2).

**Estimated Effort**: medium — contained to the training step + ~15 lines, but three correctness traps (BN double-forward stats, exact perturbation restore before step, momentum/wd applied to the perturbed grad) plus a bf16-stability assumption that needs validation.

**Risk Assessment**: dominant risk under-anneal — even tail-only cuts epochs; host load could drop a cell below ~110 (decision rule: read num_epochs first; <115 ⇒ re-run at smaller SAM_TAIL_FRAC, don't conclude SAM failed). Flat-minima gain at reduced tail-steps is a genuine coin-flip (~40% clears, ~35% ties, ~25% under-anneal), and the literature numbers are mostly on heavier backbones. bf16 numerical stability of `g/‖g‖` is load-bearing (mitigate: fp32 master-param norm, autocast only the forward; watch early-tail loss for spikes/NaN). Fallback ladder: SAM_TAIL_FRAC 0.35→0.25; ρ {0.05,0.10}; ESAM-style sparse perturbation.

### 3. One-cycle cosine decay reshape + earlier peak (PCT_START 0.15→0.10)
**Summary**: Replace the linear post-peak LR decay with a cosine half-wave to ~0 (`lr=PEAK_LR·0.5·(1+cos(π·decay))`), and in a separate cell move the peak earlier (`PCT_START` 0.15→0.10 with `EMA_WARMUP_FRAC` re-aligned to 0.10). ~5-line edit + `import math` (stdlib). Throughput-free → cannot under-anneal/crash. Same-session 3-cell read: baseline (linear/0.15) / cosine(0.15) / cosine+0.10. Full proposal `proposals/idea-02.md`.

**What it targets**: the low-LR settling tail where accuracy concentrates (EXP-001 Pattern), and the weight-EMA's tail variance (EXP-002, +0.50pp) — a flatter, lower-variance tail gives the EMA a tighter iterate cloud to average. The schedule SHAPE was set in EXP-001 and is the last untouched throughput-free training-side axis.

**Reasoning**: cosine has vanishing slope as LR→0, spending more of the final ~10% at genuinely small LR; PCT_START 0.10 lengthens the entire decay limb ~6% (shape-agnostic added low-LR time). The proposal confronts the known critique head-on: cosine trades early-tail exploration for late-tail settling, pivoting exactly at the tail midpoint — so cosine-only is expected near-noise, and the bet rides on PCT_START 0.10's tail-lengthening pushing the combined effect over +0.10pp.

**Sources**: `proposals/idea-02.md`; EXP-001 (tail pattern), EXP-002 (EMA tail synergy); EXP-011 idea-03 (the shape critique + EMA-alignment confound this answers); fast-cifar recipes (SGDR/cosine).

**Estimated Effort**: low (~5-line LR-block edit + same-session 3-cell read; the cheapest class).

**Risk Assessment**: most likely within-noise (cosine-vs-linear ≤0.1–0.2pp on an already-annealing one-cycle; honest best estimate +0.05–0.15pp — coin-flip on the bar). The avoidable failure is forgetting `EMA_WARMUP_FRAC=0.10` in the cosine+early cell (silent two-variable change). No crash/under-anneal risk. If cosine-only < baseline, the shape hypothesis is falsified and only the tail-lengthening can carry a win.

## Review

Cross-model review (Codex) in `01-idea-review.md`. **Pick: Idea-04, tail-only SAM** (evidence 7/10 after the fix, impact **8/10** — "the only credible >0.1pp ceiling among the three") over cosine (7/**4**, "right on the noise floor") and capacity (6/6, "its own arithmetic says it may reproduce the capacity-under-anneal failure"). No hard-scope violations in any idea. Key concerns + resolutions, all adopted into the plan-to-be:

1. **(Critical, load-bearing) SAM gate bug.** The idea-04 prose says "last ~35–40%" but the code sketch gates `use_sam = progress >= SAM_TAIL_FRAC` with `SAM_TAIL_FRAC=0.35` — since `progress` is elapsed-budget fraction, that runs SAM from 35%→100% = **65% of training**, predicting ~101 epochs (under-anneal danger), NOT the intended ~124. **Resolution**: define `SAM_START_FRAC = 0.65` and gate `progress >= SAM_START_FRAC` so SAM runs only in the final 35%; the `150·(1−f/2)≈124` math then holds with f=0.35 as the SAM-active fraction. This is the single most important correction and must be carried into the plan.
2. **SAM state-leak.** `sam_state={}` declared once and never cleared → stale-perturbation risk if a param lacks a grad in some step. **Resolution**: make `sam_state` local per SAM step (or clear immediately after restore).
3. **BN-freeze not exception-safe.** If the perturbed pass NaNs/throws, BN momentum stays 0 and corrupts the run. **Resolution**: wrap the perturbed pass in `try/finally` restoring BN momentum; delete `_sam_saved_momentum` after restore.
4. **Under-anneal is still the dominant risk even tail-only.** **Resolution**: pre-register `num_epochs` as the first-read gate — if a SAM cell finishes <115 ep, treat accuracy as under-anneal-confounded and re-run at a LATER start (smaller SAM fraction, e.g. `SAM_START_FRAC=0.75`), do not conclude SAM failed. Same-session baseline cell mandatory.
5. **bf16 stability of `g/‖g‖`** is load-bearing → compute the perturbation and global grad-norm on fp32 master params, autocast only the forward; watch early-tail loss for spikes/NaN.

## Idea Evaluation

Adopt the reviewer's pick: **tail-only SAM with the corrected gate**. It best fits the post-EXP-012 diagnosis — after the optimizer, eval-TTA, input-aug, and regularization-scalar axes all saturated within noise, the one remaining mechanism with a literature-backed >0.1pp ceiling is loss-geometry (flat minima), which the EXP-011 learnings entry itself names as the next-different-mechanism. Idea-02 (cosine) is the cleanest low-downside probe but its own honest estimate (+0.05–0.15pp) sits on the noise floor — better folded in later as a free rider. Idea-01 (capacity 256→320) has a high ceiling but its own epoch arithmetic (~122–128 ep) lands below its clean-anneal threshold before any host jitter, risking a milder repeat of EXP-007's under-anneal; deferred unless SAM and schedule levers stall (and then with a smaller 288 step). Full scored critique in `01-idea-review.md`.

## Chosen Idea
**Selected**: Idea-04 — **Sharpness-Aware Minimization (SAM), tail-only**, with the review's corrected gate (`SAM_START_FRAC=0.65`, SAM active only in the final 35% of the time budget) and the state-leak / BN-exception-safety / fp32-perturbation / num_epochs-first refinements. Full proposal `proposals/idea-04.md`; corrections in `01-idea-review.md` §1–5.

**Why this idea**:
The net is regularization-bound near its generalization ceiling, and every regularization lever that reallocates the existing knobs (input-aug, weight-decay, label smoothing, optimizer) has saturated within the ~0.1pp noise floor across EXP-009–012. SAM attacks generalization through a genuinely different mechanism — it changes *which* minimum the same SGD trajectory selects (flat vs sharp), not the loss, data, weight penalty, or preconditioner — and the learnings explicitly flag loss-geometry/SAM as the next axis to try. Published SAM gains on CIFAR ResNets (+0.3–1.0pp at matched epochs) are 3–10× the noise floor, the only finalist with that ceiling. The dominant cost (2× fwd-bwd → under-anneal) is controlled by spending SAM only in the low-LR tail where this recipe's accuracy concentrates (EXP-001) and where basin-selection is meaningful, keeping epochs ≈124 (> the ~110 under-anneal line) with the corrected gate.

**Hypothesis**:
Applying SAM (ρ=0.05) only in the final 35% of the time budget (`progress ≥ 0.65`, plain SGD-Nesterov before) raises `best_test_acc` above a same-session baseline by ≥0.10pp and clears the 96.48 bar, at `num_epochs ≥ 115` (the corrected gate keeps the global epoch count near ~124, above the under-anneal cliff). Falsifiable predictions: (a) if `num_epochs < 115`, the result is under-anneal-confounded (re-run at `SAM_START_FRAC=0.75`), not a SAM verdict; (b) if epochs hold ≥115 but accuracy ties same-session baseline within noise, the flat-minima gain does not survive the reduced tail-step count on this small wide-shallow net at 300s — SAM is then exhausted at this budget and the schedule/capacity levers are next; (c) early-tail loss spikes/NaN would indicate bf16 instability of the ascent step (mitigated by fp32 perturbation), a correctness failure not a method verdict.
