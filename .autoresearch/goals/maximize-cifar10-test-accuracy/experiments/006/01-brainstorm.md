# Brainstorm EXP-006
**Created**: 2026-06-28

## Web Search & Literature Review

- **cifar10-airbench `airbench96.py` TTA (Keller Jordan, arXiv:2404.00498)** (https://github.com/KellerJordan/cifar10-airbench/blob/master/legacy/airbench96.py; distilled in knowledge/references/fast-cifar10-recipes.md):
  The record 96% recipe uses `tta_level=2` = **mirror + translate** TTA, implemented entirely inside the eval forward (legitimate here — the frozen `Eval.evaluate` calls `model(inputs)` directly). Exact mechanism: base mirror `0.5*net(x)+0.5*net(x.flip(-1))`; then reflect-pad the input by 1px and take **two diagonal-shift crops** `padded[:, :, 0:32, 0:32]` (shift −1,−1) and `padded[:, :, 2:34, 2:34]` (shift +1,+1), each itself mirror-averaged; final logits `0.5*mirror + 0.5*translate`. Total **6 forward views** per image. The paper reports TTA is a meaningful, *distinct* contributor (the one feature whose speedup combines non-additively); "without any TTA" the three methods score 93.2/94.4/95.6% mean vs the headline 94/95/96 — i.e. mirror+translate TTA is worth several tenths of a pp at the 96 tier.
- **hiverge/cifar10-speedrun TTA efficiency note** (https://github.com/hiverge/cifar10-speedrun):
  Confidence-thresholded TTA — run the cheap base forward first, apply the expensive multi-view TTA only to test images whose max-softmax is below a threshold. Same accuracy, far less eval wall-clock. Useful as a cost-control rider if multi-view eval inflates total wall time toward the 600s cap.
- **"Bag of Tricks for Image Classification", He et al. CVPR 2019 (arXiv:1812.01187)** (knowledge/references/fast-cifar10-recipes.md):
  Catalogues cheap additive tricks (label smoothing — already in; cosine schedule; zero-γ residual BN; no-WD-on-bias/BN). Mostly already incorporated or supplanted by our ReZero/one-cycle recipe; the untried sliver is the WD-scope detail (exclude BN/bias from weight decay), a small regularization lever.

## Experimental History Review

Source: `04-results.tsv`, `03-experiment-learnings.md`, prior `04-analysis.md` files.

- **What's been tried (best-first):**
  - EXP-004 (current best, **96.00%**): ReZero-gated Residual(256) in layer2 → +0.13pp. Capacity at 8×8 outran its throughput cost.
  - EXP-003 (95.87): frozen ZCA patch-whitening first conv → +0.15pp (faster early convergence).
  - EXP-002 (95.72): weight EMA(0.998) + **flip-TTA gated to final 20%** → +0.50pp (eval-side, ~free).
  - EXP-001 (95.22): DavidNet/ResNet-9 + time-based one-cycle, bf16+channels_last, Cutout+LS, batch 512.
  - EXP-005 (95.90, **no-improvement**): a 2nd ReZero block at layer3/4×4 → −0.10pp. Coarse-4×4 capacity unused AND a 4×4/512 conv is ~10% slower than a FLOP-equal 8×8/256 conv (cuDNN), costing 11 epochs.
- **What worked:** eval-side wins (EMA + flip-TTA) and capacity/convergence wins at the *fine* spatial stages (whitening, layer2/8×8). Each was small (+0.1–0.5pp) — we are in the diminishing-returns tail near the documented airbench96 ceiling (~96.0–96.05 at this scale).
- **What didn't (approach-specific):** depth at the coarsest stage (layer3/4×4) — exhausted (EXP-005). Do NOT retry layer3 depth.
- **Untried gaps:** (a) **TTA is only flip/mirror** — the airbench96 record adds *translation* multi-crop TTA, which we have never tried; (b) a 2nd ReZero block at the *proven* layer2/8×8 stage; (c) a hyperparameter sweep (PEAK_LR / EMA_DECAY / TTA window / cutout / LS) around the 96.00 recipe.

## Diagnosis — What Limits the Objective

Goal metric = `best_test_acc` within a fixed 300s *training* budget (eval is off-budget, bounded only by the 10-min wall). Two structural facts frame the limiter:

1. **We are ~at the documented ceiling for this net scale.** airbench96 reports ~96.0–96.05% for a comparably-sized whitened ResNet with full TTA; we sit at 96.00. Training-side gains have shrunk to +0.13pp (EXP-004) and then went negative (EXP-005). So *training-side capacity is near saturation* for this architecture/budget — further accuracy must come from either (i) extracting more from the same trained weights at eval, or (ii) a genuinely better-targeted training change, not more coarse depth.
2. **Our eval-time augmentation is incomplete.** We do mirror-only TTA. The record recipe's TTA is mirror+translate, and the paper singles out multi-crop TTA as a distinct, non-additive contributor worth several tenths of a pp. This is the clearest *unexploited* lever: it is off the 300s training budget (so it cannot cost epochs — unlike EXP-005), it leaves the proven 96.00 training trajectory **bit-identical**, and it changes only the final-epoch eval views where `best_acc` is actually set.

The binding limiter for the next cheap win is therefore **eval-time view coverage**, not training capacity. Secondary limiters (for later loops): unverified optimality of the training hyperparameters, and whether the proven layer2/8×8 stage has any remaining capacity headroom.

## Collected Ideas
<!-- quick pass: candidates synthesized directly below; not an exhaustive bucket sweep -->

- Multi-crop (mirror+translate) TTA — port airbench96 `tta_level=2` into `forward`. [lead]
- Second ReZero block at layer2/8×8 (proven stage; avoids the 4×4 kernel penalty that sank EXP-005).
- Hyperparameter sweep around the 96.00 recipe (PEAK_LR, EMA_DECAY, TTA_START_FRAC, cutout size, label smoothing).

## Combinations
<!-- kept separate for clean single-variable attribution; noted for future loops -->

- TTA(translate) + wider TTA window (lower TTA_START_FRAC): more boosted epochs near the low-LR tail where best_acc concentrates — a natural rider *within* the TTA idea, foldable into the lead's plan rather than a separate experiment.

## Candidate Ideas

### 1. Multi-crop translation TTA (airbench96 tta_level=2)
**Summary**: Extend the eval-time TTA in `ResNet9.forward` from mirror-only to the airbench96 **mirror+translate** scheme. When `self.tta` is on, compute base mirror logits `0.5*f(x)+0.5*f(x.flip(-1))`; then reflect-pad `x` by 1px and extract two diagonal-shift crops (`[0:32,0:32]` and `[2:34,2:34]`), mirror-average each, average the two, and combine `0.5*mirror + 0.5*translate` (6 forward views total). No training-side change whatsoever — `PEAK_LR`, schedule, EMA, whitening, architecture all untouched, so the 96.00 training run is reproduced bit-for-bit and only the gated final-20% eval changes. Keep the existing `TTA_START_FRAC=0.8` gate so the 6× eval cost lands only on the ~28 final epochs (optionally widen the window or add hiverge-style confidence-thresholding if wall-clock approaches 600s).

**What it targets**: The diagnosed binding limiter — **incomplete eval-time view coverage**. We extract logits from only 2 of the 6 views the record recipe uses; translation views recover predictions the mirror-only average misses, especially near decision boundaries. (Diagnosis §2.)

**Reasoning**: Directly imports the documented record recipe. airbench's ablation isolates mirror+translate TTA as a distinct contributor (95.6→96.0 tier) and flags multi-crop as uniquely non-additive — i.e. it adds value *on top of* features we already have (mirror, whitening, EMA). It is the lowest-risk way to move the metric because it cannot cost training epochs (off-budget) and cannot perturb the proven trajectory. EXP-002 already proved eval-side TTA gating works in this codebase.

**Sources**: `airbench96.py` TTA (URL above); `knowledge/references/fast-cifar10-recipes.md`; EXP-002 analysis (flip-TTA infra); arXiv:2404.00498.

**Estimated Effort**: low (a ~10-line change to `forward`, no new state, no training change).

**Risk Assessment**: (a) Incremental gain may be <0.1pp since we already capture the mirror component — translate-only increment could be small and miss the bar. (b) 6× eval forwards on final-20% epochs inflate total wall-clock; must verify total_seconds stays <600 (mitigation: keep/raise TTA_START_FRAC, or confidence-threshold). (c) reflect-pad must match the eval normalization space (it does — padding operates on the already-normalized tensor inside forward). Worst case: clean run, ~+0.0–0.1pp, no-improvement — but zero risk to the baseline and high information value (settles whether eval-side is tapped out).

### 2. Second ReZero block at layer2 (8×8, proven stage)
**Summary**: Add a second `GatedResidual(256)` to `layer2` (which already has one ReZero block from EXP-004), at the 8×8 spatial stage that demonstrably yielded the only positive capacity result. ReZero α=0 keeps identity-init so no LR retune is needed. This is the EXP-005 idea relocated from the failed 4×4 stage to the proven 8×8 stage, where convs run at the full ~26k img/s (no cuDNN small-spatial penalty).

**What it targets**: Secondary limiter — possible residual capacity headroom at the one stage (layer2/8×8) where added depth has paid off. Attacks training-side representational capacity rather than eval coverage. (Diagnosis §1, secondary.)

**Reasoning**: EXP-004 proved capacity at layer2/8×8 helps (+0.13pp) and EXP-005 proved the *location* (4×4), not the mechanism, was the problem. A second 8×8 block avoids the throughput penalty and tests whether layer2 has more to give. Distinct from the failed EXP-005 approach (different stage, full-rate kernels).

**Sources**: EXP-004 & EXP-005 analyses; `knowledge/references/rezero-identity-init.md`; project-insights "FLOP-equal ≠ wall-clock-equal".

**Estimated Effort**: low (one-token architecture edit + trainability smoke).

**Risk Assessment**: Adds two 8×8/256 convs → real throughput cost (fewer epochs), and layer2 may already be near its capacity sweet spot (diminishing returns after EXP-004). Could repeat the EXP-004 pattern at smaller magnitude, or net-negative if the epoch loss outweighs a tiny capacity gain. Medium risk — it perturbs the proven trajectory, unlike Idea 1.

### 3. Hyperparameter sweep around the 96.00 recipe
**Summary**: Hold architecture fixed and sweep the already-exposed knobs that were set heuristically — `PEAK_LR` (0.4), `EMA_DECAY` (0.998), `TTA_START_FRAC` (0.8), `Cutout` size (8), `LABEL_SMOOTHING` (0.2) — looking for a ≥0.1pp lift. Most were inherited from the DavidNet/airbench lineage and never tuned for *this* exact net+budget.

**What it targets**: Secondary limiter — unverified optimality of training hyperparameters at the current operating point. (Diagnosis §1, secondary.)

**Reasoning**: After several structural changes (whitening, EMA, a ReZero block), the original hyperparameters may be mistuned for the current net; a focused sweep sometimes recovers ≥0.1pp cheaply. Low conceptual risk.

**Sources**: `train.py` constants; EXP-001–004 analyses; Bag-of-Tricks (arXiv:1812.01187) for LS/WD-scope.

**Estimated Effort**: medium (multiple 300s runs; each knob is one run, so a real sweep costs several GPU-loops).

**Risk Assessment**: Sweeps are expensive in loops and prone to noise-chasing near a ceiling (a 0.1pp "win" can be seed noise — and seed hacking is forbidden, so any win must be robust, not a lucky seed). Lower information density than Idea 1; likely small, possibly null. Best deferred until the cheap eval-side win (Idea 1) is settled.

## Review
Cross-model adversarial review (Codex) → `01-idea-review.md`. The reviewer scored Idea 1 highest (evidence 8/10, impact 7/10) and named it the pick: "the only candidate that attacks the stated binding limiter, keeps the 96.00% training recipe unchanged, and has the best evidence-to-effort ratio." Idea 2 (6/6 — throughput cost underplayed, stacking a 2nd block ≠ monotonic gain) and Idea 3 (4/5 — unfocused, strongest reward-hacking/noise-chasing smell, deferred) ranked below.

Top concerns raised + resolutions:
1. **Wall-clock failure risk, not training-budget risk** (6-view tail eval vs EXP-002's already-443s 2-view run; >600s = killed). → Resolution: run under `timeout 600`; keep `TTA_START_FRAC=0.8` so the 6× cost lands only on ~final-20% epochs; the plan will add a hard abort/verification on `total_seconds < 600` and a fallback to raise the gate (or add hiverge confidence-thresholding) if wall approaches the cap. This becomes an explicit plan-phase guard.
2. **Verify the exact crop/pad formula against upstream `airbench96.py`, not folklore.** → Resolution: already fetched verbatim from upstream (recorded in Web Search above): reflect-pad 1px, crops `[0:32,0:32]` and `[2:34,2:34]`, each mirror-averaged, `0.5*mirror+0.5*translate`. The plan will pin this exact formula.
3. **Gain may be <0.1pp since we already capture the mirror component → treat +0.05pp as no-improvement.** → Accepted: the goal's ≥0.1pp bar already enforces this; the experiment's value includes settling whether eval-side is tapped out, and it carries zero risk to the 96.00 baseline (training untouched).

## Idea Evaluation
Adopt the reviewer's verdict: **Idea 1, multi-crop translation TTA**. It is the unique candidate aligned with the diagnosed binding limiter (incomplete eval-time view coverage, Diagnosis §2), the lowest-effort/lowest-risk option (training trajectory bit-identical to the proven 96.00 run), and the documented record recipe. Ideas 2 and 3 are retained as future-loop directions (layer2/8×8 capacity; a focused TTA-gate tune) but deferred — the cheap, clean eval-side win should be settled first. Full scored critique in `01-idea-review.md`.

## Chosen Idea
**Selected**: Multi-crop translation TTA (airbench96 tta_level=2)

**Why this idea**:
It directly attacks the diagnosed limiter — we extract logits from only 2 of the 6 views the record recipe uses, and translation views are the one documented, non-additive TTA contributor we have never tried. It is off the 300s training budget (cannot cost epochs — the failure mode that sank EXP-005), leaves the proven 96.00 training run bit-for-bit identical (only the gated final-epoch eval changes), and is a ~10-line, no-new-state change to `ResNet9.forward`. The cross-model reviewer independently ranked it first on the best evidence-to-effort ratio. Its two flagged risks (tail-eval wall-clock; exact crop formula) are both controllable in the plan.

**Hypothesis**:
Replacing mirror-only TTA with airbench96 mirror+translate TTA (reflect-pad 1px; crops `[0:32,0:32]` and `[2:34,2:34]`; each mirror-averaged; `0.5*mirror+0.5*translate`; 6 views) on the gated final-20% epochs will raise `best_test_acc` from 96.00% to ≥96.10% (≥+0.1pp), while training remains byte-identical and total wall-clock stays under 600s. Falsifier: best_test_acc < 96.10 (translate views add <0.1pp on top of the mirror component already captured) → no-improvement, indicating the eval-side lever is near-exhausted at this net scale.
