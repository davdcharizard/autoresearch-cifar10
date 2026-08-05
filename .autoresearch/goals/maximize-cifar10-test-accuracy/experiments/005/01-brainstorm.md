# Brainstorm EXP-005
**Created**: 2026-06-28

<!-- Ideation only. Goal/metric/constraints live in 01-definition.md; baseline (96.00%) in 04-results.tsv. -->

## Web Search & Literature Review

- **airbench: "94% on CIFAR-10 in 3.29 Seconds" (Keller Jordan, arXiv:2404.00498) + `legacy/airbench96.py`** (`goals/maximize-cifar10-test-accuracy/knowledge/references/fast-cifar10-recipes.md`)
  airbench96 (the closest documented reference to our regime) reaches **96.03–96.05%** at this scale with: block widths 128/**384**/512 (block2 is 384, not our 256), 10 conv layers (3 per group + residual), and **multi-crop TTA** (`tta_level=2`: mirror × {center, ±1px reflect-pad} = 6 passes). Two implications for EXP-005: (a) the documented block2 width is 384 (capacity-via-width is on the frontier), (b) multi-crop TTA is the one airbench eval-side lever we haven't used — but airbench's *full*-TTA result at this scale is only 96.05%, **below our 96.10 bar**, so clearing the bar means going beyond the reference.
- **ReZero (Bachlechner et al. 2020, arXiv:2003.04887)** (`knowledge/references/rezero-identity-init.md`)
  Learnable scalar gate α=0 → identity at init with a live gradient; validated in EXP-004 for safe capacity addition without LR retune. Reused by the depth/width capacity ideas below.

## Experimental History Review

- **EXP-001 (95.22%)**: DavidNet + time-based one-cycle (+3.65pp). **EXP-002 (95.72%)**: EMA + flip-TTA (+0.50pp, eval-side). **EXP-003 (95.87%)**: frozen ZCA whitening (+0.15pp, early-convergence). **EXP-004 (96.00%)**: ReZero-gated `Residual(256)` in layer2 (+0.13pp) — **confirmed capacity is a binding lever** at this scale; gain outran a 32-epoch throughput cost (174→142 ep).
- **Current best 96.00%**, bar **≥96.10%**. Gains are shrinking (+3.65→+0.50→+0.15→+0.13) and we now sit at the airbench96 documented ceiling (~96.05%) — each next tenth is hard.
- **Validated patterns (compose for free)**: DavidNet+one-cycle base; EMA+flip-TTA (eval-side, orthogonal); frozen whitening (front-end); ReZero-gated capacity (use α-gate, NOT zeroed-BN-γ — post-BN ReLU kills that gradient). Most accuracy is in the low-LR tail → under-annealing (too few epochs) is the recurring threat when adding compute.
- **Untried gaps**: more capacity via **width** (block2→384, airbench96-matched) or **more depth** (2nd ReZero block); orthogonal **multi-crop TTA** (deferred from EXP-004 idea-02). A corrected shorter-warmup schedule retune remains an untried cheap probe (EXP-004 idea-03 was dropped for a math error).

## Diagnosis — what limits the objective

No single profilable bottleneck (annealed test accuracy); diagnosed from history. We are within ~0.05pp of the airbench96 documented ceiling for this ~minute-scale DavidNet. Two limiters remain credible:
1. **Representational capacity** — EXP-004 *proved* it still binds (+0.13pp from one block). But capacity curves are concave: the next capacity add likely yields less, and any add costs throughput (fewer low-LR tail epochs — the recurring under-annealing threat). The open sub-question is the most *efficient* capacity shape: width-at-8×8 vs depth-at-4×4 (FLOP-equal but different activation-traffic/throughput cost).
2. **Residual eval-time prediction variance** — flip-TTA captured the left/right component (+0.25pp at the gate, EXP-002); the translation component is uncaptured. But training's RandomCrop(pad=4) already builds shift-invariance, so the translate-TTA increment over flip-only is likely small.

Both angles are coin-flips at the hard 96.10 bar — a thorough brainstorm with cross-model review is warranted to pick the best risk/ceiling tradeoff.

## Collected Ideas

- Widen block2 256→384 (airbench96 width), keep ReZero gate + PEAK_LR=0.4. *(capacity-width / literature)*
- Add a 2nd ReZero-gated residual block in layer3 @4×4 (FLOP-equal to EXP-004's block, cheapest activation traffic). *(capacity-depth / experimental history)*
- Multi-crop TTA: mirror × {center, ±1px reflect-pad} = 6 passes (airbench tta_level=2). *(orthogonal eval-side / literature)*
- Width 320 instead of 384 (safer epoch budget ~118 vs ~106). *(capacity-width variant — risk-adjusted)*
- Corrected schedule retune: shorten warmup at fixed peak (fix EXP-004 idea-03's mechanism error). *(schedule / cheap probe — deferred, lower ceiling)*
- Cutout 8→12 (airbench96 value) for the now-higher-capacity net. *(regularization tweak — folded into a future sweep)*

## Combinations

- **Capacity + multi-crop TTA**: airbench96 uses both to hit 96.05%; orthogonal (architecture vs eval-side), additive in principle. Kept separate for clean attribution — TTA banks on top of whatever architecture wins.
- **Depth-at-4×4 + ReZero**: the cross that makes "stack another block" safe — ReZero identity-init means the deeper net starts bit-equivalent to the proven 96.00% net and earns capacity gradually, no LR retune (idea-03).

## Candidate Ideas

### 1. Capacity via width — widen block2 256→384 (airbench96 width)
**Summary**: Change layer2 to `conv_bn(128,384)+MaxPool+GatedResidual(384)` and layer3's input conv to `conv_bn(384,512)` (3 integer edits). Keep the ReZero gate and PEAK_LR=0.4. Adds ~+34% forward FLOPs / +2.2M params at the cheap 8×8/16×16 stages. (`proposals/idea-01.md`)
**What it targets**: Limiter #1 (capacity), via the airbench96-documented width. More mid-level 8×8 features → lower annealed loss floor.
**Reasoning**: airbench96 uses block2=384 on the same DavidNet lineage (96.03% over n=400). EXP-004 proved this exact block is capacity-hungry. ReZero keeps the wider block identity at init (no LR retune).
**Sources**: `proposals/idea-01.md`; airbench96 source; EXP-004 analysis; ReZero ref.
**Estimated Effort**: low — 3 integer edits, one run (+ shape/α-grad smoke).
**Risk Assessment**: Highest FLOP cost → **~106 epochs, in the under-annealing danger zone** (the developer self-flags this as the shakiest of the three and suggests width 320 → ~118 epochs as a safer risk-adjusted variant). Central ~96.02–96.08% (straddles the bar); ~30–35% clear. The +34% FLOP hit is ~2× EXP-004's, so under-annealing may eat the capacity gain.

### 2. Multi-crop TTA — mirror × {center, ±1px reflect-pad crops} (airbench tta_level=2)
**Summary**: Replace flip-only TTA inside `ResNet9.forward` with airbench's `infer_mirror_translate`: 6 forward passes (mirror over 3 reflect-pad-1 crops), weighted 0.5·mirror(center)+0.5·mean(mirror(±1px)). Eval-side only, inside one `forward()` → still one `evaluator.evaluate`/epoch. Keep the final-20% gate. Training byte-identical. (`proposals/idea-02.md`)
**What it targets**: Limiter #2 (eval-time translation-variance) — the component flip-TTA leaves uncaptured.
**Reasoning**: airbench's `tta_level=2` is the documented eval-side lever to 96.05%; EXP-002's +0.25pp flip step-up proves this model has exploitable eval-variance. Composes with capacity/EMA/whitening.
**Sources**: `proposals/idea-02.md` (+ EXP-004 idea-02); airbench96 `infer()`; EXP-002 analysis.
**Estimated Effort**: low — localized `forward()` rewrite + mandatory eval wall-clock smoke (6 passes vs 2; est. total wall ~475–490s < 600s; 4-pass fallback if hot).
**Risk Assessment**: **Lowest risk** — eval-side only, cannot regress or destabilize training. But **lowest ceiling**: we already bank the flip half, and RandomCrop(pad=4) pre-builds shift-invariance, so the translate increment is likely ~+0.05–0.15pp. Central ~96.04% (+0.04pp); ~35–45% clear. Most likely a real-but-sub-0.1pp gain (no-improvement) at the hard bar.

### 3. Capacity via depth — a 2nd ReZero-gated residual block in layer3 @4×4
**Summary**: Append `GatedResidual(512)` to layer3 (`...Residual(512), GatedResidual(512)`) — one line, reuses the existing class. 10→12 learnable convs. (`proposals/idea-03.md`)
**What it targets**: Limiter #1 (capacity), via depth at the most throughput-efficient placement.
**Reasoning**: All three placements are **FLOP-equal** (channel²·spatial invariant: 128²·256=256²·64=512²·16), but 4×4 has the **smallest activation footprint** (8192 vs layer2's 16384 elem) → least throughput hit → projects ~120–130 epochs (above the under-annealing floor, vs idea-01's ~106). ReZero identity-init makes stacking safe (bit-equivalent start, no LR retune, no early disruption — EXP-004 validated). 512-ch block = widest single transform. EXP-004 explicitly pre-registered "a second gated block" as the next probe.
**Sources**: `proposals/idea-03.md`; EXP-004 analysis (Next Steps); ReZero ref; FLOP-invariance traced from code.
**Estimated Effort**: low — one-line edit (reuses `GatedResidual`), α-grad/identity/shape smoke + throughput read, one run.
**Risk Assessment**: Safe (ReZero can't regress badly, no LR retune). Main risk is **diminishing returns** — concave capacity curve means the 2nd block likely adds <0.13pp, possibly <0.1pp; we'd be at 12 convs vs airbench96's 10. Under-annealing risk is *lower* than idea-01 (cheaper block). Central ~96.05% (+0.05pp); ~30–40% clear. Trajectory check (mid-training lead + higher tail) must be the real evidence vs ~±0.05–0.1pp noise.

Constraint/history filter: all three are in-scope (only train.py, no seed/eval changes, ≤1 eval/epoch), none retries a failed approach (capacity is a *validated* lever; TTA extends a validated lever). No candidate violates a hard constraint.

## Review

Cross-model adversarial review by Codex (full text in `01-idea-review.md`). Scored verdict: **Idea-03 8/7 (evidence/impact) — picked**; Idea-01 6/8; Idea-02 7/4. Top concerns and resolutions:

- **(Idea-01, contributed to deprioritizing it) ReZero safety claim is overstated for widening + ~106-epoch under-annealing.** Widening adds *new random* `conv_bn(128,384)`/`conv_bn(384,512)` bracket convs, so the net is NOT bit-equivalent at init (early disruption + implicit LR interaction), and the ~106-epoch estimate sits in the danger zone. → Idea-01 not selected; if width is ever revisited, use 320 (~118 ep). Note this concern does **not** apply to Idea-03: its `GatedResidual(512)` is channel-preserving (no new bracket convs), so the net IS genuinely bit-equivalent at init.
- **(Idea-02) Too low-ceiling for the hard bar + "cannot regress" overstated.** Central ~96.04% misses 96.10; training can't destabilize but logit-averaging over shifted crops *can* lower measured accuracy. → Idea-02 banked as a composable later add-on, not the primary bet; "low *training* risk, not zero *metric* risk."
- **(Idea-03, the pick) Weakest assumption: that layer3 (4×4, coarse) is the right place for capacity.** EXP-004 proved capacity binds at layer2/8×8; it did NOT prove a 4×4 block helps. → **Resolution: keep the layer3 placement (reviewer endorsed its throughput advantage — cheaper block → more annealing budget, which directly fights the recurring under-annealing failure mode), but FIX the falsification framing**: a layer3 no-improvement means "late/coarse 4×4 capacity is ineffective," NOT "capacity saturated" — the immediate fallback is a **second block at layer2/8×8 (the proven-productive location)**, not abandoning capacity.
- **(Idea-03) Throughput advantage is plausible but unverified** (4×4/512 cuDNN kernels + EMA over +4.7M params may not be as cheap as the activation-footprint argument predicts). → The α-grad/identity smoke AND a throughput read of the first ~2 epochs are **mandatory** before committing; gate on projected epochs ≥ ~110.
- No candidate violates a hard constraint (all train.py-only, ≤1 eval/epoch, seed fixed).

## Idea Evaluation

Adopting the reviewer's pick, **Idea-03 (a second ReZero-gated residual block in layer3 @4×4)**, refined per the feedback. It scored highest on evidence (8) and a solid impact (7): it directly extends EXP-004's just-validated ReZero capacity lever with the cleanest single-variable, genuinely bit-equivalent-at-init implementation, and — critically at this hard bar — carries *more annealing headroom* than the width-384 alternative whose ~106-epoch under-annealing risk the reviewer judged "probably decisive." Idea-01 is shelved (weaker safety premise + epoch loss too large near the ceiling); Idea-02 is queued as the orthogonal eval-side add-on for a later loop (low ceiling against 96.10, but composes on top of any architecture). Full scored critique in `01-idea-review.md`.

## Chosen Idea
**Selected**: Capacity via depth — append one `GatedResidual(512)` (ReZero, α=0) to `layer3` (operating at 4×4), `PEAK_LR` held at 0.4.

**Why this idea**:
EXP-004 proved representational capacity still binds at 96.00% (+0.13pp from one ReZero block), and this is the least-disruptive way to spend that validated lever again: a single channel-preserving block means the net is **genuinely bit-equivalent to the proven 96.00% net at init** (α=0 identity, no new random bracket convs — unlike widening), so no LR retune and no early-trajectory disruption (the EXP-004 mechanism, validated). Among depth placements, layer3@4×4 is FLOP-equal to EXP-004's proven block (channel²·spatial invariant) but has the smallest activation footprint → the least throughput hit → the most annealing budget (~120–130 epochs projected, comfortably above the ~110 under-annealing floor that the harder-FLOP width-384 idea violated). This throughput headroom is the decisive advantage at a hard bar where under-annealing is the recurring killer. The honest caveat (per review): a 4×4 *coarse* block is a less-proven capacity location than EXP-004's 8×8 — if it fails, the fallback is a 2nd block at layer2/8×8, not abandoning capacity.

**Hypothesis**:
Appending one ReZero-gated `Residual(512)` to layer3 (10→12 learnable convs, `PEAK_LR=0.4` unchanged) lifts `best_test_acc` from 96.00% to **~96.05–96.15%** (central ~96.08%), with ~30–40% probability of clearing the ≥96.10% bar. Falsifiable on the trajectory: identity-init means ep1/ep10 **match EXP-004 within noise** (no early disruption), the 12-conv net should **lead EXP-004 mid-training** (ep25–50) and settle a higher tail; α.grad ≠ 0 confirms the block trains. Throughput should stay ≥ ~115 epochs (cheaper 4×4 block than EXP-004's 8×8). If the tail lands ≤96.00% despite ≥115 epochs and a live gate, the conclusion is "late/coarse 4×4 capacity is ineffective" (pivot to a 2nd layer2 block or the orthogonal multi-crop TTA), NOT "capacity is saturated."

