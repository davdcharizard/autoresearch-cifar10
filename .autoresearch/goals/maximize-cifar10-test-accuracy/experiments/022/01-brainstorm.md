# Brainstorm EXP-022
**Created**: 2026-06-30

<!-- Ideation only. Goal/metric/constraints live in 01-definition.md; baseline in 04-results.tsv. -->

## Web Search & Literature Review
<!-- External web search was attempted this loop but declined by the operator; ideation is grounded in
     the per-goal knowledge base (references/fast-cifar10-recipes.md) plus established CIFAR-10
     backbone literature already internalized. Standard references cited below by name for the
     downstream plan phase to re-pull if needed. -->

- **Wide Residual Networks** (Zagoruyko & Komodakis, BMVC 2016, arXiv:1605.07146): widening a pre-activation ResNet (depth 16–28, widen factor k=4–10) beats very deep thin ResNets at EQUAL or fewer FLOPs and trains faster wall-clock per unit accuracy. WRN-28-10 reaches ~96.0% plain / **~97.1% with cutout** on CIFAR-10. Key structural differences from our DavidNet: pre-activation basic block (BN→ReLU→Conv→BN→ReLU→Conv), N>1 blocks per stage, 3 stages at 32×32/16×16/8×8, **GAP** readout (not global MaxPool), no whitening/scale_out tricks. This is the canonical "higher-ceiling" CIFAR backbone and the strongest evidence that a different topology has a ceiling clearly above our 96.4.
- **Cutout / DeVries & Taylor 2017** (arXiv:1708.04552): WRN-28-10 + cutout = 97.1% is the headline number we already partially exploit (we run Cutout12 + RandomErasing). Confirms our aug stack is compatible with WRN's published ceiling.
- **Identity Mappings in Deep ResNets** (He et al., ECCV 2016, arXiv:1603.05027): pre-activation ordering (BN-ReLU-Conv) gives a clean identity gradient path and trains deeper nets more stably than post-activation — the ordering WRN uses and our DavidNet does not.
- **ConvMixer** (Trockman & Kolter, 2022, arXiv:2201.09792): isotropic patch-embed + alternating depthwise (large-kernel) and pointwise convs; a genuinely different (patch/mixing) inductive bias that reaches ~96% on CIFAR-10. Distinct from the ResNet family entirely.
- **Knowledge base**: `references/fast-cifar10-recipes.md` — the DavidNet→hlb→airbench lineage chose wide-SHALLOW ResNet-9 for *speed*; airbench's 95→96 levers (width trick, depth trick) are exactly what EXP-007/014/021 already closed on OUR net. The lineage's choice does not prove ResNet-9 has the highest *ceiling* — only the best accuracy-per-second in the 35s regime.

## Experimental History Review
<!-- Sources: 04-results.tsv, project-insights.md, 03-experiment-learnings.md -->

- **Current best / baseline**: EXP-008 = **96.38%** (commit 07c3760). The base recipe is the whitened ResNet-9/DavidNet: frozen ZCA patch-whitening conv (3→54) → conv_bn stem → 3 stages (each conv_bn→MaxPool→Residual; layer2 has a ReZero GatedResidual) → global MaxPool → scale_out·Linear; weight EMA(0.998) + tail flip-TTA; SGD-Nesterov one-cycle (peak 0.4, mean-loss, wd 5e-4, LS 0.2, linear triangular); Cutout12 + RandomErasing; bf16/channels_last; batch 512; ~150 ep (173 compiled).
- **What worked**: the wins are EXP-001 (DavidNet+one-cycle, +3.65), EXP-002 (EMA+TTA, +0.50), EXP-003 (whitening, +0.15), EXP-004 (one ReZero block, +0.13), EXP-008 (stronger aug, +0.38). All but EXP-004 are recipe/eval-side and **backbone-agnostic** — they transfer to any backbone.
- **What didn't (16 STRAIGHT NULLS, EXP-006→021)**: the within-DavidNet search space is exhaustively mapped FLAT — capacity-width (EXP-007/014), capacity-depth (EXP-005/021), optimizer/Muon (009/010), all 3 input-aug mechanisms occlusion/mixing/transform (008-won/011/015), reg-scalars wd+LS (012), SAM loss-geometry (013), epochs/throughput (014), BN-stat noise (016/017), anti-aliased downsampling (018), channel-attention/SE (019), schedule-shape/cosine (020). The decisive datapoint: **EXP-014 + EXP-021 bought epochs via torch.compile and BOTH extra width AND extra depth, properly annealed, STILL tied** → the ~96.3-96.5 plateau is a genuine GENERALIZATION CEILING *of this backbone*, not an epoch/throughput/anneal limit.
- **What hasn't been tried**: a **wholesale different backbone**. Every prior experiment kept the DavidNet skeleton. The project-insights High-Importance ceiling bullet explicitly names the remaining high-EV direction: "a structurally different backbone (pre-activation/pyramidal/attention-stem); the banked torch.compile +12% recipe is TWICE-validated and ready to fund the new backbone's per-step cost."
- **Hard transferable assets**: (a) torch.compile off-budget-warmup recipe = +12% throughput, banked (EXP-014/021); (b) EMA+TTA, one-cycle, strong aug, whitening — all backbone-agnostic; (c) the #1 failure mode is **under-anneal** — any heavier net fits fewer epochs; `num_epochs` is the first-class diagnostic (≤~110 = under-anneal; healthy band 130-173).

## Collected Ideas
<!-- 5-10 one-liners across buckets; ≥1 imports an outside technique. -->

- **(Literature, lead)** Replace DavidNet with a budget-sized **pre-activation Wide ResNet** (WRN-16-k / WRN-22-k, 3 stages of N basic blocks, GAP head), sized via a throughput pre-smoke to anneal ≥130 ep, wrapped in the proven recipe (whitening stem + EMA + TTA + one-cycle + strong aug + compile).
- **(Algorithm/representation, moonshot)** **ConvMixer** isotropic backbone: patch-embed (patch 2 → 16×16 tokens) + L blocks of depthwise(k=5-9)+pointwise mixing + GAP; a genuinely non-ResNet inductive bias.
- **(History recombine + orthogonal)** **Resolution-preserving pre-act ResNet**: keep DavidNet's fast 8×8/256 stage but add a *fourth* conv stage / delay pooling so more capacity sits at 16×16 (where EXP-004 showed capacity is useful), pre-activation ordering, GAP head.
- **(Moonshot)** **Hybrid conv-stem + small ViT**: 3-conv stem to 8×8 tokens then 4-6 transformer blocks (attention inductive bias absent from every prior experiment).
- **(Literature)** **PyramidNet-style gradual widening** (additive channel growth per block) + ReZero gating, budget-sized — known ~98% ceiling but heavy; almost certainly under-anneals at 300s.
- **(Orthogonal)** **DenseNet-BC** (growth-rate k, bottleneck+compression) — high param-efficiency ceiling but concat + many small convs are throughput-poor → under-anneal risk.
- **(Simplification)** Drop whitening + scale_out and let a clean WRN stand on standard design, to test whether the DavidNet-specific tricks are load-bearing or vestigial on a higher-ceiling net.
- **(Literature)** **ResNeXt / grouped-conv** cardinality block — modest documented gains over WRN at equal FLOPs; same family, low novelty.

## Combinations
<!-- A+B crosses; one line each on why the cross beats its parts. -->

- **Wide ResNet + banked torch.compile**: WRN's larger per-step cost is exactly what the twice-validated +12% compile headroom was banked to fund — compile turns a borderline-under-anneal WRN size into an annealing one, making the higher-ceiling backbone reachable in 300s.
- **Wide ResNet + whitening stem + EMA + tail-TTA (proven recipe wrapper)**: the +0.50 (EMA+TTA) and +0.15 (whitening) wins are backbone-agnostic eval/front-end levers; stacking them on WRN's higher base ceiling plausibly clears 96.48 by more than either the backbone-swap or the recipe alone.
- **Wide ResNet + cutout/RandomErasing (current aug)**: WRN-28-10's published 97.1% is *specifically with cutout* — our existing Cutout12+RandomErasing is the exact aug regime under which WRN's ceiling advantage is documented, so the aug stack and the new backbone are mutually reinforcing rather than redundant.
- **Resolution-preserving stage + pre-activation**: delaying a pool to keep a 16×16 stage only helps if the deeper gradient path stays clean — pre-activation's identity gradient makes the extra-depth-at-higher-resolution affordable without the dead-block risk that ReZero patched on DavidNet.

## Candidate Ideas

### 1. Budget-sized pre-activation Wide ResNet (WRN) backbone
**Summary**: Replace the DavidNet backbone wholesale with a pre-activation Wide ResNet — the canonical higher-ceiling CIFAR-10 conv net — while keeping every backbone-agnostic recipe win. Concretely: a `conv 3×3 (3→16k₀)` stem (optionally fed by the existing frozen ZCA whitening conv), then 3 stages of N pre-activation basic blocks (`BN→ReLU→Conv3×3→BN→ReLU→Conv3×3`, identity/projection shortcut), widths `16k, 32k, 64k` at spatial 32/16/8 with stride-2 at stage entries, a final `BN→ReLU→GAP→Linear` head (replacing global MaxPool + scale_out). Start from a size that the literature places well above our ceiling (WRN-28-10 → 97.1% w/ cutout) but **down-scale depth/width via a throughput pre-smoke** so the net anneals at ≥~130 epochs under torch.compile in 300s — likely WRN-16-4, WRN-22-4, or WRN-16-8 depending on measured img/s. Keep EMA(0.998), tail flip-TTA, one-cycle time-based schedule, LS 0.2, Cutout12+RandomErasing, bf16/channels_last, batch 512. Run **same-session vs the DavidNet baseline control**, and (if a winner) a confirmation pair, per the noise-floor protocol.

**What it targets**: the diagnosed limiter — a **generalization ceiling of the DavidNet backbone specifically** (~96.4), proven flat to width/depth/anneal by EXP-014+021. WRN is a structurally different topology (pre-activation, multi-block stages, GAP head, no whitening/scale_out dependence) with a *documented* CIFAR-10 ceiling of ~97.1% under cutout — clearly above 96.48. It attacks the ceiling by changing the architecture family, not by adding capacity to the saturated one.

**Reasoning**: This is the highest-evidence "different backbone" available. WRN's 97.1%-with-cutout is the exact aug regime we run, so the ceiling advantage is documented under matched augmentation. The #1 risk (under-anneal) is directly mitigated by (a) the twice-validated banked compile +12% (EXP-014/021) and (b) sizing the net via a pre-smoke against the known healthy 130-173 ep band — the same num_epochs-first gate that diagnosed every prior capacity null. The recipe wrapper (EMA+TTA+whitening+aug) is backbone-agnostic and already worth ~+1.2pp combined, so it stacks on WRN's higher base.

**Sources**: WRN (arXiv:1605.07146); cutout (arXiv:1708.04552); pre-act ResNet (arXiv:1603.05027); `references/fast-cifar10-recipes.md`; project-insights High-Importance ceiling bullet (EXP-014/021); compile recipe `references/torch-compile-throughput.md`.

**Estimated Effort**: medium-high — a new backbone module in train.py + a throughput pre-smoke to fix the size; recipe wrapper unchanged. Single experiment, possibly 2-3 size cells in one same-session run.

**Risk Assessment**: Primary risk is **under-anneal** — even compile-funded, a too-large WRN fits <110 ep and loses on epochs not ceiling (exactly EXP-007/021). Mitigation: pre-smoke img/s and pick the largest size that holds ≥130 ep; include a smaller fallback cell. Secondary risk: the recipe (peak LR 0.4 mean-loss, scale_out, whitening) is tuned for DavidNet and may need light adaptation for a GAP-head WRN (no scale_out; LR may need re-checking) — keep changes minimal and documented. Worst case: WRN ties at ~96.4 too, which would be a strong result (ceiling is recipe/data-bound, not backbone-bound) and would redirect future loops to data/aug-ceiling work.

### 2. Resolution-preserving pre-activation ResNet (more capacity at 16×16)
**Summary**: A lighter-touch "different backbone": keep a ResNet skeleton but re-allocate where capacity lives, using pre-activation ordering and a GAP head. DavidNet pools aggressively (32→16→8→4) and puts its only extra capacity at 8×8. EXP-004 showed capacity at the 8×8/16×16 stages is the useful kind. This idea **delays a pooling step** (or adds a stage of pre-activation blocks at 16×16) so more residual depth operates at 16×16, where features are still spatially rich, with pre-activation's clean identity gradient making the added depth affordable. GAP head replaces global MaxPool. Wrapped in the same proven recipe + compile, sized to anneal ≥130 ep.

**What it targets**: same ceiling limiter, but tests a specific hypothesis about *where* DavidNet's ceiling comes from — its aggressive early pooling discards 16×16 spatial capacity that a higher-ceiling net (WRN keeps a full 16×16 stage) exploits. Attacks the ceiling via resolution/capacity *placement* + pre-activation, a smaller departure than a full WRN.

**Reasoning**: Partially de-risks idea 1: it reuses more of the validated DavidNet recipe (so fewer confounds), while still introducing the two structural features WRN credits for its ceiling (16×16 multi-block stage + pre-activation + GAP). If a full WRN under-anneals, this intermediate point may anneal while still raising the ceiling. Pre-activation is cheap and well-evidenced (arXiv:1603.05027).

**Sources**: pre-act ResNet (arXiv:1603.05027); WRN (arXiv:1605.07146); EXP-004 (capacity-at-8×8 win), EXP-005 (4×4 capacity wasted) — both argue higher-resolution capacity is the useful kind; `03-experiment-learnings.md`.

**Estimated Effort**: medium — moderate edits to the ResNet9 module (reorder to pre-act, add/relocate a 16×16 stage, GAP head).

**Risk Assessment**: Could land too close to "more of the same" within-ResNet capacity (which the ceiling already resists) and tie. The 16×16 stage is throughput-expensive (more spatial × channels) → under-anneal risk; needs the same pre-smoke gate. Less upside than a full WRN if the ceiling is genuinely a family property.

### 3. ConvMixer isotropic patch-mixing backbone (moonshot)
**Summary**: Abandon the conv-ResNet family entirely for an isotropic ConvMixer: a patch-embed conv (patch size 2, stride 2 → 16×16 tokens, dim h≈256) + BN/GELU, then L (≈8) residual blocks each = depthwise conv (large kernel k≈5-9, "spatial mixing") + pointwise 1×1 conv ("channel mixing"), each with BN+GELU, then GAP→Linear. A genuinely different inductive bias (patch tokenization + separable large-kernel mixing) absent from all 21 prior experiments. Wrapped in the recipe + compile, sized to anneal.

**What it targets**: the ceiling, via the most *mechanistically distinct* architecture — if the ~96.4 plateau is a property of the conv-ResNet inductive bias, only a non-ResNet backbone can break it. ConvMixer's separable large-kernel mixing has a different effective receptive field and capacity profile.

**Reasoning**: Highest novelty / highest variance. ConvMixer reaches ~96% on CIFAR but typically with longer training and heavy aug; at 300s/~150 ep its from-scratch ceiling here is genuinely unknown. Worth one slot as the diversified non-ResNet bet, but the evidence that it CLEARS 96.48 in our budget is weaker than WRN's.

**Sources**: ConvMixer (arXiv:2201.09792); contrast with the all-ResNet experimental history.

**Estimated Effort**: medium — compact module, but large-kernel depthwise throughput is uncertain (depthwise convs can be kernel-inefficient on this GPU → under-anneal).

**Risk Assessment**: From-scratch CIFAR ConvMixer at low epoch budgets often underperforms tuned CNNs; depthwise-large-kernel throughput may force tiny dims → under capacity. Higher chance of a sub-96.4 result than idea 1. Justified only as the diversified moonshot, not the lead.

## Review
<!-- Reviewer's top concerns + resolutions; full review in 01-idea-review.md. -->

Codex (cross-model) reviewed all three finalists and picked **idea 1 (budget-sized pre-activation WRN)**: evidence 9/10, impact 8/10 — clearly ahead of idea 2 (6/5) and idea 3 (4/6). Top concerns and how they fold into the chosen idea / plan:

1. **"Won or lost on size selection — enforce 'budget-sized' brutally."** WRN-28-10's 97.1% is matched-epoch and for the FULL net; a 300s-budget variant may not inherit it. **Resolution**: make the experiment a **size-gated sweep** — pre-smoke throughput for WRN-16-4 / WRN-22-4 / WRN-16-8 (and a small fallback), pick by **measured `num_epochs ≥ 130`** (healthy band 130-173), NOT by param count or paper accuracy. Run the largest size that holds the band.
2. **"Reject any result < ~110 ep as under-anneal, not a backbone verdict."** **Resolution**: pre-register the num_epochs gate; treat a sub-110-ep cell as a disqualified (under-annealed) datapoint exactly as EXP-007/021, not as evidence against WRN.
3. **"Don't assume DavidNet recipe gains stack additively."** EMA/TTA/whitening gains were measured on DavidNet; whitening changes the WRN stem input shape and GAP removes `scale_out`. **Resolution**: keep the backbone-agnostic wins (EMA, tail-TTA, Cutout12+RandomErasing, one-cycle, LS 0.2, compile), but treat whitening / scale_out-removal / peak-LR as **confounds to control** — run a **same-session DavidNet compiled control**, and if the WRN early-epoch trajectory looks broken (divergence/flat), do a minimal peak-LR sanity check before declaring a verdict. Start WRN WITHOUT the whitening stem (clean standard WRN) to avoid the stem-shape confound; whitening can be added as a follow-up rider if WRN itself shows promise.
4. **Idea 2 too close to the exhausted space** (capacity-placement variant of already-closed width/depth) and **idea 3 weakly evidenced for the 96.48 bar** (cited ~96%, budget behavior unknown, large-kernel depthwise throughput risk). **Resolution**: hold 2 and 3 as fallbacks — pursue 3 (ConvMixer) only as a diversification experiment if WRN ties; pursue 2 only if reframed as a genuinely new stage topology.

## Idea Evaluation
<!-- How the verdict lands; scored critique in 01-idea-review.md. -->

Adopt the reviewer's pick (idea 1, WRN) without override — it is the only candidate with BOTH a documented CIFAR-10 ceiling comfortably above 96.48 AND a direct match to the mandated "wholesale different backbone" direction (EXP-014/021 ceiling diagnosis). Ideas 2 and 3 scored materially lower (2 leans on capacity-placement logic that fully-annealed DavidNet tests already exhausted; 3 is novel but cited at only ~96% with unknown budget throughput). See `01-idea-review.md` for the full scored critique.

## Chosen Idea
**Selected**: Budget-sized pre-activation Wide ResNet (WRN) backbone, size-gated by a throughput pre-smoke.

**Why this idea**:
After 16 straight nulls, EXP-014+021 proved the ~96.4 plateau is a generalization ceiling **of the DavidNet backbone specifically** — flat to width, depth, and extra anneal epochs. The only untried high-EV direction is a structurally different backbone, and WRN is the highest-evidence one: a pre-activation, multi-block-per-stage, GAP-head conv net with a documented CIFAR-10 ceiling of **~97.1% under cutout** — the exact aug regime we run. The #1 risk (under-anneal) is directly fundable by the twice-validated banked torch.compile +12% (EXP-014/021) and controllable by sizing the net via a pre-smoke against the known healthy 130-173 ep band. The backbone-agnostic recipe wins (EMA+TTA ≈ +0.50, strong aug, one-cycle) stack on WRN's higher base ceiling. Same-session vs a DavidNet compiled control + a confirmation pair guards against the recurrent low-control-draw artifact.

**Hypothesis**:
A budget-sized pre-activation Wide ResNet (the largest of WRN-16-4 / WRN-22-4 / WRN-16-8 that anneals at `num_epochs ≥ 130` under torch.compile within the 300s budget), wrapped in the proven EMA + tail-flip-TTA + Cutout12/RandomErasing + one-cycle recipe, will reach `best_test_acc ≥ 96.48` (≥ +0.10pp over the 96.38 baseline) and beat a same-session DavidNet compiled control by more than the ~0.1-0.2pp noise floor, replicated on a confirmation pair — because WRN's documented under-cutout ceiling (~97.1%) lies above the DavidNet family's ceiling that every prior within-architecture lever failed to breach. If instead WRN ties at ~96.4 despite annealing ≥130 ep, the limiter is recipe/data-bound rather than backbone-bound, redirecting future loops off architecture entirely.
