# Brainstorm EXP-007
**Created**: 2026-06-28

## Web Search & Literature Review

- **cifar10-airbench `airbench96.py` architecture & recipe (Keller Jordan, arXiv:2404.00498)** (https://github.com/KellerJordan/cifar10-airbench/blob/master/legacy/airbench96.py; fetched 2026-06-28):
  The documented **96.03%** net (n=400 runs, ~35s on A100) uses: whitening conv (2×2, 24ch) → 3 ConvGroups of width **128 / 384 / 512**, each group = Conv→MaxPool→BN→GELU with **3 conv layers + a residual over the last two**, → MaxPool(3) → Linear(512→10, ×1/9). **GELU** throughout. Augmentation: flip + translate=4 + **cutout=12**. SGD-Nesterov, label_smoothing 0.2, batch 1024, **37 epochs**. Key contrasts vs our 96.00 net (64/128/256/512, ReLU, cutout 8, ~2 convs/group, ~150 epochs in 300s): airbench's middle is **wider (384 vs 256)**, uses **GELU**, **cutout 12**, and **3 convs/group**. Crucially, airbench reaches 96.03 in only 37 epochs / ~35s, while we fit ~150 epochs in 300s — **our epoch budget is far more generous, yet our net is smaller → we are capacity-bound, with budget to spend on a bigger net.**
- **fast-CIFAR lineage (David Page / hlb / airbench)** (knowledge/references/fast-cifar10-recipes.md):
  hlb-CIFAR10 also uses GELU. The lineage scales the NET (width/epochs) + whitening + TTA to climb 94→96; it does NOT use mixup (short-budget fast-training avoids it). This argues the documented path beyond ~96.0 is **more capacity**, not heavier augmentation schemes like mixup.

## Experimental History Review

Source: `04-results.tsv`, `03-experiment-learnings.md`, `project-insights.md`, prior `04-analysis.md`.

- **Best-first:** EXP-004 (**96.00**, current best): ReZero `GatedResidual(256)` at layer2/8×8 → +0.13pp. EXP-003 (95.87): whitening. EXP-002 (95.72): EMA + flip-TTA. EXP-001 (95.22): DavidNet base.
- **Recent failures:** EXP-005 (95.90, −0.10pp): 2nd ReZero block at layer3/**4×4** — coarse capacity unused AND 4×4/512 convs ~10% slower (cuDNN small-spatial kernels). EXP-006 (95.93, −0.07pp): multi-crop TTA — the views work (visible +0.28pp onset) but the increment-over-mirror is below the noise floor.
- **CRITICAL new finding (HIGH signal):** a **~0.1pp run-to-run NOISE FLOOR**. The time-budgeted loop fits a host-throughput-dependent step count (142/131/150 epochs across byte-identical-training runs as the shared host load varied), so the trained model — and `best_test_acc` — varies ±~0.1pp even with the seed fixed (no-seed-hacking forbids averaging it out). **The +0.1pp bar sits AT the noise floor → only changes with clearly >0.1pp expected headroom can register.** (03-experiment-learnings.md Protocol Findings; project-insights Medium.)
- **What worked:** capacity at the FINE/fast spatial stages (whitening, layer2/8×8) and eval-side EMA/TTA. **What's exhausted:** depth at layer3/4×4 (EXP-005); eval-side TTA polish (EXP-006).
- **Untried gaps:** widening (more channels, not depth) at the proven 8×8 stage toward airbench96's 384; activation (GELU) + stronger cutout (12) recipe alignment; a 2nd ReZero block at the proven layer2/8×8 stage.

## Diagnosis — What Limits the Objective

The metric is `best_test_acc` in a fixed 300s training budget. Two facts set the limiter:

1. **We are capacity-bound at the current net scale, near the documented ceiling.** Our 7.78M-param net sits at 96.00; airbench96's *wider* net (middle 384 vs our 256, GELU, 3 convs/group) reaches 96.03 in only 37 epochs. We fit ~150 epochs in 300s — **far more annealing budget than airbench96 uses** — yet score the same, which says the bottleneck is *model capacity*, not optimization/epochs. VRAM is a non-constraint (1.6/98 GB). The lineage's path beyond ~96.0 is a bigger net, and EXP-004 already proved added capacity at layer2/8×8 pays (+0.13pp). This is the limiter with the clearest >0.1pp headroom.
2. **The ~0.1pp noise floor is the *measurement* constraint.** It does not bound the achievable accuracy, but it bounds what a *single run* can prove: any change whose true effect is <0.1pp is indistinguishable from epoch-count jitter (this is exactly why EXP-006's real TTA gain still netted −0.07pp). So the next experiment must target a lever with clearly-larger-than-noise headroom — pointing at capacity scaling (limiter 1), not micro-optimization.

Conclusion: spend this loop on **adding model capacity at the throughput-cheap fast spatial stages (8×8/16×16)**, accepting some epoch loss, since EXP-004 showed the capacity gain there can outrun the epoch cost — and explicitly NOT at layer3/4×4 (EXP-005-failed) and NOT on sub-noise eval-side polish (EXP-006-exhausted).

## Collected Ideas
<!-- thorough pass; spanning capacity / recipe-alignment / proven-stage-depth angles -->

- Widen the proven 8×8 stage (layer2 256→384) toward airbench96 width. [capacity — primary]
- GELU activation + cutout 8→12 (throughput-free airbench96 recipe alignment). [cheap orthogonal]
- Second ReZero `GatedResidual(256)` at layer2/8×8 (proven-stage depth). [conservative capacity]
- (considered, rejected) Mixup/CutMix — the fast-CIFAR lineage avoids it in short budgets; low fit.
- (considered, deferred) Full airbench96 net port (GELU + 3-conv groups + 384 + 2×2 whiten) — biggest headroom but a large multi-variable change needing LR retune; revisit if single-lever capacity bets stall.

## Combinations
<!-- noted for future loops; kept separate now for clean single-variable attribution under the noise floor -->

- Widen-layer2 (Idea 1) + GELU/cutout12 (Idea 2): both point toward airbench96; stacking raises the chance of clearing the noise floor, but conflates capacity vs recipe attribution. Defer the cross until a single-lever winner is found, then graft the cheap throughput-free Idea-2 changes onto it.

## Candidate Ideas

### 1. Widen layer2 (8×8 stage) 256→384 toward airbench96 proportions
**Summary**: In `ResNet9.__init__`, widen the middle stage: `layer2` `conv_bn(128,256)`→`conv_bn(128,384)` and `GatedResidual(256)`→`GatedResidual(384)`; ripple `layer3` stem `conv_bn(256,512)`→`conv_bn(384,512)`. layer3 output stays 512 so `pool`/`fc=Linear(512,10)` are unchanged. Hold `PEAK_LR=0.4` (the new GatedResidual(384) is still ReZero identity-init, α=0, so no LR retune; the widened main-path kaiming convs are variance-invariant under kaiming). Clean single-variable capacity test mirroring EXP-004 at the same proven 8×8 stage, at airbench96's documented middle width. (proposals/idea-01.md)

**What it targets**: The capacity limiter near the ~96.0 ceiling (Diagnosis §1) — adds representational *width* where EXP-004 showed capacity is binding and kernels run at full speed, exploiting the generous epoch budget.

**Reasoning**: EXP-004's capacity add at layer2/8×8 outran its epoch cost (+0.13pp; capacity lead by ep25); airbench96 documents 384 as the working middle width; widening avoids the 4×4 cuDNN penalty that sank EXP-005. Width (vs depth) adds capacity without extra sequential conv latency.

**Sources**: proposals/idea-01.md; experiments/004/04-analysis.md; airbench96.py; knowledge/references/fast-cifar10-recipes.md.

**Estimated Effort**: low (≈3 line edits, one run).

**Risk Assessment**: Throughput cost is the main risk — layer2 conv FLOPs scale ∝ width² (256→384 = 2.25× those convs) + layer3 stem input 1.5×; expect epochs to drop from ~150 toward ~115–130. If the wider net under-anneals, the low-LR tail (where most accuracy lands) could fall below 96.00. The ~0.1pp noise floor means the capacity gain must clearly exceed 0.1pp; EXP-004's +0.13pp barely cleared it, so a 1.5× widen is a reasonable bet for a larger margin but not guaranteed. Fallback if borderline: 256→320 (recovers epochs).

### 2. GELU + cutout 12 (throughput-free airbench96 recipe alignment)
**Summary**: Two one-line edits: in `conv_bn`, `nn.ReLU(inplace=True)`→`nn.GELU()` (changes the activation network-wide — every Conv-BN-act unit, including inside Residual/GatedResidual); at the augmentation site, `Cutout(8)`→`Cutout(12)`. Both are airbench96/hlb values. No architecture-shape, schedule, optimizer, or LR change → **zero throughput cost** (same epoch count), sidestepping the capacity-vs-epochs tension. Recommended as ONE combined "cheap recipe-alignment" change. (proposals/idea-02.md)

**What it targets**: The optimization-smoothness + regularization gap vs the documented 96.03 recipe (Diagnosis §1, the non-capacity divergences) — throughput-free, so it cannot cost epochs like EXP-005 did.

**Reasoning**: airbench96 (96.03) and hlb both use GELU + cutout 12; these are the only remaining throughput-free divergences from the reference net. GELU's everywhere-nonzero gradient is a mild optimization smoother; cutout 12 is stronger regularization, absorbed by the fully-annealed ~150-epoch tail. Kaiming `nonlinearity="relu"` left as-is (BN renormalizes, so the init-gain mismatch is harmless and avoids a 2nd variable).

**Sources**: proposals/idea-02.md; knowledge/references/fast-cifar10-recipes.md; airbench96.py.

**Estimated Effort**: low (two-line diff, one run).

**Risk Assessment**: The dominant risk is the ~0.1pp noise floor: each change alone is plausibly <0.1pp and would be lost under epoch-count jitter (exactly EXP-006's fate). Combining GELU+cutout12 raises the chance the summed effect clears the bar, at the cost of per-change attribution (acceptable — both are documented airbench96 values). Residual risk: cutout 12 slightly under-fits within budget and cancels GELU's gain. Worst case: clean no-improvement, zero baseline risk (byte-reversible, no throughput change).

### 3. Second ReZero block at layer2 (8×8, proven stage)
**Summary**: One-token edit: append a second `GatedResidual(256)` to `layer2` → `nn.Sequential(conv_bn(128,256), MaxPool2d(2), GatedResidual(256), GatedResidual(256))`. Reuses the class verbatim; adds ~1.18M params (two 8×8/256 convs); ReZero α=0 keeps identity-init so `PEAK_LR=0.4` stays. Requires the 2-step trainability smoke (α.grad≠0 on step 1 is necessary-but-not-sufficient; verify branch-conv grad becomes nonzero on step 2 after α moves). (proposals/idea-03.md)

**What it targets**: Residual-capacity headroom at the one stage where added depth demonstrably paid off (EXP-004), via depth rather than width.

**Reasoning**: EXP-004 proved 8×8/256 ReZero capacity is real; EXP-005 proved *where not* (4×4). This places the 2nd block at the proven full-speed stage — the explicit "Unexplored Avenue #1" from EXP-005's report. Cleaner single-variable edit than widening (no shape ripple, ReZero-identity-init, no retune).

**Sources**: proposals/idea-03.md; experiments/004 & 005 analyses; project-insights ReZero entry.

**Estimated Effort**: low (one-token edit + smoke).

**Risk Assessment**: Diminishing returns is the primary risk — the EXP-006 reviewer flagged that the first block's +0.13pp does not imply a second pays; layer2 may be near its sweet spot, and after the epoch loss (two extra 8×8 convs → ~125–135 epochs) it could net-negative. The ~0.1pp noise floor compounds this: a thin capacity margin minus epoch loss may not clear the bar. Upside ceiling plausibly lower than a well-chosen width bump (Idea 1).

## Review
Cross-model adversarial review (Codex) → `01-idea-review.md`. Scored verdict: **Idea 1 (widen layer2) 8/8 — pick**; Idea 3 (2nd ReZero block) 7/5; Idea 2 (GELU+cutout12) 4/3. The reviewer: "Idea 1 is the only candidate with a credible path to clearly beating the ~0.1pp noise floor while directly attacking the diagnosed capacity bottleneck. Idea 3 is cleaner but has weaker marginal headroom; Idea 2 is too likely to disappear into measurement noise."

Top concerns + resolutions (fold into the plan):
1. **Under-annealing from throughput loss is the main failure mode.** Widening makes layer2's two 8×8 convs 2.25× and the layer3 stem 1.5× heavier; if epochs collapse, the low-LR-tail gain is lost. → Resolution: the plan will make `num_epochs` and `img/s` **first-class verdict evidence** (record them; if epochs fall well below ~115–130, attribute any shortfall to under-annealing), and pre-register the **256→320 fallback** to recover epochs if the 384 run under-anneals.
2. **airbench's 96.03 vs our 96.00 is only +0.03pp (below the noise floor) and confounded** (width + GELU + cutout12 + 3-conv groups). → Resolution: airbench justifies the **384 target width**, NOT a >0.1pp guarantee; the real headroom argument rests on EXP-004's measured +0.13pp from layer2 capacity and the capacity-bound diagnosis, not on matching airbench's number.
3. **"No LR retune" is plausible but not identity-preserving** (widening changes the main path at layer2[0]/layer3[0], unlike a ReZero add). → Resolution: hold `PEAK_LR=0.4` for clean attribution but watch the early trajectory for instability/slow fit; kaiming+BN makes width changes variance-stable, so retune is unlikely needed, but the early-epoch curve is the check.

## Idea Evaluation
Adopt the reviewer's verdict: **Idea 1, widen layer2 256→384**. It is the unique candidate aligned with the diagnosed capacity limiter that also has clearly-greater-than-noise expected headroom (the decisive filter this loop, given the ~0.1pp floor). Idea 3 (proven-stage depth) is the natural fallback if width disappoints (thinner headroom but cleaner); Idea 2 (GELU+cutout12) is demoted to a future throughput-free *rider* on a capacity win, not a standalone experiment. Full scored critique in `01-idea-review.md`.

## Chosen Idea
**Selected**: Widen layer2 (8×8 stage) 256→384 toward airbench96 proportions

**Why this idea**:
The diagnosis identifies model capacity as the binding limiter near the ~96.0 ceiling (we fit ~150 epochs — far more annealing budget than airbench96's 37 — yet score the same as a wider net, and EXP-004 measured +0.13pp from adding capacity at exactly this 8×8 stage). Among the candidates it is the only one with a credible >0.1pp-headroom path past the noise floor: it adds substantial *width* (1.5×) at the proven full-speed stage, avoids the 4×4 cuDNN penalty that sank EXP-005, and is a clean single-variable edit (hold PEAK_LR, ReZero block stays identity-init). The cross-model reviewer independently scored it highest and named it the pick. Its main risk (epoch loss → under-annealing) is observable and has a pre-registered 320-width fallback.

**Hypothesis**:
Widening layer2 from 256→384 channels (with the layer3 stem input rippled 256→384, everything else byte-identical and PEAK_LR=0.4 held) will raise `best_test_acc` from 96.00% to ≥96.10% (≥+0.1pp), because the added representational width at the capacity-binding 8×8 stage outruns the throughput-driven epoch loss (projected ~115–130 epochs vs ~150), reproducing EXP-004's capacity-beats-epochs result at larger magnitude. Falsifier: best < 96.10 — either the wider net under-anneals (epochs collapse and the low-LR tail falls short, visible in num_epochs/trajectory → try the 320 fallback) or layer2 width is capacity-saturated (epochs adequate but accuracy flat → capacity at this scale is exhausted, pivot to depth/other stages).
