# Brainstorm EXP-004
**Created**: 2026-06-28

<!-- Ideation only. Goal/metric/constraints live in 01-definition.md; baseline (95.87%) in 04-results.tsv. -->

## Web Search & Literature Review

- **airbench: "94% on CIFAR-10 in 3.29 Seconds" (Keller Jordan, arXiv:2404.00498)** (`goals/maximize-cifar10-test-accuracy/knowledge/references/fast-cifar10-recipes.md`; https://arxiv.org/abs/2404.00498)
  The documented 94→96 accuracy step is driven by three changes: (1) a **bigger network** — airbench96 adds a third conv layer plus a **residual connection over the last two convs of each block** (10 conv layers total vs the shallower 94-variant), (2) a **retuned schedule** — peak LR reduced by ~**0.78×**, shorter warmup, decay-to-zero, and (3) **multi-crop TTA** (`tta_level=2`: horizontal mirror over {center, +1px, −1px} reflect-pad crops). Without TTA the family reaches 93.2/94.4/95.6%; multi-crop TTA carries the 96-config to 96.05% and is singled out in the paper as the one feature whose gain is NOT explained by epochs-to-94% — i.e. a genuinely orthogonal eval-side lever. Whitening is the enabling trick for the aggressive short-warmup schedule.
- **airbench96 reference source (`legacy/airbench96.py`)** (https://github.com/KellerJordan/cifar10-airbench/blob/master/legacy/airbench96.py)
  Concrete 96-config: 3 convs/block with `x = x + x0` residual over last two; widths whiten=24/block1=128/block2=384/block3=512; lr 9.0/1024, 37 epochs, batch 1024, warmup 10%, cutout 12, wd 0.012/1024, momentum 0.85; `infer_mirror_translate` = 0.5·mirror(center) + 0.5·mean(mirror(+1px), mirror(−1px)), 6 forward passes.

## Experimental History Review

- **EXP-001 (95.22%)**: DavidNet/ResNet-9 + time-based triangular one-cycle (peak 0.4) replacing ResNet-20+MultiStepLR → +3.65pp. Validated base recipe. Learning: budget is training-TIME (300s), size the schedule to time; most gain arrives in the low-LR tail.
- **EXP-002 (95.72%)**: weight EMA (0.998, use_buffers) + flip-TTA gated to final 20% → +0.50pp, ~free, orthogonal eval-side win. The flip half was an attributable +0.25pp tail step-up at the gate — direct in-codebase evidence the model has exploitable eval-time prediction variance.
- **EXP-003 (95.87%)**: frozen 3×3 ZCA patch-whitening first conv (off-budget eigendecomp, optimizer-excluded) → +0.15pp by accelerating early convergence (ep10 85.5% vs 81.6%); gain survived 9 fewer epochs. Magnitude modest because the fully-annealed 174-epoch regime compresses whitening's epoch-starved upside.
- **What worked**: stacking orthogonal levers — base recipe (capacity/schedule) → EMA+TTA (robustness/eval-variance) → whitening (input conditioning). Each composed additively.
- **What hasn't been tried (gaps)**: (a) **representational capacity** — every net so far is the same 8-conv DavidNet; EXP-003 explicitly deferred a capacity probe. (b) **schedule retune exploiting whitening** — EXP-003's pre-registered #1 next step (whitening conditions input → tolerate higher peak LR / shorter warmup); held fixed in EXP-003 for clean A/B. (c) **eval-side multi-crop TTA** — we do flip-only; airbench's `tta_level=2` adds translation crops. Muon optimizer remains deferred (higher risk/effort).
- **Diminishing returns**: +3.65 → +0.50 → +0.15pp. Approaching the documented ~96% ceiling for this 6.5M-param scale at 300s; the airbench-documented path to 96% is bigger net + retuned schedule + multi-crop TTA.

## Diagnosis — what limits the objective

The metric has no profilable single bottleneck (it's annealed test accuracy), so I diagnose from history and the codebase. The recipe is **fully annealed** (LR→0 by end; gains concentrate in the low-LR tail) and the cheap orthogonal eval-side/robustness levers (EMA, flip-TTA, whitening) are largely spent. Three plausible remaining limiters, each with airbench evidence:
1. **Representational capacity** — all four experiments ran the identical 8-conv DavidNet. The documented 95→96 step is a bigger net (airbench96: +1 conv + residual per block). Capacity is the one major lever never probed here; if the model's loss floor is capacity-limited, no schedule/TTA tweak reaches 96%.
2. **Schedule under-using whitening conditioning** — the one-cycle (peak 0.4, warmup 0.15) was tuned for un-whitened inputs. A whitened (better-conditioned) surface tolerates larger peak steps and needs less warmup; the current schedule may leave tail-annealing steps and exploration headroom on the table.
3. **Residual eval-time prediction variance** — flip-TTA captured the left/right component (+0.25pp at the gate); the model's MaxPool/global-pool head is not perfectly shift-invariant, so a translation component of variance remains uncaptured. airbench's multi-crop TTA targets exactly this.

These are genuinely different angles of attack (architecture / training-dynamics / eval-side), warranting a thorough brainstorm.

## Collected Ideas

- Add a residual block to layer2 (the only stage without one) → 10-conv net matching airbench96's depth, with the prescribed ~0.78× LR reduction. *(capacity / literature)*
- Multi-crop TTA: extend flip-only to mirror × {center, ±1px reflect-pad crops} inside forward() → airbench `tta_level=2`. *(eval-side orthogonal lever / literature)*
- Whitening-enabled one-cycle retune: raise PEAK_LR (0.4→0.45) + shorten warmup (PCT_START 0.15→0.10) to spend more steps in the productive tail. *(schedule / experimental history — EXP-003's pre-registered next step)*
- Widen channels (1.25× width) instead of depth — higher capacity but ~56% FLOP cost (much heavier epoch hit than depth-via-residual). *(capacity variant — dominated by depth idea on the capacity/throughput trade)*
- Muon optimizer port — change training dynamics. *(moonshot / deferred again: high risk/effort, needs a reference port + LR smoke; weaker evidence than the three above)*
- Identity/Dirac-init for the new residual block + whitening (airbench pairs whitening with identity init). *(rider on the capacity idea, deliberately out of scope to keep that change minimal)*
- Cutout size sweep (8→12, airbench96 uses 12). *(low-ceiling regularization tweak; folded into a future sweep, not a standalone experiment)*

## Combinations

- **Capacity (depth) + LR retune**: a deeper residual net is exactly what airbench pairs with the 0.78× LR cut — the LR reduction is a *required co-change* for stability of the deeper net, not an independent knob. This cross is idea-01 as proposed.
- **Multi-crop TTA + capacity**: airbench96 uses both together to hit 96.05%; they are orthogonal (eval-side vs architecture) and additive in principle. Kept as separate experiments for clean attribution — TTA can be banked on top of any future architecture.
- **LR retune + shorter warmup**: shortening warmup reallocates pre-productive ramp steps into the annealing tail where all gains live; a whitened net no longer needs the long warmup, making the reallocation near-free on the early side. This cross is idea-03 as proposed.

## Candidate Ideas

### 1. Depth bump: add a Residual block to layer2 + 0.78× LR retune (airbench96-style capacity)
**Summary**: Add a channel/spatial-preserving `Residual(256)` block to `layer2` — currently the only stage without one — bringing the net from 8 to 10 learnable convs (matching airbench96's conv count), and reduce `PEAK_LR` 0.40→0.31 (the documented airbench94→96 factor of 0.78×) to keep the deeper residual stack stable. Whitening + EMA + flip-TTA stay byte-identical. The added block lives at 8×8 resolution (cheapest place to add depth: ~4× cheaper per channel than layer1's 16×16 convs), so the FLOP increase is ~12–18%, projecting ~145–155 epochs (from 174). A throughput-only smoke probe gates the run: proceed only if projected epochs ≥ ~130. (`proposals/idea-01.md`)

**What it targets**: Limiter #1, representational capacity — the never-probed lever. Mechanism: more conv layers in the mid stages → richer deeper feature hierarchy at 8×8 → lower achievable annealed loss floor → higher best_test_acc, *conditional on* the deeper net still completing enough low-LR tail steps. Whitening's measured early-convergence lead (EXP-003) partially buys back the lost epochs.

**Reasoning**: airbench96 (arXiv:2404.00498) reaches 96.05% precisely by adding a third conv + residual per block with ~0.78× LR — the documented, validated 95→96 step. We port the cheapest slice (one residual block) onto a recipe that already banks EMA+TTA+whitening (+0.65pp of free scaffolding). VRAM is non-binding (1.6GB/98GB), so capacity is free on memory.

**Sources**: `proposals/idea-01.md`; airbench arXiv:2404.00498 + `legacy/airbench96.py`; `knowledge/references/fast-cifar10-recipes.md`; EXP-003 analysis (whitening offsets epoch cost); EXP-001 insight (VRAM free).

**Estimated Effort**: low — 1 line added (`Residual(256)`), 1 line changed (`PEAK_LR`), no new modules/deps; one throughput smoke + one official 300s run.

**Risk Assessment**: Highest ceiling, highest variance. Strongest risk: **under-annealing** — spending ~20 tail epochs (where all prior gains live) to buy 2 conv layers; if the deeper net needs proportionally more steps, the tail reads under-trained and best_test_acc lands ≤95.87%. Secondary: depth+LR is a 2-variable change (a no-improvement can't cleanly separate "capacity doesn't help" from "0.31 mistuned for our smaller delta"). Tertiary: kaiming-init residual is a near-no-op early and may not warm up within a shorter budget (airbench uses identity init — deferred rider). Central estimate ~95.95–96.0%, right at the bar.

### 2. Multi-crop TTA: mirror × {center, ±1px reflect-pad crops} (airbench tta_level=2)
**Summary**: Replace the 2-pass flip-only TTA inside `ResNet9.forward` with airbench's `infer_mirror_translate`: average logits over horizontal mirror applied to 3 reflect-pad crops {[0:32,0:32], center, [2:34,2:34]} = 6 forward passes, weighted 0.5·mirror(center) + 0.5·mean(mirror(±1px)). Pure eval-side change inside forward() — still one `evaluator.evaluate` per epoch (the constraint is on evaluate() calls, not forward passes). Keep the final-20% gating (`TTA_START_FRAC=0.8`) to bound eval wall-clock. Training, schedule, EMA, whitening, seed all byte-identical. (`proposals/idea-02.md`)

**What it targets**: Limiter #3, residual eval-time prediction variance — specifically the translation component that flip-TTA leaves uncaptured. Mechanism: a MaxPool/global-pool net is not perfectly shift-invariant; averaging logits over ±1px crops cancels the spatial-position component of prediction variance → fewer borderline examples misclassify → higher correct count in the unchanged eval loop.

**Reasoning**: airbench singles out multi-crop TTA as a genuinely orthogonal eval-side lever (gain not explained by epochs-to-94%); `tta_level=2` carries airbench96 to 96.05%. EXP-002's clean +0.25pp flip step-up is direct in-codebase proof this model has exploitable eval-time variance and that view-averaging converts it to accuracy here. Lowest-risk lever (no training change, can't destabilize).

**Sources**: `proposals/idea-02.md`; airbench `legacy/airbench96.py` `infer()` + arXiv:2404.00498 §3.5; EXP-002 analysis (flip step-up); prepare.py `Eval.evaluate` calls `model(inputs)`.

**Estimated Effort**: low — localized rewrite of forward() + small `_mirror` helper; no new constants/deps. Care item: pre-run smoke of total wall (<600s); 4-pass fallback (center+one shift) if it runs hot.

**Risk Assessment**: Lowest risk, but **marginal gain may be <0.1pp**. The airbench 96.05% headline is vs NO TTA; we already bank the flip component (the larger half). Training-time RandomCrop(pad=4) already builds shift-invariance, possibly shrinking the translate headroom. Honest central estimate ~95.95% (+0.08pp), ~45–55% probability of clearing the bar. Secondary: reflect-pad border mildly off-distribution (low at pad=1); eval runs fp32 (no bf16), so passes are slower than a bf16 estimate — reinforcing wall-clock caution.

### 3. Whitening-enabled one-cycle retune: PEAK_LR 0.4→0.45 + PCT_START 0.15→0.10
**Summary**: Exploit the ZCA-whitened (better-conditioned) loss surface with a tight two-knob one-cycle retune — modestly raise the peak (`PEAK_LR` 0.40→0.45) and shorten the warmup (`PCT_START` 0.15→0.10), reallocating pre-productive ramp steps into the annealing tail where all prior gains concentrate. Forced secondary edit: `EMA_WARMUP_FRAC` 0.15→0.10 to keep "EMA starts at ramp completion" invariant intact. No architecture change, ~174 epochs unchanged. (`proposals/idea-03.md`)

**What it targets**: Limiter #2, schedule under-using whitening conditioning. Mechanism: whitening sphereizes the surface (EXP-003 confirmed early-convergence lead) → tolerates a larger peak step without instability → flatter/lower basin; shorter warmup → more steps in the EMA-denoised, TTA-read low-LR tail that historically produced the gains.

**Reasoning**: This is EXP-003's explicit pre-registered #1 next step ("a better-conditioned input tolerates larger steps; the one-cycle peak could go above 0.4 for a compounding gain"). airbench's 96-config also shortens warmup + decays to zero for its whitened net. 0.45 is squarely inside the documented stable one-cycle band (~0.4–0.6 mean-loss convention). Cheapest lever — single 300s run, near-free probe of remaining schedule headroom.

**Sources**: `proposals/idea-03.md`; EXP-003 analysis Unexplored Avenues/Next Steps; EXP-001/002/003 tail-is-the-lever evidence; airbench schedule notes; `fast-cifar10-recipes.md`.

**Estimated Effort**: low — 2–3 constant edits, one 300s run, no new code paths.

**Risk Assessment**: Cheapest but plausibly **lowest ceiling**. EXP-003 already runs a fully-annealed 174-epoch schedule whose own analysis warned whitening's benefit compresses in this regime — the conditioning headroom may already be largely consumed by 0.4/174ep. Central estimate +0.05–0.15pp sits *right at* the bar, and a single fixed-seed run has ~±0.05–0.1pp time-schedule noise that could mask a true <0.1pp effect or fake a pass. Mechanism trace (faster early loss? tail starts earlier/higher?) must be the real evidence, not just the final number. Secondary: higher peak → noisier tail partially offsetting EMA (mitigated by the small 0.45 raise).

## Review

Cross-model adversarial review by Codex (full text in `01-idea-review.md`). Scored verdict: **Idea-01 7/8 (evidence/impact) — picked**; Idea-02 8/5; Idea-03 4/4. Top concerns and resolutions:

- **(Fatal to Idea-03) Schedule mechanics error.** The decay slope in `train.py:264-268` is `PEAK_LR/(1-PCT_START)·(1-p)`. The proposed change makes it `0.45/0.90=0.50` vs current `0.40/0.85=0.47` — i.e. LR is *higher* throughout the decay, not lower, so Idea-03's core "more low-LR tail" mechanism is mathematically wrong. → **Idea-03 dropped as written.** (A clean shorter-warmup-at-fixed-peak retune remains a possible future probe, but it is not this loop's lead.)
- **(Idea-01) The 0.78× LR import is confounded/over-aggressive.** airbench's 0.78× accompanies a *full* block-tripling + width + GELU + cutout + warmup + TTA change, not a single added `Residual(256)`; transferring it raw may under-step our net. → **Resolution: drop the raw 0.78× cut. Keep `PEAK_LR=0.4` and instead zero-init the new residual branch's final BN γ (identity init)** so the deeper net *starts identical to the proven EXP-003 net* and stability needs no LR retune. This removes the depth+LR confound entirely (single-variable capacity test) and is airbench-faithful (airbench uses identity/Dirac init for added convs).
- **(Idea-01) Real failure mode is under-annealing, and a throughput smoke only checks wall time, not convergence.** → **Resolution: identity init directly mitigates this** (the block earns capacity gradually from identity, never disrupting the annealed tail), and the plan will add an **early-trajectory convergence check vs EXP-003** (ep1/ep10/ep25 + tail-crossing epoch), not just a wall-time smoke.
- **(Idea-01) Title/spec inconsistency** ("extra conv to layer2/3 stems" vs the actual single `Residual(256)` edit). → Resolution: the chosen change is **exactly one added `Residual(256)` block in layer2**; the misleading title phrasing is discarded.
- **(Idea-02) No flip-only safety floor + unverified fp32 eval wall-clock.** Valid but marginal (impact 5/10), and replacing flip-TTA means a regression if translate crops hurt. Banked as the next experiment to stack on top of whatever architecture wins.

## Idea Evaluation

Adopting the reviewer's pick, **Idea-01 (capacity via a layer2 residual block)**, refined per the feedback. It scored highest on the union of evidence (7) and impact (8) and is the only candidate attacking the never-probed limiter (representational capacity) with a documented path to 96% — the others are eval/schedule polish with central estimates pinned right at the noise floor of the +0.1pp bar. Idea-03 is removed (broken mechanism); Idea-02 is sound but low-ceiling and is queued to compose on top of the capacity result in a later loop. Full scored critique in `01-idea-review.md`.

## Chosen Idea
**Selected**: Depth bump — add an **identity-initialized residual block** to `layer2`, with **`PEAK_LR` held at 0.4** (no LR retune).

> **Implementation note (see `02-plan.md` Design correction):** the identity init is realized via **ReZero** (a learnable scalar gate `α`=0, `x + α·c2(c1(x))`), NOT by zeroing the residual branch's final BatchNorm γ. The plan-phase adversarial review caught that the BN-zero variant is **dead** in this codebase — `conv_bn`'s post-BN ReLU makes `ReLU(0)=0` with zero derivative, so no gradient would ever reach the block and it would stay identity forever. ReZero preserves the exact "starts identity, earns capacity gradually" intent with a live gradient path. The hypothesis below is unchanged.

**Why this idea**:
Representational capacity is the one major lever untouched across EXP-001→003, and airbench (arXiv:2404.00498) documents that the 95→96 step is precisely an added conv + residual per block. The refinement resolves the review's two substantive concerns at once: identity-initializing the new block (zero-init final BN γ) means the deeper net *starts bit-equivalent to the proven 95.87% EXP-003 net*, so (a) it needs no LR cut → no depth+LR confound and a clean single-variable capacity test at the validated `PEAK_LR=0.4`, and (b) it cannot disrupt the low-LR annealed tail where all prior gains live — it only adds capacity as training proceeds, directly defusing the under-annealing failure mode. The block lives at 8×8 resolution (~12–18% FLOPs, projected ~150 epochs), VRAM is non-binding, and whitening + EMA + flip-TTA stay on for free. Whitening's measured early-convergence lead (EXP-003) partially buys back the fewer epochs.

**Hypothesis**:
Adding an identity-initialized `Residual(256)` block to layer2 (8→10 learnable convs, `PEAK_LR=0.4` unchanged) lifts `best_test_acc` from 95.87% to **~95.95–96.1%** (central ~96.0%), clearing the ≥95.97% bar, *provided* the deeper net at ~150 epochs reaches a lower annealed loss floor than the 8-conv net at 174 epochs. Mechanism is falsifiable on the trajectory: the identity-init block should make early epochs (ep1/ep10/ep25) **match EXP-003 within noise** (no early disruption, unlike a kaiming-init block) while the annealed tail settles **higher**. If instead the tail lands ≤95.87%, capacity is not binding at this scale/budget (under-annealing dominates) — a clean negative result.

