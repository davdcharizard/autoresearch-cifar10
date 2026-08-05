# Brainstorm EXP-016
**Created**: 2026-06-30

<!-- Ideation only. Goal/metric/constraints live in goals/{slug}/01-definition.md; baseline in 04-results.tsv. -->

## Web Search & Literature Review

- **Ghost Batch Normalization** (Hoffer et al. 2017 "Train longer, generalize better", arXiv:1705.08741; David Page "How to train your ResNet" / DavidNet, myrtle.ai)
  Splitting a large batch's BN statistics into small "ghost" sub-batches (e.g. 32 within a 512 batch) injects beneficial regularization NOISE into the activations and was a KEY trick in the DavidNet fast-CIFAR recipe this goal is built on. Mechanism = activation-statistic noise — orthogonal to input-aug / weight-decay / label-smoothing / loss-geometry (all saturated here). Our net runs batch 512 with FULL-batch BN (the low-noise regime), so ghosting is a genuinely untested regularization axis on essentially the same architecture David Page tuned. (web search 2026-06-30: GBN "added beneficial regularization noise" on DavidNet.)
- **BlurPool / anti-aliased downsampling** (Zhang, ICML 2019 "Making Convolutional Networks Shift-Invariant Again", arXiv:1904.11486)
  Strided pooling/conv aliases (violates shift-invariance); inserting a low-pass blur before subsampling restores it and gives ~+0.5–1pp + robustness on ImageNet ResNets. A DIFFERENT inductive-bias lever (not capacity/reg). Our net has 3×MaxPool2d(2)+MaxPool2d(4) — all aggressive aliasing downsamplers. Risk: on 32×32 with already-aggressive pooling, blurring may wash out signal.
- **Squeeze-and-Excitation** (Hu et al., CVPR 2018, arXiv:1709.01507)
  Per-channel attention (global-avg-pool → 2 FC → sigmoid gate). +0.5–1pp on ResNets at ~free params; a different KIND of capacity (channel recalibration) than the conv-width/depth that saturated. Adds a small per-step cost (under-anneal watch).
- **Existing knowledge**: `knowledge/references/fast-cifar10-recipes.md` (DavidNet/airbench lineage); `policy-augmentation.md`, `mixing-augmentation.md` (input-aug lane now saturated, EXP-015).

## Experimental History Review

- **Current best 96.38 (EXP-008)**; **9 straight no-improvements (EXP-006→015)** → strong generalization-ceiling diagnosis (~96.3–96.5).
- **Saturated axes (do NOT retry the same approach)**: input augmentation across ALL 3 mechanisms — occlusion (EXP-008 won), mixing (CutMix EXP-011 tied), transform/policy (RandAugment EXP-015 tied); conv capacity — width@8×8 (EXP-007/014 saturated), depth@4×4 (EXP-005); optimizer (Muon EXP-009/010); regularization SCALARS — wd-shaping + LS retune (EXP-012); loss-geometry (tail-SAM EXP-013); eval-side TTA (EXP-006); buying epochs (compile EXP-014).
- **What worked & is in the base recipe**: DavidNet+one-cycle (EXP-001), EMA+flip-TTA (EXP-002), ZCA whitening conv (EXP-003), 1 ReZero block@layer2 (EXP-004), strong occlusion aug (EXP-008).
- **Key meta-finding (EXP-014/015)**: the ceiling is NOT epoch/throughput-bound and NOT movable by more aug/capacity/reg-scalars → the analyses explicitly call to PIVOT to a different MECHANISM / inductive bias / backbone.
- **Untried gaps**: (a) a different REGULARIZATION mechanism — normalization-statistic noise (GhostBN); (b) different INDUCTIVE BIASES — anti-aliased downsampling (BlurPool), channel attention (SE); (c) input-representation upgrades (bigger whitening); (d) schedule SHAPE (throughput-free, low-EV).
- **Protocol constraints**: ~0.1pp noise floor → SAME-SESSION baseline + clearly >0.1pp; fixed seed; watch `num_epochs` (~142–155); CPU-side cost inflates WALL not epochs, GPU per-step cost cuts epochs (under-anneal) — read num_epochs first on any compute-adding block (EXP-005/007/013).

## Diagnose What Limits the Objective
The metric is at a **generalization ceiling** for the CURRENT whitened-ResNet-9 at 300s (~96.4), established across 9 no-improvements that collectively rule out epoch-budget, conv-capacity, input-augmentation (all 3 mechanisms), optimizer, regularization-scalars, loss-geometry, and eval-TTA. The limiter is therefore the architecture's effective function class / inductive bias and the regularization MECHANISMS it employs — NOT any quantity already swept. The productive move (per EXP-014/015 Next Steps) is an intervention ORTHOGONAL to every saturated axis: a new regularization mechanism (activation-noise) or a new inductive bias (shift-invariance, channel attention). These are cheap, surgical, in-train.py changes — preferred over a wholesale backbone swap that would most likely lose to the highly-tuned 96.38 baseline.

## Collected Ideas

- **(lit/reg-mechanism)** GhostBatchNorm: replace `nn.BatchNorm2d` in `conv_bn` with a ghost-BN that computes train stats over sub-batches of ~32–128 (of the 512 batch) → regularizing activation noise (David Page).
- **(lit/inductive-bias)** BlurPool: replace MaxPool2d(2)/(4) with blur-low-pass-then-subsample (anti-aliased) for shift-invariance (Zhang 2019).
- **(lit/capacity-type)** Squeeze-Excitation channel attention on the residual blocks (Hu 2018).
- **(orthogonal/throughput-free)** Cosine anneal / longer low-LR tail replacing the linear triangular decay.
- **(representation)** Bigger frozen whitening front-end (5×5 patches → more whitened channels) for a richer linear input basis.
- **(simplification)** concat-pool (avg+max) before the FC head instead of max-only global pool.
- **(moonshot/history-recombine)** GhostBN × EMA: GBN's noisier per-step iterates are exactly what the existing weight-EMA denoises → the two may compound (EMA realizes the regularization benefit without the noise cost at eval).

## Combinations

- **GhostBN + EMA (already in recipe)**: GBN injects iterate noise for regularization; the EXP-002 weight-EMA averages it out at eval → plausibly captures GBN's generalization benefit while cancelling its eval-time variance. Strongest synergy; near-zero extra cost.
- **GhostBN + cosine tail**: noisier BN early + a smoother low-LR tail where accuracy concentrates → the anneal cleanly settles the GBN-regularized net.
- **BlurPool + GhostBN**: orthogonal axes (inductive bias + reg-noise) → could stack if either alone helps; but test singly first to attribute.

## Candidate Ideas

### 1. Ghost Batch Normalization (regularizing activation-statistic noise)
**Summary**: Replace the `nn.BatchNorm2d` inside `conv_bn` (7 BN layers) with a Ghost-BatchNorm that, during TRAINING, computes normalization statistics over small ghost sub-batches (split the 512 batch into groups of `ghost_size` ∈ {32, 64, 128}) instead of the full batch — injecting regularizing noise into the normalized activations — while accumulating a single set of running stats for eval (standard BN at inference). Implement as a drop-in module exposing `running_mean`/`running_var` buffers so the existing `AveragedModel(use_buffers=True)` EMA keeps working. Test a small same-session sweep of `ghost_size` (e.g. 64 and 128) vs a baseline control. Near-throughput-free (a reshape + BN over groups), so `num_epochs` should stay ~150.
**What it targets**: The generalization ceiling via a regularization MECHANISM never tried here — activation-statistic noise — orthogonal to the saturated input-aug / weight-decay / label-smoothing / loss-geometry axes (`03-experiment-learnings.md`; `project-insights.md` diminishing-returns entry, which is about SAME-axis saturation). The net runs full-batch (512) BN = the LOW-noise regime; David Page found ghosting essentially THIS architecture added real accuracy.
**Reasoning**: (a) GBN is a documented key trick of the DavidNet recipe this goal descends from, grounded on the same net/dataset; (b) it is a genuinely different regularization axis, so the diminishing-returns law (which closed input-aug) does not directly apply; (c) it composes naturally with the existing weight-EMA (the EMA denoises GBN's noisier iterates — combination A); (d) cheap and surgical (train.py-only, no new deps, no LR retune expected).
**Sources**: Hoffer et al. 2017 (arXiv:1705.08741); David Page "How to train your ResNet"; `knowledge/references/fast-cifar10-recipes.md`. Proposal: `proposals/idea-01.md`.
**Estimated Effort**: low-medium (a custom GhostBatchNorm2d + wire into conv_bn; env-toggle ghost_size; verify EMA buffer compatibility + eval uses running stats).
**Risk Assessment**: PRIMARY — it is still "more regularization" on a net that is already heavily regularized (LS 0.2 + Cutout12 + RE + EMA); the activation-noise increment may be redundant and tie (like the 2nd/3rd aug), landing within noise. SECONDARY — implementation traps: train/eval stat handling, running-stat momentum scaling with ghost count, EMA buffer-averaging interaction, bf16 stat stability. TERTIARY — too-small ghost_size (e.g. 32) could OVER-regularize/destabilize at 150ep (watch ep25 + num_epochs). Mitigate with a mild ghost_size sweep (64/128) and same-session control.

### 2. BlurPool (anti-aliased downsampling — shift-invariance inductive bias)
**Summary**: Replace the three `MaxPool2d(2)` (and possibly the final `MaxPool2d(4)`) with anti-aliased downsampling: MaxPool(stride 1) → fixed Gaussian/binomial blur conv → stride-2 subsample (Zhang 2019). This restores approximate shift-invariance lost to aliased strided pooling — a DIFFERENT inductive bias than capacity/aug/reg. Fixed (non-learned) blur kernel, so few new params.
**What it targets**: The generalization ceiling via a better downsampling inductive bias (shift-invariance), distinct from every saturated quantity. Targets the architecture's function-class limiter directly.
**Reasoning**: Documented +0.5–1pp + robustness on ImageNet ResNets; CIFAR translation-augmentation (RandomCrop pad-4) means shift-invariance is directly relevant. Mechanistically orthogonal to all prior experiments.
**Sources**: Zhang ICML 2019 (arXiv:1904.11486). Proposal: developed inline.
**Estimated Effort**: medium (BlurPool module + replace 3–4 pool sites; tune blur kernel size).
**Risk Assessment**: On 32×32 with already-aggressive 2×/4× pooling, low-pass blurring may DESTROY fine detail and HURT (CIFAR ResNets are not always BlurPool-positive at small resolution). Adds a blur-conv per downsample → mild per-step GPU cost → possible epoch loss (under-anneal watch). Medium risk it ties-or-loses.

### 3. Squeeze-Excitation channel attention
**Summary**: Add SE blocks (global-avg-pool → FC reduce (ratio 8–16) → ReLU → FC expand → sigmoid → channel-wise gate) to the residual branches (`Residual`/`GatedResidual`). A different KIND of capacity — dynamic channel recalibration — than the conv width/depth that saturated.
**What it targets**: The generalization ceiling via channel-attention capacity, orthogonal to spatial-conv width/depth.
**Reasoning**: +0.5–1pp on ResNets at near-free params (Hu 2018); cheap; identity-friendly (can init the expand-FC so the gate starts ~1, ReZero-style, for a clean single-variable test).
**Sources**: Hu et al. CVPR 2018 (arXiv:1709.01507); `knowledge/references/rezero-identity-init.md` (for identity-init discipline). Proposal: developed inline.
**Estimated Effort**: medium (SE module + insert into blocks; identity-init the gate).
**Risk Assessment**: capacity-saturation finding (EXP-014) is a headwind — SE is a capacity add, and adds per-step cost (global-pool + 2 FC per block) → epoch loss / under-anneal risk. The channel-attention axis is different from width/depth, but the net may simply not be capacity-limited at all → tie. Medium-low confidence vs GBN.

## Review

Cross-model adversarial review (Codex) in `01-idea-review.md`. Scored: **GhostBatchNorm 7.5/6.5 (pick)** > BlurPool 5.5/7.0 > SE 4.5/5.5. Top concerns + resolutions:

1. **Bar is ≥96.48 at the noise floor → need same-session control + clearly-above-noise result.** Resolved: 3-cell same-session design (c0 standard-BN control / cA ghost 128 / cB ghost 64); absolute 96.48 bar (`proposals/idea-01.md`).
2. **BlurPool & SE pay the per-step GPU compute tax → trade away anneal epochs (under-anneal); matched-epoch lit gains don't transfer.** Resolved: GBN chosen precisely because it is near-throughput-free (reshape + grouped stats, no extra convs); the two compute-taxed ideas demoted. num_epochs is a verdict gate regardless.
3. **CRITICAL correctness: `AveragedModel(use_buffers=True)` would EMA-average NOISY ghost running-stats into eval.** Resolved: GBN normalizes per-ghost for TRAIN noise but updates the eval `running_mean/var` buffer from FULL-BATCH moments (clean), so EMA-averaging stays sound; plus a `GHOST_SIZE=512 ≡ nn.BatchNorm2d` numerical-equivalence smoke before the official run (`proposals/idea-01.md` Correctness design).
4. **GBN redundancy risk (just another within-noise regularizer).** Resolved: it is a DIFFERENT axis (activation-stat noise), so the same-axis diminishing-returns law doesn't directly apply; mitigate by sweeping ghost 128/64 and AVOIDING 32 first unless ep25 healthy. Honest null framing if it ties.
6. **SE "identity init" underspecified (plain sigmoid starts at 0.5, not identity).** Noted for any future SE attempt: use `2·sigmoid(·)` + zero-init final FC for an exact-identity gate. (SE not pursued this loop.)

## Idea Evaluation

Adopt the reviewer's pick — **GhostBatchNorm**. It is the only finalist that is (a) grounded on essentially THIS architecture/batch regime (David Page's DavidNet), (b) a genuinely untried regularization MECHANISM orthogonal to every saturated axis, and (c) near-throughput-free, sidestepping the under-anneal tax that handicaps BlurPool/SE under the 300s budget. BlurPool has a higher novelty ceiling but weak local evidence (32×32 detail-loss risk) and a compute penalty; SE sits on the already-closed capacity lane. No override. Full scored critique: `01-idea-review.md`.

## Chosen Idea
**Selected**: Idea 1 — Ghost Batch Normalization (regularizing activation-statistic noise), ghost_size sweep 128/64 vs standard-BN control (`proposals/idea-01.md`).

**Why this idea**:
The net is at a generalization ceiling that 9 experiments have shown is not movable by input-aug, conv-capacity, optimizer, regularization-scalars, loss-geometry, or epochs. GhostBN attacks it from a mechanism none of those touched — activation-statistic noise — and is the one such mechanism with direct prior evidence on this exact net family (a documented DavidNet trick). Our batch-512 full-batch BN is the low-noise regime GBN is designed to improve. It composes naturally with the existing weight-EMA (which denoises GBN's noisier iterates), is cheap and surgical (train.py-only, no new deps, no LR retune), and avoids the per-step compute tax that makes BlurPool/SE under-anneal at 300s. The reviewer ranked it the clear winner.

**Hypothesis**:
GhostBatchNorm at ghost_size 64–128 (vs full-batch 512) injects activation-statistic noise orthogonal to the saturated regularization axes and, composed with the existing EMA, raises `best_test_acc` to ≥96.48 over the same-session baseline at matched ~150 epochs (near-throughput-free), with ep25 within ~0.5pp of control and full anneal. If every GBN cell ties at healthy epochs/ep25, BN-noise regularization is redundant with the existing heavy regularization stack and the ceiling is not regularization-mechanism-movable — narrowing the remaining options to inductive-bias / backbone changes.
