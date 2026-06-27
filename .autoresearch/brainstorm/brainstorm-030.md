# Brainstorm EXP-030
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **cifar10-fast / "How to Train Your ResNet" (David Page)** (knowledge/README.md References row; github.com/davidcpage/cifar10-fast)
  The record-setting short-budget ResNet-9 uses a **concat(avg-pool, max-pool) global head** ("concat pooling") feeding the classifier — max-pooling preserves peak-activation evidence that average pooling dilutes; one of the few head-level modernizations with short-budget CIFAR pedigree. No isolated ablation magnitude is recorded in our knowledge base; fastai's adaptive concat pooling is the same construct, long defaulted in their vision stack.
- **In-scope file re-read (TASK.md, prepare.py — this loop)**: TASK.md explicitly allows "model architecture … model size, model type"; Eval is fp32 eager, `model.eval()`, batch 256, plain CE+argmax — the evaluated object must remain the plain model (TTA-style wrappers stay rejected per brainstorm-029). prepare.py offers no new legitimate angle.
- No new external fetches — remaining ideation rests on in-project measured evidence.

## Experimental History Review

- **Current best**: 96.71 @ 1990397; bar 96.81; σ ≈ 0.16, baseline mean ≈ 96.57 ⇒ true effect ≥ +0.3 needed (EXP-027).
- **24 consecutive misses (007–029).** Closed axes now include: capacity (uniform/realloc/shortcuts/depth-trades), every recipe constant (bracketed), gradient noise (both directions), augmentation pressure (both directions + time-varying), init (both directions), activations (cost ladder), throughput (kernel tier EXP-021; pipeline tier structurally — loader stalls are uncharged, brainstorm-029), optimizer geometry (Muon), EMA/smoothing, FixRes tail, **and eval-time BN stat substitution (EXP-029, −10.93 inverted: normalization constants are calibrated parts of the function)**.
- **Five structural laws now screen candidates**: free in early heat; free in epochs (dt); training-numerics equivalent; gradient-noise neutral; normalization-constant preserving. Plus the max-statistic lens: only converged-plateau LEVEL pays (EXP-011/016/028).
- **Untried gaps that survive all five laws**: (a) head topology — the ONLY remaining pedigree'd micro-architecture change (concat pooling); (b) multi-resolution budget reshaping (progressive resizing) — radical, multi-mechanism; (c) objective-shaping aux losses — weak priors at depth 20. The candidate pool is thin and mostly sub-screen; this loop's value is partly informational (closing the head axis cleanly).

## Candidate Ideas

### 1. Concat avg+max global pooling head (cifar10-fast pedigree)
**Summary**: Replace `F.adaptive_avg_pool2d(out, 1)` with the concatenation of adaptive avg-pool and adaptive max-pool over the final 8×8×256 feature map, doubling the fc input 256→512 (`fc = nn.Linear(2*w3, num_classes)`, params 2,570→5,130, total params +2,560). One extra pooling kernel per forward (~0.05ms, screened by the early-dt gate); Kaiming init on the new fc as the baseline already does.

**Reasoning**: The average over 64 spatial positions dilutes localized high-evidence activations (an object part firing strongly in 3 cells contributes 3/64 of its peak); the max pathway preserves exactly that evidence, and the classifier learns to weight both views. It is the only remaining architecture change with short-budget CIFAR pedigree (cifar10-fast ResNet-9, fastai default). Against the five laws: dt ≈ free (one pool + 2× tiny fc); early heat ≈ free — max-pooled features are informative from init (unlike EXP-018/020's learned-from-zero structures), and only 2,560 new params must be learned; numerics of every existing op unchanged; noise unchanged (same batches/objective); normalization constants untouched. The mechanism targets plateau LEVEL: a richer converged classifier on the same trunk.

**Sources**: knowledge/README.md (cifar10-fast row); train.py:106 (`adaptive_avg_pool2d`), train.py:82 (fc); EXP-026 protocol (early-dt gate); EXP-017/020 (early-heat pricing of learned-at-init structures — contrast case).

**Estimated Effort**: low — ~4 lines in train.py (forward + fc width).

**Risk Assessment**: Magnitude is the main risk: no recorded isolated ablation; prior +0.1–0.2 sits below the +0.3 screen — a noise-band outcome is the most likely result, in which case the head axis is closed for the cost of one clean run. Fails gracefully (worst case a small converged deficit, e.g., max-path noise hurting calibration). The graph change forces a fresh inductor compile — absorbed by the existing startup warmup. fc re-init costs a brief early-heat learning of the new half — bounded by its 2,560 params (vs EXP-020's failure with whole projection convs).

### 2. Progressive resizing (low-res early phase, full-res finish)
**Summary**: Train the first ~50–60% of the budget on 28×28 random-crops (FLOPs ×0.77 → dt ~17–18ms), then switch to the full 32×32 pipeline; pre-warm BOTH input shapes through torch.compile during startup. Buys ~+11% total steps (~+15 epochs).

**Reasoning**: The only remaining lever that adds epochs without touching kernels-at-a-given-shape (numerics law applies per-shape). fastai/MosaicML use it as a wall-clock win. But it is multi-mechanism: low-res training is itself a quality cost at fixed steps; the mid-run resolution switch shocks the (calibrated!) BN statistics — EXP-029 just measured how violently this recipe responds to stat/distribution perturbation, and EXP-025 showed mid-run distribution switches carry shock+adaptation costs. Net arithmetic: +15 epochs ≈ +0.25 minus an unknown low-res quality toll — likely sub-bar, with two failure modes and a complex compile story (a mid-run recompile inside the timed region would be a charged catastrophe if pre-warming misses).

**Sources**: brainstorm-029 (loader/budget accounting), EXP-025/029 (distribution-switch evidence); MosaicML ResNet recipes (background knowledge).

**Estimated Effort**: medium-high. **Risk Assessment**: multi-mechanism (violates single-variable discipline), BN-shock precedent, compile complexity; documented as the radical fallback if simple axes exhaust.

### 3. Auxiliary classifier head on stage 2 (deep supervision, small weight)
**Summary**: Add a tiny aux head (pool+fc on the 16×16×128 stage-2 output), loss = CE_main + 0.3·CE_aux, removed at eval.

**Reasoning / why weak**: Deep supervision pays at large depth (GoogLeNet-era, 22+ layers; modern use at 50+); at depth 20 with healthy gradient flow through residuals the auxiliary gradient adds little, while the aux backward costs real dt (extra backward through stages 1–2 ≈ +1–2ms → 6–12 epochs) and CHANGES the objective weighting (a heat/pressure perturbation of a certified-optimal recipe). Three of five laws at risk simultaneously.

**Sources**: EXP-026 (dt ladder), goal-learnings Patterns (recipe local optimum).

**Estimated Effort**: low-medium. **Risk Assessment**: expected negative by the same arithmetic that killed hardswish; included for completeness of the objective-shaping axis.

## Idea Evaluation

**Evidence strength**: Idea 1 has real (if unquantified) pedigree in the exact dataset and budget class (cifar10-fast); Idea 2 has pedigree only in regimes (ImageNet-scale) whose transfer record to this campaign is 0-for-8 (project-insights High); Idea 3 has pedigree only at much larger depths. **Mechanism clarity**: Idea 1's mechanism (preserve peak-activation evidence at the head; richer converged classifier) is clean and single-variable; Idea 2 is three mechanisms in one run (epoch gain, low-res toll, switch shock) — uninterpretable on failure; Idea 3's mechanism is absent at depth 20. **Expected impact**: Idea 1: 0 to +0.3 (unknown, the only candidate whose range touches the bar without a known offsetting cost); Idea 2: net ≈ 0 ± a lot; Idea 3: likely negative. **Risk profile**: Idea 1 fails gracefully and closes an axis cleanly; Idea 2 can fail three ways including a charged mid-run recompile. **Feasibility**: Idea 1 is a 4-line diff behind the standard gate battery.

After 24 misses the discipline that still pays is: probe each remaining law-compliant axis with the cheapest clean single-variable experiment. Idea 1 is exactly that; Idea 2 is the documented radical fallback.

## Chosen Idea
**Selected**: Idea 1 — Concat avg+max global pooling head

**Why this idea**:
It is the last untried architecture modernization with direct short-budget CIFAR-10 pedigree, the only remaining candidate that passes all five structural laws with no known offsetting cost (≈0 dt, init-informative features, 2,560 new params, no numerics/noise/normalization change), and a pure plateau-LEVEL intervention. A clean run either finds the head axis's gain or closes it — both outcomes move the campaign.

**Hypothesis**:
Concatenating max-pooled features to the averaged ones gives the classifier access to peak-activation evidence that average pooling dilutes, raising the converged plateau. Testable prediction: dt stays 22.4±0.5ms and epochs ≈139 (the change is throughput-free); the trajectory tracks the baseline family early (no deferral toll beyond the first ~2 epochs); the converged plateau (final-7 median) lands above the baseline mean by the head's true effect — success iff best_test_acc ≥ 96.81. If the plateau lands within ±0.15 of the mean (noise band), the head axis is closed: average pooling loses no usable evidence at this trunk width.
