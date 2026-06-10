# Brainstorm EXP-006
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Izmailov et al. 2018, "Averaging Weights Leads to Wider Optima and Better Generalization" (SWA)** (well-known result; no external fetch needed — standard technique):
  Averaging weights along the SGD trajectory finds flatter minima that generalize better than the final SGD iterate. A running/exponential average of weights, evaluated instead of the raw iterate, is a near-free test-accuracy boost (~+0.2–0.7 pp on CIFAR ResNets). Mechanism is orthogonal to capacity and input-space regularization — it smooths the optimization endpoint.
- **Polyak & Juditsky 1992 / model-EMA practice (timm, MoCo, MAE training recipes)**:
  Exponential moving average (EMA) of model weights is standard in modern image-training recipes; reliably small positive gain, throughput-neutral (one in-place lerp per step), evaluated at eval time only.
- **Hu et al. 2018, "Squeeze-and-Excitation Networks"**:
  Channel-wise recalibration (global-avg-pool → 2 FC bottleneck → sigmoid gate) adds ~0.3–0.5 pp on CIFAR ResNets at small param/FLOP cost — but introduces several extra small kernels per block.

## Experimental History Review

Prior experiments on this goal (index TSV):
- EXP-000 (92.06): modern recipe (bf16 + channels_last + time-fraction cosine + Nesterov + label smoothing). Showed capacity, not budget, was the ceiling at k=1.
- EXP-001 (94.90, +2.84): widen to WRN k=4 {64,128,256}, 4.3M params + 1×1 projection shortcuts. Width was the dominant lever.
- EXP-002 (95.42, +0.52): Cutout(16) augmentation (per-sample CPU — became a dataloader bottleneck, 79→54 epochs).
- EXP-003 (96.00, +0.58): vectorized GPU Cutout → recovered 54→77 epochs. **Current best, commit f59de56.**
- EXP-004 (95.26, no-improvement): k=6 widening → compute-bound (~22ms/step), only 35 epochs → underfit. **k=4 is the capacity sweet spot.**
- EXP-005 (96.05, no-improvement): WD 1e-4→5e-4 → eval loss down (0.204→0.196) but acc only +0.05. **Regularization axis near-saturated.**

What worked: capacity (width to k=4) then regularization (Cutout) then efficiency (GPU Cutout buying epochs).
What didn't: more width (k=6, compute-bound) and more L2 (WD 5e-4, saturated).
Key gaps / signals:
- goal-learnings § Protocol: **epoch count jitters 65–77; sub-~0.2pp deltas are noise** → target ≥~0.3pp expected gain.
- goal-learnings § Patterns: efficiency (keep per-step ops vectorized/on-GPU, no `.item()` syncs) is itself an accuracy lever.
- project-insights: H20 has massive VRAM headroom (490 MB / 98 GB); 300s wall-clock is the only binding budget; at k=4 the regime is **memory/launch-bound** (small width → many small kernels dominate, GPU underutilized).
- EXP-005 Next Steps explicitly flagged: **a different axis** (weight averaging / architectural efficiency / more epochs via faster steps), peak-LR sweep, and noted regularization/width are at local ceilings.

Untried axes: **weight averaging (EMA/SWA) at eval time**, architectural recalibration (SE), peak-LR tuning, throughput-up via larger batch.

## Candidate Ideas

### 1. EMA weight averaging for evaluation
**Summary**: Maintain an exponential moving average of the model weights during training and evaluate the EMA copy (not the raw SGD iterate) each epoch, reporting `best_test_acc` from it. Implement with `torch.optim.swa_utils.AveragedModel` using `get_ema_multi_avg_fn(decay)` (both in core torch — **no new dependency**), with `use_buffers=True` so BN running stats are averaged alongside parameters (avoids a separate BN-recalibration pass that would cost budget). Update the EMA once per step after `optimizer.step()`; evaluate `ema_model` in the per-epoch eval. Decay ≈ 0.999 (effective window ~1000 steps ≈ 2–3 epochs at ~390 steps/epoch), chosen so the average tracks the cosine-annealed tail rather than stale high-LR weights.

**Reasoning**: This is the cleanest *different axis* the learnings call for — orthogonal to capacity and input/L2 regularization, both of which are saturated. SWA/EMA reliably finds flatter minima and gives a small-but-consistent test-acc gain on CIFAR ResNets. Critically it is **throughput-neutral**: one in-place `lerp_` over 4.3M params per step is trivial vs the conv forward/backward, and adds essentially zero kernel-launch pressure in this launch-bound regime (unlike SE). It cannot underfit and degrades gracefully to ≈ the raw model.

**Sources**: Izmailov 2018 (SWA); Polyak averaging; EXP-005 report § Next Steps (weight averaging flagged); goal-learnings § Protocol (need ≥0.3pp / fresh axis).

**Estimated Effort**: low (≈15 lines: build AveragedModel, one update call in loop, eval the EMA model).

**Risk Assessment**: With cosine LR annealed to ~0, the final raw iterate is already well-settled, so the EMA gain may be modest (~0.1–0.3pp) and could land within the noise band → no-improvement. Decay mistuning (too high → averages stale high-LR weights and *hurts*; too low → ≈ raw model). BN-buffer averaging via `use_buffers=True` is an approximation to a proper BN recompute, but standard and adequate. Worst case: graceful no-improvement; safe failure mode.

### 2. Squeeze-and-Excitation channel recalibration
**Summary**: Add an SE module to each `BasicBlock` — global average pool → FC(C→C/r) → ReLU → FC(C/r→C) → sigmoid → channel-wise scale of the block's conv output (before the residual add), reduction r=16. Keeps width k=4 (the sweet spot) but lets the network reweight channels adaptively.

**Reasoning**: A new axis — architectural *efficiency* of a fixed-width model rather than more width (which failed at k=6) or more regularization (saturated). SE gives ~0.3–0.5pp on CIFAR ResNets at tiny param cost, and VRAM is free.

**Sources**: Hu et al. 2018 (SENet); EXP-005 § Next Steps (architectural efficiency / SE flagged); project-insights (VRAM free).

**Estimated Effort**: medium (new module + integrate into all 18 blocks; verify shapes).

**Risk Assessment**: **Throughput risk is real here** — SE adds several small kernels (pool, 2 FCs, sigmoid, broadcast-mul) per block × 18 blocks. In the established memory/launch-bound regime, more small kernels can raise per-step time and cut epochs (the exact failure mode of EXP-002's CPU Cutout and EXP-004's k=6). If epochs drop materially, the SE gain may be erased. Higher attribution complexity than idea 1.

### 3. Peak-LR sweep on the wide net (0.3 and 0.15)
**Summary**: PEAK_LR=0.2 was inherited from the k=1 era and never retuned for the k=4 net. Try a higher (0.3) and/or lower (0.15) peak with the same warmup+cosine schedule.

**Reasoning**: LR is the least-tuned recipe knob on the wide model; an under/over-shot peak leaves accuracy on the table. Throughput-neutral, trivial change.

**Sources**: EXP-005 § Next Steps (peak-LR sweep flagged, "LR can outweigh WD").

**Estimated Effort**: low (one constant; but a real sweep is ≥2 runs).

**Risk Assessment**: Expected magnitude is small — the current schedule (warmup + cosine-to-0, label smoothing, BN) is already forgiving of LR, so the gain is likely sub-0.2pp (noise band). A single value is a coin-flip vs noise; a proper sweep costs multiple loops for low expected payoff.

## Idea Evaluation

All three respect the hard constraints (train.py-only, no new deps, single H20/300s, eval once/epoch, no seed hacking). The decision criteria are *expected gain above the ~0.2pp noise floor* and *risk profile in this launch-bound regime*.

- **Evidence strength**: EMA (idea 1) and SE (idea 2) both have solid literature; peak-LR (idea 3) has the weakest expected magnitude.
- **Mechanism clarity**: EMA — smooths the optimization endpoint to a flatter minimum; fully orthogonal to the saturated capacity/regularization axes. SE — adaptive channel reweighting adds representational efficiency at fixed width. Both clear.
- **Throughput / risk**: this is the deciding factor. The regime is memory/launch-bound (per project-insights and the k=6 / CPU-Cutout episodes), so kernel-launch count directly trades against epochs. EMA adds ~one fused op per step → ~zero launch pressure → epochs preserved → no risk of the EXP-004 underfit trap. SE adds many small kernels per block → genuine epoch-loss risk that could erase its own gain.
- **Expected impact**: SE's nominal magnitude (~0.3–0.5pp) is higher than EMA's (~0.1–0.3pp), but SE's gain is conditional on epochs holding, which is exactly what's uncertain here; EMA's smaller gain is far more robust.

EMA wins on risk-adjusted expected value: it is the lowest-risk, throughput-neutral, genuinely-different-axis move, and is the cleanest single-variable test (raw vs averaged weights, nothing else changes — strong attribution). SE is the strong follow-up if EMA lands in the noise. Peak-LR is deprioritized (low expected magnitude, sweep cost).

## Chosen Idea
**Selected**: EMA weight averaging for evaluation

**Why this idea**: It is the fresh axis the learnings explicitly call for, orthogonal to the saturated capacity and regularization axes, throughput-neutral in a launch-bound regime (so it cannot fall into the EXP-004/EXP-002 epoch-loss trap), uses only core torch (no new dependency), and is a clean single-variable change with strong attribution. Its failure mode is graceful (≈ raw model → no-improvement), never destabilizing.

**Hypothesis**: Evaluating an EMA copy of the weights (decay ≈ 0.999, BN buffers averaged) instead of the raw SGD iterate will raise `best_test_acc` above the current 96.00% baseline by clearing the +0.1pp bar (target ≥96.10%, expected ~96.1–96.4%), at unchanged epoch count (~77) and unchanged peak VRAM, because trajectory averaging settles into a flatter, better-generalizing minimum than the final iterate.
