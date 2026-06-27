# Brainstorm EXP-031
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Progressive resizing (fastai DAWNBench ImageNet recipe; MosaicML ResNet-50 recipes)** (background knowledge; mosaicml.com composer docs)
  Train the early portion of the budget at reduced input resolution, finish at full resolution. Crucially, this technique is NATIVELY a wall-clock-budget method — its claims are made at fixed time, not fixed epochs (MosaicML: quality-neutral at −30% wall time on ResNet-50/ImageNet). Every prior external transfer failure in this campaign (0-for-10) involved fixed-epoch/final-iterate evidence; this is the first candidate whose home regime MATCHES ours.
- **EXP-006 conversion law (in-project)**: throughput → metric conversion works at unchanged hyperparameters and per-shape numerics (+25 epochs = +0.48). Progressive resizing adds steps without touching any 32×32-phase kernel, constant, or the data pipeline (if downsampling is done on-GPU inside the step).
- **knowledge/README.md scan**: no existing entry on progressive resizing; cifar10-fast/airbench entries do not use it (their budgets are too short to amortize a switch).

## Experimental History Review

- **Current best**: 96.71 @ 1990397; bar 96.81; mean ≈96.57, σ ≈0.16 ⇒ true effect ≥ +0.3 needed.
- **25 consecutive misses (007–030).** All static axes closed: capacity, recipe constants, gradient noise, augmentation pressure, init, activations, kernel-tier throughput (EXP-021: "no numerics-preserving speedup identified"), pipeline-tier throughput (loader stalls uncharged), optimizer geometry, eval-time normalization (EXP-029 inverted), head topology (EXP-030: 6th deferral variant — gradient quality).
- **What progressive resizing uniquely offers**: it attacks the EPOCH count — the single currency with a measured POSITIVE conversion law (EXP-006) — via resolution, the one throughput lever not yet priced. At 24×24, conv FLOPs scale ×(24/32)²=0.5625; projected dt ~14–17ms → first half of the budget yields ~+3,500–4,000 extra steps ≈ +35–40 epoch-equivalents, by far the largest epoch delta available from any remaining idea (Muon paid −17; this pays +35).
- **Known hazards from history**: (a) mid-run distribution changes carry adaptation costs (EXP-025 switch-shock; though its failure mode — clean-data overfitting — does not apply here: augmentation and dataset stay constant); (b) BN running stats follow within ~7 batches of the switch (momentum 0.1) — the EXP-029 failure mode (stats/weights mismatch) self-heals here because training CONTINUES at the new resolution; (c) any charged mid-run compile/autotune stall must be pre-warmed away (EXP-006 startup-warmup pattern, extended to both shapes); (d) low-res learning quality is the real unknown — the formative high-LR phase learns at the wrong object scale.
- **Protocol notes**: watchdog thresholds must be phase-aware (low-res phase legit dt ~14–18ms); post-hoc profile must be computed per segment; expected epochs ~165–180.

## Candidate Ideas

### 1. Progressive resizing 24→32 (GPU in-step downsample, switch at 50% of budget)
**Summary**: Add constants `LOW_RES = 24`, `RES_SWITCH_FRAC = 0.5`. Inside the timed step, before the autocast block: `if progress < RES_SWITCH_FRAC: inputs = F.interpolate(inputs, size=LOW_RES, mode="bilinear")` (fp32, charged, ~0.1ms). Everything else — transforms, loader, schedule, optimizer, model — byte-identical to baseline. The model is fully convolutional with adaptive pooling (24→12→6 spatial flows work; the pad-shortcut's `::stride` slicing handles even sizes). Startup compile warmup runs 3 iterations at EACH shape (24 and 32) so both inductor graphs and cudnn.benchmark tunings land in startup. The 32×32 phase (final 50%, the anneal + plateau) is numerically identical to the baseline's regime.

**Reasoning**: The arithmetic is the largest available: ~+35–40 epoch-equivalents (vs +25 epochs = +0.48 in EXP-006), against an unknown low-res quality toll. MosaicML/fastai evidence says the toll is ≈0 at ImageNet scale IN THE SAME (time-budgeted) regime — the first regime-matched external evidence the campaign has had. The switch hazards are bounded by mechanism: BN batch stats adapt instantly, running stats within ~0.2s, augmentation/dataset/objective constant through the switch, and the entire second half (including the full anneal tail that EXP-025 proved load-bearing) runs exactly the baseline recipe at full resolution. Downsampling INSIDE the timed region keeps budget accounting honest (the speedup pays for its own interpolation).

**Sources**: brainstorm-030 § Candidate 2 (the documented fallback, now matured); exp-report-006 (conversion law); exp-report-025/029 (switch-hazard bounds); MosaicML composer progressive-resizing method card (background); train.py:215–239 (timed step), :185–198 (warmup).

**Estimated Effort**: medium — ~12 lines (constants, interpolate branch, dual-shape warmup) + phase-aware watchdog.

**Risk Assessment**: (a) Low-res toll is THE unknown: CIFAR images are information-dense at 32px; 24px may cost more than ImageNet's 160-vs-224 analogue — if the toll exceeds ~0.3, the epoch gain nets to zero (graceful no-improvement, and the resolution axis closes with a measured number). (b) A missed pre-warm shape ⇒ charged mid-run compile (~10–20s = catastrophic); mitigated by warming both shapes AND a switch-boundary check in the watchdog (dt spike right after 50%). (c) Eval-vs-train res mismatch during phase 1 depresses early evals — cosmetic under the max-statistic. (d) Two new constants (24, 0.5) chosen from external recipes, not tuned — a miss does not bracket the axis (interior points remain), flagged for analysis honesty.

### 2. LogSumExp (soft-max) pooling head
**Summary**: Replace avg pooling with temperature-controlled LogSumExp pooling, the smooth approximation of max — dense gradients fix EXP-030's diagnosed argmax-routing failure while still preserving peak evidence.

**Reasoning / why weak**: Directly repairs EXP-030's mechanism, but (a) LSE over 64 positions × 256 ch × 512 batch is a pointwise exp/log ladder — EXP-026 priced such ops at +1–4.5ms (~6–25 epochs); (b) zero pedigree in any budget regime; (c) introduces a temperature hyperparameter with no anchor. Three speculative knobs stacked on a just-measured −0.91 axis.

**Sources**: exp-report-030 § Unexplored Avenues; exp-report-026 (pointwise cost ladder).
**Estimated Effort**: low. **Risk Assessment**: high probability of repeating EXP-026's exact arithmetic (gain ≤ cost); would need the early-dt gate at ~24ms.

### 3. Entropy-targeted objective shaping (confidence penalty replacing label smoothing)
**Summary**: Swap LS 0.1 for an explicit output-entropy penalty (Pereyra et al. 2017) at matched regularization strength.

**Reasoning / why weak**: LS and confidence penalties are near-equivalent regularizers in practice (Pereyra's own results show comparable gains); the recipe's soft-target pressure point is already certified optimal (LS 0.1 within the closed pressure axis). This is a substitution on a closed axis dressed as a new objective — expected noise-band outcome.

**Sources**: goal-learnings § Patterns (pressure dose-response, both sides closed); arXiv 1701.06548 (background).
**Estimated Effort**: low. **Risk Assessment**: near-certain noise-band/no-improvement; weak information value (axis already closed).

## Idea Evaluation

**Evidence strength**: Idea 1 carries the campaign's first regime-MATCHED external evidence (time-budgeted recipes) plus the in-project conversion law; Ideas 2–3 lean on mechanisms the campaign has already priced negatively (pointwise cost; pressure substitution). **Mechanism clarity**: Idea 1's gain mechanism is exact arithmetic (steps = time/dt, dt = f(res²)) and its hazards are individually bounded by prior measurements; Idea 2's net sign depends on an unmeasured dt; Idea 3 has no plausible path to +0.3. **Expected impact**: Idea 1 is the only remaining candidate whose central estimate (+35–40 epochs ≈ +0.4–0.5 gross) EXCEEDS the bar with room for a moderate toll; Ideas 2–3 cap below the screen. **Risk profile**: Idea 1's failure modes are either graceful (toll eats the gain → measured closure of the resolution axis) or gated (compile stall → watchdog kill); Idea 2 likely gate-killed; Idea 3 wastes a loop. **Feasibility**: Idea 1 is ~12 lines plus protocol care — well within one loop.

## Chosen Idea
**Selected**: Idea 1 — Progressive resizing 24→32 (GPU in-step downsample, switch at 50%)

**Why this idea**:
It is the only remaining candidate whose expected gross effect clears the +0.3 screen: it attacks the epoch count — the one currency with a measured positive conversion law (EXP-006: +25 epochs = +0.48) — and its external evidence comes, for the first time in this campaign, from the SAME regime (fixed wall clock). Every hazard the history flags (switch shock, BN lag, charged compile, tail integrity) is either bounded by a prior measurement or engineered away (dual-shape warmup, in-step charged downsample, untouched full-res anneal tail).

**Hypothesis**:
Running the first 50% of the 300s budget at 24×24 (bilinear in-step downsample) yields dt ≈ 14–17ms in phase 1 → total ≈ 165–180 epochs (vs 139), with the full-res second half numerically identical to baseline. If the low-res quality toll is ≤ ~0.2 (the regime-matched evidence says ≈0), the extra ~35 epoch-equivalents convert at the EXP-006 rate into a plateau ≥ +0.3 over the baseline mean — best_test_acc ≥ 96.81. Falsifiable early: phase-1 dt ≥ 20ms kills the arithmetic (gate); a post-switch eval that fails to rejoin the baseline family trajectory within ~10 epochs signals the toll dominating (let it complete; the plateau decides).
