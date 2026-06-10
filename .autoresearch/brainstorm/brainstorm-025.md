# Brainstorm EXP-025
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Smith, Kindermans, Ying, Le — "Don't Decay the Learning Rate, Increase the Batch Size" (ICLR 2018)** (https://arxiv.org/abs/1711.00489)
  Increasing the batch size during training yields the SAME training/test learning curves as decaying the LR, reaching **equivalent test accuracy after the same number of epochs but with FEWER parameter updates** (works for SGD, momentum, Nesterov, Adam). One can scale LR ∝ batch to further cut updates. KEY corollary for us: a larger batch reaches the same accuracy per epoch but each update consumes more images — so at a budget gated by per-step COMPUTE time, a larger batch (if step-time stays ~flat) processes more images per second of compute ⇒ MORE effective epochs in the 300s budget.
- **Goyal et al. — "Accurate, Large Minibatch SGD" (2017)** (well-known; linear scaling rule)
  Linear LR scaling rule: when multiplying batch by k, multiply LR by k, with a gradual warmup to avoid early instability. Validated to large batches on ImageNet ResNet; the warmup is essential at the higher peak LR.
- **He et al. — "Bag of Tricks for Image Classification" (CVPR 2019)** (knowledge: general)
  Catalogue of free (compute-neutral, param-neutral) convergence/accuracy tricks: zero-initialize the last BN γ in each residual block (blocks start as identity → cleaner early gradient flow), and exclude BN/bias params from weight decay ("no bias decay"). Each contributes a small consistent gain; cosine LR + label smoothing (already in our recipe) are also on the list.

## Experimental History Review

Current best **96.22%** (EXP-012, commit 6c417a4; TA+Cutout(16)+compile, 91 epochs, loss 0.195). Bar = 96.32 (+0.1pp). 24 experiments; the plateau is extremely well-mapped:

- **Closed scalar knobs (all interior optima):** LR-peak 0.2 (EXP-016/017), Cutout 16px (EXP-013/021), label-smoothing 0.1 (EXP-023), weight-decay (EXP-005). Pivot OFF scalar-knob tuning.
- **Closed augmentation family:** auto-aug policy saturated (TA≈RA, EXP-014), label-mixing underfits (Mixup/CutMix, EXP-011/018), occlusion-strength optimal at 16. TA+Cutout(16) is the aug ceiling.
- **Closed regularizer-ADDING:** WD↑, Mixup, CutMix, in-block dropout all regress/null — recipe is **convergence-bound, not overfit-bound** (project-insight Medium).
- **Closed capacity/structural-compute:** k≥5 width (EXP-004/009), pre-act blocks (EXP-015), BlurPool (EXP-024) all hit the **epoch wall** — any COMPUTE-ADDING change cuts epochs and under-trains (project-insight High, count 4).
- **Closed weight-averaging:** EMA/SWA approach cosine-to-0 from below, never beat it (EXP-006/019/020, count 3).

**Untried gaps (this loop targets them):** (1) **batch size** — held at 128 for all 24 runs, never varied; (2) **zero-init residual γ / selective weight-decay** (Bag-of-Tricks free convergence levers); (3) **per-channel input std-norm** (std=(1,1,1)→true CIFAR std, expected BN-absorbed null).

**KEY codebase insight (re-read of train.py L218-242):** the 300s budget is gated on `total_training_time`, which is the **sum of per-step COMPUTE time `dt`** (timer `t0` starts AFTER the dataloader yields the batch, line 218; `dt` covers H2D copy + fwd + bwd + step + `cuda.synchronize`). Dataloader wait is NOT counted. Consequence: the number of optimizer steps ≈ 300 / mean(dt), and **epochs = steps × batch / 50000**. So if `dt` stays ~flat when batch grows (the launch-bound regime — memory/insights say k=4 is launch-bound at ~8ms/step, VRAM only ~0.5GB of 98GB), a larger batch yields proportionally MORE effective epochs within the same compute budget. This makes batch size a uniquely clean throughput lever here — the inverse of the compute-ADDING changes that hit the epoch wall.

## Candidate Ideas

### 1. Large-batch throughput exploitation (batch 128→256 + linear LR scaling)
**Summary**: Raise `BATCH_SIZE` 128→256 and scale `PEAK_LR` 0.2→0.4 (linear scaling rule, Goyal 2017), with a modestly longer warmup (`WARMUP_FRAC` 0.05→0.08) for early stability at the higher peak. Add `persistent_workers=True` + `prefetch_factor=4` to the DataLoader purely as a wall-clock safeguard (keeps the data pipeline ahead of the now-faster-per-image GPU; does NOT affect the accuracy mechanism, which is compute-gated). The cosine schedule is time-fraction driven so it anneals fully over 300s regardless of batch/step count — no schedule retuning needed beyond the peak.

**Reasoning**: The budget is gated on per-step COMPUTE `dt`, and k=4 is launch-bound (~8ms/step, trivial VRAM). If `dt` stays ~flat as batch doubles, the 300s buys ~2× the images ⇒ ~2× effective epochs (e.g. 91→~150+). Smith 2018 shows a larger batch matches small-batch accuracy at equal epochs with fewer updates; here the EXTRA epochs from the throughput headroom are the upside — they help iff the strong-augmentation TA recipe is not yet epoch-saturated. Strong, diverse augmentation (TA) characteristically converges slowly and benefits from more passes, so unlike the pre-TA recipe (EXP-007 saturated at ~77 ep), the TA recipe plausibly still wants epochs. This is the single largest untested lever and the only one that can ADD effective training without hitting the compute/epoch wall.

**Sources**: Smith 2018 (https://arxiv.org/abs/1711.00489); Goyal 2017 linear scaling; train.py L218-242 budget mechanism; project-insights (launch-bound k=4, EXP-007 epoch-saturation caveat is pre-TA); knowledge/papers (to add: large-batch-scaling.md).

**Estimated Effort**: low (3 hyperparameter edits + 2 DataLoader kwargs; one training run).

**Risk Assessment**: Two ways to fail. (a) If the net is actually compute/memory-bound (not launch-bound) at batch 256, `dt` doubles → same images → same epochs but lower gradient noise + sharper minima ⇒ mild generalization regression (graceful no-improvement; ALSO cleanly resolves the long-standing launch-bound assumption). (b) Large-batch generalization gap even with LR scaling, OR the TA recipe IS epoch-saturated so extra epochs don't help. Wall-clock risk: more epochs ⇒ more per-epoch eval calls; mitigated by faster per-image throughput and persistent_workers — monitor total_seconds < 600. VRAM is a non-issue (256 ≈ 1GB of 98GB). Failure mode is no-improvement, not crash/invalid.

### 2. Bag-of-Tricks free convergence levers (zero-init residual γ + no-bias-decay)
**Summary**: Two compute-neutral, param-neutral best-practice changes (He et al. 2019): (a) after model init, zero-initialize `bn2.weight` (the last BN γ) in every `BasicBlock` so each residual branch starts at 0 → blocks begin as identity, improving early gradient flow/conditioning; (b) build the SGD optimizer with two param groups so weight decay applies ONLY to conv/linear weights, NOT to BN γ/β or biases ("no bias decay"). Everything else byte-identical.

**Reasoning**: The recipe is convergence-bound, so convergence-quality levers that add no compute are the right class (no epoch-wall risk — the confound that killed EXP-004/009/015/024). Both tricks are documented small consistent gains on ResNets and are untried here. They're synergistic standard practice; bundling them maximizes the chance of clearing the +0.1pp bar in one fair, throughput-neutral 91-epoch run.

**Sources**: He et al. "Bag of Tricks" (CVPR 2019); train.py L65-92 (BasicBlock, bn2), L110-115 (init), L192-198 (optimizer); project-insights (convergence-bound; compute-neutral changes are the safe class).

**Estimated Effort**: low (a small init loop + param-group split; one run).

**Risk Assessment**: ResNet-20 is shallow, so zero-init-γ's benefit (largest for deep nets) may be marginal (<0.1pp); no-bias-decay's effect is small because WD is already only 1e-4. Combined gain may fall below the noise floor (~0.2pp) → no-improvement. Bundling muddies attribution if it gains, but both are standard so that's acceptable. No regression risk beyond noise; compute-neutral so no epoch confound.

### 3. Per-channel input std-normalization
**Summary**: Change `std` from `(1,1,1)` to the true CIFAR-10 per-channel std `(0.2470, 0.2435, 0.2616)` in the `transforms.Normalize`, so inputs are unit-variance per channel (mean already subtracted).

**Reasoning**: The only remaining untouched scalar in the input pipeline; closes the input-normalization axis cleanly. Compute-neutral, zero confound risk.

**Sources**: train.py L152-155 (the `std=(1,1,1)` comment explicitly flags this); project-insights / memory (flagged as the last cheap probe, expected BN-absorbed null).

**Estimated Effort**: low (one-line edit; one run).

**Risk Assessment**: Expected NULL — the stem conv + first BatchNorm absorb any global per-channel affine rescaling of the input, so accuracy should be unchanged within noise. Low expected value; primarily an axis-closer.

## Idea Evaluation

All three respect the hard constraints (train.py-only, no new deps, single GPU, ≤1 eval/epoch, no seed hacking, <600s wall-clock) and none is a retry of a logged failed approach (batch size and zero-init-γ/no-bias-decay and input-std are all untried).

- **Evidence strength**: Idea 1 has the strongest *mechanism-specific* evidence tied to THIS project — the budget-is-compute-gated code fact + the launch-bound regime make batch scaling a genuine free-epochs lever, backed by canonical large-batch literature (Smith 2018, Goyal 2017). Idea 2 has solid but generic literature whose effect size shrinks on a shallow net with already-small WD. Idea 3 is an expected null.
- **Mechanism clarity**: Idea 1 is crisp — larger batch + flat dt ⇒ more images/compute-second ⇒ more effective epochs ⇒ gain iff TA recipe is epoch-hungry. Idea 2's mechanism (better init conditioning) is clear but its magnitude on ResNet-20 is doubtful. Idea 3's mechanism predicts no change.
- **Expected impact**: Idea 1 has by far the highest ceiling (could add ~50-60% more effective epochs to a strong-aug recipe that plausibly still wants them) and is the most INFORMATIVE — it definitively tests the load-bearing launch-bound/epoch-hunger assumptions the whole plateau conclusion rests on. Idea 2 targets the convergence-bound constraint safely but likely lands below the bar. Idea 3 is a closer.
- **Risk profile**: All fail gracefully to no-improvement (no crash/invalid path). Idea 1's worst case additionally yields a definitive launch-bound diagnosis — high information even on failure.
- **Feasibility**: All low-effort. Idea 1 needs wall-clock monitoring (more evals), handled by the abort criteria + persistent_workers safeguard.

Idea 1 dominates on ceiling, mechanism clarity, and information value, and is the natural pivot now that scalar-knob and compute-adding axes are exhausted. Ideas 2 and 3 are reserved for subsequent loops (2 is the next pick; 3 is the final axis-closer).

## Chosen Idea
**Selected**: Large-batch throughput exploitation (batch 128→256 + linear LR scaling 0.2→0.4 + warmup 0.05→0.08)

**Why this idea**:
It is the single largest untested lever and the only known way to ADD effective training within the fixed budget WITHOUT triggering the epoch wall that has killed every compute-adding structural change. The mechanism is grounded in a concrete code fact (the budget gates on per-step compute `dt`, not wall-clock or epoch count) plus the launch-bound regime (VRAM ~0.5GB/98GB, ~8ms/step), and is backed by canonical large-batch literature (Smith 2018: equal accuracy at equal epochs with fewer updates; Goyal 2017: linear LR scaling). Even a null is maximally informative — it resolves the launch-bound and TA-epoch-hunger assumptions that the "96.22 is the ceiling" conclusion depends on.

**Hypothesis**:
At batch 256 with PEAK_LR scaled to 0.4 and warmup extended to 0.08 of the budget, per-step `dt` stays ~flat (launch-bound) so the 300s compute budget fits markedly more effective epochs than the 91 at batch 128. Because the strong TrivialAugment recipe is not epoch-saturated (unlike the pre-TA recipe of EXP-007), the additional training converges to a better optimum and `best_test_acc` rises above the 96.32 bar — while total wall-clock stays < 600s and num_params is unchanged (4,299,866). Falsifiable: if `dt` roughly doubles (compute-bound) or epoch count does not rise / accuracy regresses, the launch-bound-free-epochs premise is wrong.
