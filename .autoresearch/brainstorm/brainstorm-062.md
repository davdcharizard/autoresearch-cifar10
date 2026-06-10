# Brainstorm EXP-062
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- (none consulted — no new external technique; this loop probes untouched in-recipe scalars. Warmup-length / gradient-clipping behavior is standard and well-understood; no high-signal source adds beyond the project's own LR experiments EXP-016/017/029.)

## Experimental History Review
- **Current best: 96.45 (EXP-054)** = k=4 WideResNet-20 + AugMix-p0.5 + Cutout16 + cosine peak0.2/warmup0.05/Nesterov/LS0.1/WD1e-4 + compile, 91 ep, dt 8ms.
- **62 experiments, 8 improvements. The plateau is mapped across EVERY major lever** (goal-learnings High Importance): augmentation (all policies AA/TA/RA/AugMix + sub-levers + both CPU/GPU delivery — EXP-060 closed the last policy), capacity (×4 directions, epoch wall = channel width), optimizer (family/objective/gradient-dynamics), LR-peak/schedule-shape (EXP-016/017/029, cosine-to-0 + peak0.2 optimal), normalization-as-regularizer (GhostBN), eval-time BN-statistics (EXP-061 Precise-BN HURT −1.6pp), residual scaling, head, batch (128 optimal), activation, regularizers (dropout/Mixup/CutMix/SE), weight-averaging (EMA/SWA), throughput→epochs (saturated ~91; EXP-045/046 +5 ep didn't help).
- **Genuinely UNTESTED knobs (all low-ceiling scalars)**: WARMUP_FRAC (0.05, never isolated — only PEAK_LR and schedule-shape were swept); gradient clipping (never tried); BN momentum/eps (never tuned); per-channel std normalization (std=(1,1,1) currently — but BN-absorbed → near-certain null, not worth a run).
- **Near-miss never combined with current best**: aug cooldown @0.10 (EXP-034, 96.26 on the OLD TA recipe) — but the augmentation family is closed and the recipe is wall-tight (EXP-061: eval-side overhead overran 600s).

## Candidate Ideas

### Idea 1 — WARMUP_FRAC isolation (longer warmup, 0.05 → 0.10)
- **Summary**: Single-variable change of `WARMUP_FRAC` 0.05 → 0.10 (linear-warmup fraction of the time budget before cosine anneal). Everything else byte-identical to EXP-054.
- **Reasoning**: WARMUP_FRAC is the one LR-schedule knob never isolated (EXP-016/017 swept PEAK_LR; EXP-029 swept schedule shape). With strong AugMix augmentation producing noisy early-batch gradients at the high peak LR (0.2), a longer warmup (~9 ep vs ~4.5 ep) could stabilize early training and reach a marginally better basin. Throughput-neutral, free, clean.
- **Sources**: train.py L24/L35-41 (warmup logic); EXP-016/017 (PEAK_LR), EXP-029 (schedule shape).
- **Estimated Effort**: Trivial (one constant).
- **Risk Assessment**: Low; failure mode no-improvement. Longer warmup eats into mid-training high-LR time — EXP-016/017 showed the LR regime is finely balanced (±0.05 peak hurt ~0.5pp), so the ceiling is near-noise and a small regression is possible. No-downside otherwise (no extra wall, no scope risk).

### Idea 2 — Gradient-norm clipping (clip_grad_norm_ at a high threshold)
- **Summary**: Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` between backward and optimizer.step(), at a permissive threshold (e.g. 5.0) that only clips outlier spikes.
- **Reasoning**: AugMix distorts ~50% of images; heavily-distorted batches can produce large-loss → large-gradient spikes during the long high-LR (0.2) plateau that perturb the converging weights. A high-threshold clip tames only those spikes, leaving normal steps untouched — potentially smoother convergence to a slightly better minimum.
- **Sources**: train.py L245-246 (loss.backward/step); standard practice.
- **Estimated Effort**: Trivial (one line; threshold a constant).
- **Risk Assessment**: Low; this net already trains stably (no divergence), so clipping is most likely a null. A too-low threshold would hurt (under-stepping) — mitigated by a permissive 5.0. Compiled forward unaffected (clip is on the eager param grads, outside the compiled graph). Near-noise ceiling.

### Idea 3 — Aug cooldown @0.10 on the AugMix recipe (combine near-miss + current best)
- **Summary**: Disable AugMix+Cutout for the final 10% of training (crop+flip only), porting EXP-034 (only ever on the TA recipe) onto the AugMix best.
- **Reasoning**: Lets weights AND BN stats jointly re-adapt to the clean distribution in the tail (the EXP-061 insight: joint adaptation is why cooldown ≠ BN-recalib-alone). Mechanism could compound with stronger aug.
- **Sources**: EXP-033/034/035; EXP-061 (joint-adaptation insight).
- **Estimated Effort**: Low.
- **Risk Assessment**: Low-moderate; augmentation family flagged closed, and the recipe is wall-tight (EXP-061 eval-side overhead overran 600s — but cooldown REMOVES work in the tail, so it's wall-NEUTRAL-to-faster, unlike BN-recalib). Expected lift small (~+0.04 on TA).

## Idea Evaluation
- **Evidence strength**: all three are low-evidence probes on an exhausted plateau. Idea 1 is the cleanest "never-isolated scalar"; Idea 3 has a within-noise +0.04 precedent (TA); Idea 2 is the most speculative.
- **Mechanism clarity**: Idea 1 clear-but-minor (warmup stability); Idea 2 plausible-but-this-net-is-stable; Idea 3 clear (joint clean-adaptation) but small.
- **Expected impact**: all near-noise. Idea 1 and Idea 3 have the clearest (small) mechanisms; Idea 2 is least likely to do anything on a stably-training net.
- **Risk profile**: Idea 1 cleanest (no wall/scope risk, throughput-neutral). Idea 3 is wall-neutral-to-faster (removes tail work) — actually SAFER on wall than the base recipe. Idea 2 risks under-stepping if threshold too low.
- **Feasibility**: all trivial.
- **Conclusion**: Lead with **Idea 1 (WARMUP_FRAC isolation)** — the single most-defensible untested scalar, throughput- and wall-neutral, zero scope/wall risk. It is explicitly the "remaining clean probe" flagged in prior reports. Idea 3 is the natural follow-up (and is wall-safe, unlike EXP-061). Honest expectation: near-noise on a deeply-mapped plateau; run per NEVER-STOP.

## Chosen Idea
- **Selected**: WARMUP_FRAC isolation — `WARMUP_FRAC` 0.05 → 0.10, all else byte-identical to EXP-054.
- **Why this idea**: It is the one LR-schedule scalar never isolated (PEAK_LR and schedule-shape were swept, warmup-length never), it is throughput- and wall-neutral (zero risk of the EXP-061 wall-overrun failure mode), and it is a clean single-variable probe. On a plateau where every structural lever is closed, an untested scalar is the most defensible principled long-shot.
- **Hypothesis**: Doubling the warmup fraction to 0.10 will raise best_test_acc to ≥ 96.55 (baseline 96.45 + 0.1pp) by stabilizing early training under noisy AugMix gradients at the high peak LR; given the finely-balanced LR regime (EXP-016/017), the most likely outcome is a within-noise null, but the probe is free and clean.
