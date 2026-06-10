# Brainstorm EXP-061
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- **Precise BN / BN recalibration (Ioffe 2017 "Batch Renormalization"; used in detectron2, RegNet/pycls training)** (well-established practice): after training, BN *running* statistics (an EMA of batch stats) can mismatch the true population statistics seen at eval. The standard fix recomputes BN running mean/var with a forward-only pass over data drawn from the EVAL-time input distribution. The mismatch is amplified when training-time inputs differ systematically from eval inputs — exactly the case under heavy input augmentation.
- **Cutout (DeVries & Taylor 2017)** (in-project, EXP-003): zeros a contiguous 16×16 region — ~25% of a 32×32 image — on EVERY training image. This drags the per-channel input MEAN toward the dataset mean-fill and inflates VARIANCE vs clean images. Since `mean,std=(...,(1,1,1))` (mean-only normalization, no /std), these shifts pass into conv1→bn1 and propagate, biasing every BN layer's running stats relative to clean eval inputs.
- **AugMix (in-project, EXP-054)**: applies heavy photometric/spatial distortion to ~50% of images, further shifting the training-time activation distribution that BN running stats track.

## Experimental History Review
- **Current best: 96.45 (EXP-054)** = k=4 WideResNet-20 + CPU AugMix-p0.5 + GPU Cutout(16) + cosine peak0.2/Nesterov/LS0.1/WD1e-4 + compile, 91 ep, dt 8ms, converged (loss 0.197).
- **The plateau is mapped across EVERY standard lever** (per goal-learnings High Importance): augmentation (ALL policies AA/TA/RA/AugMix + ALL sub-levers, EXP-060 closed the last policy), capacity (×4 directions, epoch wall), optimizer (family/objective/gradient-dynamics), LR/schedule, normalization-as-regularizer (GhostBN), residual scaling, head, batch, activation, throughput→epochs (saturated ~91 ep).
- **BN-statistics handling is NOT fully closed.** goal-learnings dismisses "Precise-BN" as "near-certain negligible (~35k converged BN updates)" — but that reasoning is about CONVERGENCE/noise, NOT about the augmentation-induced DISTRIBUTION SHIFT (train inputs heavily occluded/distorted, eval inputs clean). These are distinct mechanisms.
- Weight-averaging experiments (EXP-019/020 SWA) did a "BN-recomputed tail eval", but for a DIFFERENT purpose: the SWA-AVERAGED weights had never accumulated valid BN stats, so recompute was mandatory to make them usable (and used the training/augmented loader). Recomputing BN on CLEAN data for the NORMAL SGD iterate — to de-bias eval-time stats — is untested.
- **Near-misses available to combine**: aug cooldown @0.10 (EXP-034, 96.26 on the OLD TA recipe, never on AugMix); WARMUP_FRAC never isolated.

## Candidate Ideas

### Idea 1 — Clean-data BN recalibration before eval in the final epochs
- **Summary**: In the last few epochs (e.g. final ~3, detected by `total_training_time/TIME_BUDGET_S > 1−frac`), before the per-epoch `evaluate()` call, recompute every BN layer's running mean/var with a forward-only pass over UN-augmented training images (crop+flip+normalize only — NO AugMix, NO Cutout) on the eager `model`. Implement `recalibrate_bn(model, clean_loader, n_batches~40)`: set BN momentum=None (cumulative), reset running stats + num_batches_tracked, run `model.train()` forward under no_grad+autocast over the clean batches, restore momentum; then `evaluate()` as usual. One eval per epoch preserved (recalibration precedes the existing single eval).
- **Reasoning**: Cutout zeros ~25% of pixels on every training image and AugMix distorts ~50%, so BN running stats (an EMA of augmented batch stats) systematically mismatch the clean eval distribution. At eval, BN normalizes clean activations with augmented-distribution stats → mis-normalized inputs to the learned γ,β. Recomputing stats on clean data restores proper normalization to match what γ,β expect, directly targeting an UNCORRECTED train→eval gap. Crucially, `evaluate()` and the recompute run OUTSIDE the `total_training_time` (Σdt) timer (see train.py L275 — eval is after the dt loop), so this costs ZERO training budget and ~0 epochs; only a tiny WALL cost (~40 fwd passes/epoch × few epochs ≈ <1s).
- **Sources**: Precise-BN practice (above); train.py L221–282 (loop/eval structure, Cutout L231, BN in `model`); EXP-019/020 (SWA BN-recompute, different purpose); goal-learnings Precise-BN dismissal (convergence-framed, not distribution-shift).
- **Estimated Effort**: Low-moderate — a ~15-line helper + a second clean DataLoader + a tail-epoch guard. Runs on the eager model (no cudagraph issue).
- **Risk Assessment**: Low-risk failure mode (no-improvement). Real risks: (a) effect could be negligible if BN's affine params already absorb the bias → null; (b) WALL: the AugMix recipe is wall-tight (~593s, EXP-054) — keep recalibration to the last ~3 epochs with ~40 batches (<1s) and gate the wall early; (c) BN momentum/reset bug — smoke-test that running stats change and restore. Not reward hacking: standard training-side technique, frozen `evaluate()` untouched, no TTA.

### Idea 2 — Aug cooldown @0.10 on the current AugMix recipe (combine near-miss + current best)
- **Summary**: Disable AugMix + Cutout for the final 10% of training (crop+flip only), porting the EXP-034 cooldown (only ever tested on the older TA recipe) onto the EXP-054 AugMix best.
- **Reasoning**: EXP-034 cooldown @0.10 gave +0.04 over its TA baseline; its mechanism (let the model fit the clean test distribution in the tail) should COMPOUND with stronger aug — AugMix opens a larger train↔test gap than TA, so more clean-tail headroom. Also implicitly fixes the BN mismatch (Idea 1) in the tail, but at the cost of trading aug-epochs. Throughput-neutral.
- **Sources**: EXP-033/034/035 (cooldown sweep), EXP-054 (current best).
- **Estimated Effort**: Low — frac-based switch to drop aug in the final 10%.
- **Risk Assessment**: Low; flagged under the closed augmentation family (but never on AugMix). Expected lift small (~+0.04), likely below bar.

### Idea 3 — WARMUP_FRAC isolation (cheap scalar probe)
- **Summary**: Single-variable sweep of WARMUP_FRAC (0.05 → e.g. 0.02 or 0.10), the one schedule knob never isolated (PEAK_LR and schedule-shape are settled).
- **Reasoning**: Shorter warmup → more time at high LR (more effective exploration); longer → more stable early. A clean, cheap, throughput-neutral probe.
- **Sources**: train.py L24/L35–41; EXP-016/017 (PEAK_LR), EXP-029 (schedule shape).
- **Estimated Effort**: Trivial (one constant).
- **Risk Assessment**: Low; near-noise ceiling (warmup is a minor knob on a BN+tuned-cosine recipe).

## Idea Evaluation
- **Evidence strength**: Idea 1 has the most concrete, mechanism-specific support — Precise-BN is established practice, and the Cutout-zeros-25%-of-every-image fact makes the train→eval BN mismatch demonstrably systematic (not speculative). Idea 2's evidence is a within-noise +0.04. Idea 3 is a minor knob.
- **Mechanism clarity**: Idea 1 — sharp and falsifiable (de-bias eval-time BN stats; if affine params already compensate → null). Idea 2 — clear but small. Idea 3 — vague/minor.
- **Expected impact**: Idea 1 highest AND cheapest — it is the only candidate targeting a genuinely UNCORRECTED gap rather than re-probing a closed axis, and it costs ~0 epochs (the others are also cheap but lower-ceiling). If the Cutout/AugMix BN bias is material, +0.1–0.3 is plausible.
- **Risk profile**: All fail gracefully. Idea 1's only real risk is wall (mitigated by tail-only + few batches + early gate) and a BN-handling bug (mitigated by smoke test).
- **Feasibility**: Idea 1 moderate but well-scoped; runs on eager model (no compile interaction); eval/recompute are off the Σdt timer.
- **Conclusion**: Idea 1 is the lead — the freshest, most mechanism-grounded, near-zero-cost probe, and the one that exploits a structural property of the current recipe (heavy occlusion/distortion vs clean eval) that no prior experiment addressed. Ideas 2/3 are fallbacks.

## Chosen Idea
- **Selected**: Clean-data BN recalibration before eval in the final epochs — recompute BN running mean/var on un-augmented (no-Cutout, no-AugMix) training images on the eager model, in the last ~3 epochs, preceding each epoch's single `evaluate()` call.
- **Why this idea**: It is the one remaining lever that targets a genuinely UNCORRECTED train→eval gap rather than re-probing a closed axis: Cutout zeros ~25% of every training image and AugMix distorts ~50%, so the BN running stats (used only at eval) are biased toward an augmented/occluded distribution that eval never sees. Recomputing on clean data is standard Precise-BN practice, costs ~0 training budget (eval/recompute are off the Σdt timer), and is distinct from the convergence-framed Precise-BN dismissal and from the SWA BN-recompute (averaged weights, augmented data).
- **Hypothesis**: Recomputing BN running statistics on clean (un-augmented) training images before eval in the final epochs will raise best_test_acc to ≥ 96.55 (baseline 96.45 + 0.1pp), because it corrects the systematic Cutout/AugMix-induced bias in the eval-time BN normalization at ~zero epoch cost; if the learned BN affine parameters already absorb this bias, the result will be a within-noise null.
