# Experiment Report EXP-064: Gradient-norm clipping at a permissive threshold (max_norm=2.0)

- **Date**: 2026-06-09
- **Verdict**: no-improvement
- **Primary metric**: best_test_acc = **96.34%** (baseline 96.45, bar 96.55; delta **−0.11pp**)

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher-is-better) by editing only `train.py`, within the fixed 300s Σdt GPU-time budget on a single H20 (≤600s wall). Baseline 96.45 (EXP-054); bar = 96.55.

## Idea & Hypothesis
**Chosen idea**: Insert `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)` between backward and step — the last untested gradient-side knob. **Rationale**: AugMix distorts ~50% of images; heavily-distorted batches produce large-gradient spikes, and EXP-016/017 placed the recipe's peak LR (0.2) at/above its stability edge, so a permissive clip could tame those spikes and smooth convergence to a marginally better basin. **Hypothesis**: clipping raises best_test_acc to ≥96.55; honest prior expectation was a near-noise null (this net never diverged in 64 runs; optimizer/gradient-dynamics polish is a closed family per project-insights).

## Approach
One-line insertion (all else byte-identical to EXP-054): `clip_grad_norm_(model.parameters(), max_norm=2.0)` after `loss.backward()`, before `optimizer.step()`. The clip runs eagerly on the eager `model.parameters()` grads, OUTSIDE the compiled forward → cudagraph-safe (verified: dt stayed 8ms, no recompile). max_norm=2.0 chosen permissive to clip only true outlier steps.

## Execution
One run, GPU 1 (both GPUs idle at launch). Exit 0 in 574.1s wall, 91 epochs, 35218 steps. dt 609×8ms + 93×9ms + 1×10ms + 1×11ms (warmup) — uncontended, throughput identical to EXP-054 (the clip adds no measurable dt; modern clip_grad_norm_ avoids a host sync and the loop already syncs each step for timing). 0 NaN/error. No retries or adjustments.

## Results
best_test_acc 96.34% — a **−0.11pp regression** vs baseline 96.45, missing the bar by 0.21pp. The diagnostic signal is the loss: **final_test_loss 0.1939 < EXP-054's 0.1968** — clipping measurably LOWERED (better-conditioned) the loss but did NOT lift top-1. This is the **textbook polish-vs-top1 signature** that project-insights Medium documents for the entire optimizer/gradient-dynamics family: GC (EXP-031, loss 0.1894, top-1 flat), SAM (EXP-036), AdamW (EXP-043), PolyLoss (EXP-041, loss 0.158, top-1 flat) — every one lowered loss or improved conditioning without converting to accuracy. Gradient clipping behaves identically: bounding the largest gradient steps smooths the optimization trajectory (lower terminal loss = better-fit/calibrated) but the generalization bound is set by capacity + augmentation, not optimization smoothness, so top-1 doesn't move. The hypothesis (spike-taming → better basin) is refuted in the sense that mattered: there WAS a measurable optimization effect (loss ↓), but it did not transfer to test accuracy. This is the 65th experiment / 11th lever-perturbation to leave the 96.45 ceiling intact, and closes gradient clipping as the last gradient-side knob.

## Verification
- **Necessary condition 1 — `best_test_acc >= 96.55`**: 96.34 < 96.55 → **FAILED**. Stopped at first failed condition.
- Conditions 2 (in-budget) and 3 (no hard-constraint violation) not formally evaluated (aborted), but both would pass: total_seconds 574.1 < 600, num_params 4,299,866, num_epochs 91, 0 NaN, git diff == train.py only, clip_grad_norm_ is stock torch (no new dep), uncontended 8ms dt.
- **Trustworthiness**: fully trustworthy — clean uncontended run, throughput matched the reference, the loss-down/top-1-flat pattern is internally consistent and matches four prior polish-lever results. No false-failure / false-pass / integrity concerns.
- **Verdict basis**: valid in-budget run that missed the bar → **no-improvement**.

## Unexplored Avenues
- **Tighter clip (max_norm=1.0 or 0.5)**: would clip more steps, but EXP-016/017's finely-balanced LR regime suggests over-constraining gradients would under-step → regression; the loss-down/top-1-flat signature says the optimization is already fine, so a tighter clip targets a non-bottleneck — low confidence, likely regression.
- **Adaptive/per-layer clipping (AGC)**: more sophisticated, but the polish-vs-top1 pattern predicts the same null — the bottleneck is capacity/augmentation, not gradient conditioning. Low confidence.
- The gradient-side knob class (centralization EXP-031, clipping EXP-064) is now closed; do not revisit.

## Next Steps
1. **Per-channel std normalization** (low confidence) — the std=(1,1,1) code oddity (brainstorm-064 Idea 2); genuinely untested but BN-absorbed and confined to conv1 → likely null; MUST first verify the eval transform shares train.py's normalization (else train/eval mismatch → invalid).
2. **A genuinely radical, compute-neutral architectural change** (low-medium confidence, per the NEVER-STOP "think harder" directive) — every scalar, schedule, augmentation, optimizer, normalization, and both near-miss combinations are now closed; the remaining headroom (if any) is a structural idea not yet tried that does NOT add dt (the epoch wall forbids any compute/layer addition). This is the only unexplored class left.
3. **Accept 96.45 as the robust k=4/300s ceiling** — 11 consecutive no-improvements this exploration phase, all near-noise; the plateau is exhaustively mapped. Continue probing per NEVER-STOP but with calibrated (low) expectations.

## Key Learning
Gradient clipping (max_norm=2.0) lowered final_test_loss to 0.1939 (< EXP-054's 0.1968) but left top-1 within-noise below baseline (96.34) — the textbook polish-vs-top1 signature shared by GC/SAM/AdamW/PolyLoss. Bounding gradient spikes is a real optimization effect that does NOT transfer to test accuracy on this capacity/augmentation-bound net. The gradient-side knob class is now closed.
