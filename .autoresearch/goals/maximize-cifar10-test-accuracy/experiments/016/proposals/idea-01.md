# Proposal idea-01: Ghost Batch Normalization (regularizing activation-statistic noise)

**Cross-model review pick** (`01-idea-review.md`: GBN 7.5/6.5, the best-aligned candidate). Concerns folded in below.

## Core change (train.py only)
Add a `GhostBatchNorm2d(num_features, ghost_size)` module and use it in `conv_bn` in place of `nn.BatchNorm2d` (7 BN sites). Env-toggle `GHOST_SIZE` (0/512 = standard BN; 128, 64 = ghosted).

## Mechanism
Batch 512 with FULL-batch BN = the low-noise regime. Splitting BN's TRAINING statistics into ghost sub-batches of size g (512/g groups) makes each group normalize by its own noisier mean/var → injects regularizing activation noise. This is a regularization MECHANISM orthogonal to the saturated axes (input-aug occlusion/mixing/transform; weight-decay; label-smoothing; loss-geometry). Documented key trick of the DavidNet recipe this net descends from (Hoffer et al. 2017 arXiv:1705.08741; David Page "How to train your ResNet").

## Correctness design (folds review #3 — the load-bearing risk)
The existing `AveragedModel(use_buffers=True)` EMA averages BN `running_mean/var` into the eval model. If those running stats were the NOISY per-ghost stats, eval would be polluted. Therefore:
- **Train forward**: reshape [N,C,H,W] → view as [G, g, C, H, W], compute per-ghost mean/var over (g,H,W), normalize each ghost by its own stats (this is the regularization noise), then apply affine γ/β.
- **Running-stat update**: update `running_mean/running_var` from the **FULL-batch moments** (the correct population estimate over all 512), NOT the per-ghost stats — so eval buffers stay clean and EMA-averaging them is sound. (Decouples train-noise from eval-stat-quality.)
- **Eval forward**: standard BN using the (clean, EMA-averaged) running stats — identical code path to `nn.BatchNorm2d.eval()`.
- Expose `running_mean`, `running_var`, `weight`, `bias`, `num_batches_tracked` buffers/params with the SAME names/shapes as `nn.BatchNorm2d` so the EMA `use_buffers=True` copy and `.to(channels_last)` work unchanged.
- Edge: if N not divisible by g, the loader uses `drop_last=True` (always 512) so G=512/g is integer for g∈{64,128,256}; assert and fall back to full-batch if not.
- bf16: compute stats in fp32 (`.float()`) for stability, like standard BN under autocast.

## EQUIVALENCE SMOKE (review #3): with `GHOST_SIZE=512` the module must be numerically equivalent (within ~1e-3) to `nn.BatchNorm2d` on a fixed input in both train and eval mode — a unit test before the official run.

## Design — SAME-SESSION multi-cell (folds review #1, #4)
- **c0** baseline: standard BN (GHOST_SIZE=512) — same-session noise control (stored 96.38 too weak at the floor).
- **cA** GHOST_SIZE=128 (4 ghosts) — mild noise, the safest documented setting.
- **cB** GHOST_SIZE=64 (8 ghosts) — stronger noise. AVOID 32 first (review #4: only go smaller if ep25 stays healthy).
Each a separate `train.py` process, `CUDA_VISIBLE_DEVICES=1 timeout 600`, nvidia-smi logged before. GBN is near-throughput-free (reshape + grouped stats) → expect num_epochs ~150 (watch it; reject < ~142).

## Synergy with EMA (combination A)
GBN adds per-step iterate noise; the existing weight-EMA (decay 0.998) averages it → plausibly captures the regularization benefit while cancelling eval-time variance. No EMA change needed.

## Verification
- Best GBN cell `best_test_acc` ≥ **96.48** (baseline 96.38 + 0.1pp) AND clearly above same-session c0.
- num_epochs ~142–155 (throughput-free check); ep25 within ~0.5pp of c0 (over-regularization/instability watch) + fully annealed (best≈final).
- Integrity: train.py-only; prepare.py byte-unchanged; ≤1 eval/epoch; seed 42; equivalence smoke passed.
- ON A WIN: bake the winning GHOST_SIZE as the train.py default so bare `uv run train.py` reproduces it.

## Hypothesis
GhostBatchNorm at ghost_size 64–128 injects activation-statistic noise orthogonal to the saturated regularization axes and, composed with the existing EMA, lifts best_test_acc to ≥96.48 over the same-session baseline at matched ~150 epochs. If every GBN cell ties at healthy epochs/ep25, BN-noise regularization is redundant with the existing stack → the ceiling is not regularization-mechanism-movable.

## Effort: low-medium. Sources: Hoffer 2017 (arXiv:1705.08741); David Page How-to-train-your-ResNet; knowledge/references/fast-cifar10-recipes.md.
