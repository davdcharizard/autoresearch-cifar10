# Squeeze-Excitation channel attention (Hu et al. 2018, CVPR, arXiv:1709.01507)

Standing reference for SE — content-adaptive per-channel recalibration. The first CHANNEL-ATTENTION lever on this goal (chosen EXP-019), a different functional form than capacity/optimizer/regularization/downsampling.

## What it is
Per residual block: **squeeze** (global-avg-pool each channel → [N,C,1,1]) → **excite** (2-layer bottleneck `C→C/r→C`, ReLU then a gate nonlinearity) → multiply each channel of the feature map by its scalar gate. Learns to emphasize/suppress channels conditioned on global image content. ~`2·C²/r` params/block (tiny). Reported ~0.5–1pp ImageNet top-1 across backbones at <1% params/FLOPs.

## Why it's a fresh axis here
Our net's accuracy axes are saturated (width/depth EXP-007/014, optimizer 009/010, input-aug 008/011/015, wd/LS 012, SAM 013, BN-noise 016/017, anti-aliasing 018). SE adds cross-channel dependency modeling — a NEW function form, not raw width. Compute = GAP reduction + two 1×1 convs on 1×1 spatial → near-throughput-free (avoids the under-anneal trap; standard ops, no fused-kernel break).

## Implementation on THIS harness (load-bearing)
- Place inside the residual branch, after `c2`, before the add.
- **Identity-init the gate** so the validated recipe is unperturbed at init: use `2*sigmoid(...)` with `fc2` ZERO-init → gate = 2·sigmoid(0) = 1.0 (exact identity). NOT a plain sigmoid (→0.5 gate shrinks the branch). The un-gated `Residual(128)/Residual(512)` blocks are NOT ReZero-protected, so this matters (EXP-019 reviewer concern). For `GatedResidual(256)` (ReZero α=0) SE sits inside the α-gate, already identity at init.
- `fc1/fc2` = 1×1 `nn.Conv2d` (trainable, optimizer picks up); kaiming-init then zero-init fc2 override.
- Eval/EMA: deterministic, train≡eval; `AveragedModel(use_buffers=True)` averages SE weights; flip-TTA fine (gate from GAP is flip-invariant). bf16/channels_last preserved.
- THROUGHPUT: pre-measure with a full-train-step probe; per-block GAP could add a CUDA-sync stall → gate num_epochs≥135; layer3-only fallback.

## Strength / placement
Start r=16 at layer2+layer3 (channel-rich, small-spatial → cheap GAP). Then r=8 or all-3-blocks as the 2nd operating point. Watch ep25 (≥ c0, not depressed by init) + full anneal.

## Status on this goal
TESTED EXP-019 → **no-improvement** (ties same-session control). cA (layer2+3, r=16) 96.39 vs c0 96.11 = +0.28pp (session 1) did NOT replicate: confirmation cAb 96.31 vs c0b 96.29 = +0.02pp; cA absolute never cleared 96.48. The session-1 c0 was a low host draw (the reviewer's modest-EV prior held). cB (all-3 SE) 96.21 < cA → adding layer1 SE is net-negative. Implementation validated: identity-init `2*sigmoid`+zero-fc2 worked exactly (ep25 ≥ c0), throughput-neutral (per-block GAP is an async reduction, no sync stall, 144ep). Do NOT re-run SE at other ratios/placements on this backbone; channel attention is redundant on this saturated net. Re-test only as a free rider on a materially different backbone. See experiments/019/04-analysis.md.

## Sources
- Hu et al. 2018 https://arxiv.org/abs/1709.01507 (CVPR)
