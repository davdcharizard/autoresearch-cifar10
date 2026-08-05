# Proposal idea-02: Squeeze-Excitation channel attention — adaptive feature recalibration

## Core change (train.py only)
Insert a lightweight **Squeeze-Excitation (SE)** block (Hu et al. 2018, "Squeeze-and-Excitation Networks", arXiv:1709.01507) into the residual branches. SE adaptively recalibrates channel responses: global-average-pool each channel to a scalar (squeeze), pass through a 2-layer bottleneck MLP (`C → C/r → C`, ReLU then sigmoid), and multiply each channel by its gate (excite). Env `SE_RATIO` (reduction r, default 16) and `SE_MIN_CH` to gate which stages get SE.

```python
class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        cr = max(8, c // r)
        self.fc1 = nn.Conv2d(c, cr, 1); self.fc2 = nn.Conv2d(cr, c, 1)
    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)          # squeeze: GAP -> [N,C,1,1]
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s                               # excite: per-channel rescale
```
Wire into the `Residual`/`GatedResidual` branch after `c2` (standard SE-ResNet placement, before the residual add): `Residual.forward → x + self.se(self.c2(self.c1(x)))`. To keep the ReZero identity-init property of `GatedResidual` intact, SE sits INSIDE the α-gated branch (`x + alpha*se(c2(c1(x)))`), so the block is still exact identity at α=0. Apply at layer2 (256) and layer3 (512) first — the channel-rich stages where SE pays off most and the spatial maps are small (GAP is cheap).

## Mechanism — why this is a genuinely DIFFERENT lever
Every prior experiment adjusted capacity (width/depth), the optimizer, regularization strength, or the loss surface — none changed the *functional form* of how channels interact. SE adds **content-adaptive channel gating**: the network learns to emphasize/suppress feature channels conditioned on the global image content, an attention mechanism orthogonal to all saturated axes. It is a different representational capacity than raw width (EXP-007/014 showed width is saturated) — SE adds almost no width but a new *modeling capability* (cross-channel dependency), at ~`2·C²/r` params per block (tiny: ~16K at C=256, r=16).

## Why it targets the limiter
The limiter is the generalization ceiling (project-insights High, EXP-014). SE is one of the few proven ways to raise a convnet's accuracy WITHOUT adding meaningful width/depth — the original paper reports consistent top-1 gains across ResNet/Inception on ImageNet (~0.5–1pp) at <1% extra params/FLOPs. Because it adds negligible compute (a GAP + two 1×1 convs on a [N,C,1,1] tensor), it should be near-throughput-free, avoiding the under-anneal trap that sank capacity adds (EXP-005/007) and SAM (EXP-013).

## Throughput (pre-registered check)
SE's compute is dominated by the GAP reduction (a `mean` over H,W — a cheap reduction) and two 1×1 convs on a 1×1 spatial map (trivial). The GAP introduces a small reduction but no kernel-fusion break (unlike GhostBN EXP-016). Expectation: ≤~3% cost (≥~144 epochs). MUST be measured; if a cell under-anneals, restrict to layer3 only. Watch that the per-block GAP does not add a CUDA sync stall.

## Design — SAME-SESSION multi-cell
- c0: unchanged baseline — full-speed same-session anchor.
- cA: SE in layer2+layer3 residual branches, r=16 — PRIMARY.
- cB: SE in all three blocks (layer1+2+3) OR r=8 (stronger gating) — second operating point per the throughput smoke and cA's ep25.

## Correctness / EMA / ReZero / eval
- SE `fc1/fc2` are standard `nn.Conv2d` 1×1 → trainable, kaiming-init via the existing `_weights_init` (Conv2d branch). Optimizer picks them up automatically.
- ReZero preserved: SE lives inside the α-gated branch of `GatedResidual`, so α=0 ⇒ exact identity at init (one-step backward smoke confirms α.grad≠0 and SE grads finite).
- Eval/EMA: SE is deterministic, train≡eval; `AveragedModel(use_buffers=True)` averages the SE conv weights as normal params. flip-TTA unaffected (SE gate is computed from GAP, flip-invariant).
- bf16/channels_last: 1×1 convs + mean reduction preserve memory format; sigmoid/relu in autocast as usual.
- Smoke: (i) σ-gate ∈ (0,1) per channel; (ii) at α=0 the GatedResidual output == input (identity) bit-exact; (iii) one-step backward finite grads on all SE params + α; (iv) num_params rises by exactly the SE param count.

## Verification
- Best SE cell ≥ **96.48** AND > same-session c0 by >0.1pp, replicated with a confirmation re-run on any apparent win (low-c0-draw lesson, EXP-016/017).
- num_epochs ≥ ~144 (throughput check); ep25 within ~0.5pp of c0; fully annealed.
- Integrity: train.py-only; prepare.py byte-unchanged; ≤1 eval/epoch; seed 42; SE-disabled cell ≡ baseline.
- ON A WIN: bake SE (winning r/placement) as default.

## Hypothesis
Content-adaptive channel recalibration (SE) adds a modeling capability absent from the saturated width/depth/regularization axes and lifts best_test_acc ≥96.48 over the same-session control at near-full epochs. If it ties at healthy epochs/ep25, channel-attention is redundant with the existing representation on this small net at 300s (plausible — SE's ImageNet gains may not transfer to a 7.8M-param CIFAR net with strong aug), pointing back to the downsampling/stem inductive bias (idea-01) or a genuine ceiling.

## Effort: low-medium. Risk: (1) SE's gains are ImageNet-scale; on a small CIFAR net with heavy aug + EMA they may be within the ~0.1pp noise floor (honest prior — channel attention helps most on large, deep, channel-rich nets); (2) the per-block GAP could add a sync stall costing epochs (mitigated: layer3-restriction, num_epochs gate); (3) SE inside the ReZero branch interacts with the α ramp — must verify identity-init holds.
## Sources: Hu et al. 2018 "Squeeze-and-Excitation Networks" (arXiv:1709.01507, CVPR); project-insights High (generalization ceiling EXP-014, backbone-pivot mandate); knowledge/references/rezero-identity-init.md (ReZero identity-init must be preserved); train.py:109-137 (Residual/GatedResidual branches).
