# Proposal idea-01: Squeeze-Excitation channel attention — adaptive feature recalibration

## Core change (train.py only)
Insert a lightweight **Squeeze-Excitation (SE)** block (Hu et al. 2018, CVPR, arXiv:1709.01507) into the residual branches. SE: global-average-pool each channel → 2-layer bottleneck MLP (`C→C/r→C`, ReLU then sigmoid) → multiply each channel by its gate. Env `SE_RATIO` (r, default 16), `SE_LAYERS` (which residual blocks).

```python
class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        cr = max(8, c // r)
        self.fc1 = nn.Conv2d(c, cr, 1); self.fc2 = nn.Conv2d(cr, c, 1)
    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)                 # squeeze
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))  # excite gate in (0,1)
        return x * s
```
Placed inside the residual branch, after `c2`, before the add: `Residual.forward → x + self.se(self.c2(self.c1(x)))`. For `GatedResidual` (ReZero α=0), SE sits inside the α-gated branch (`x + alpha*se(c2(c1(x)))`) so the block is still exact identity at init. **Init fix (EXP-018 reviewer concern #4)**: for the un-gated `Residual(128)`/`Residual(512)`, zero-init `fc2` so the gate starts at sigmoid(0)=0.5 (a constant 0.5 scaling, which the following BN/training absorbs) OR — cleaner — use `2*sigmoid` so the gate starts at 1.0 (identity-preserving). Default: zero-init `fc2.weight`/`fc2.bias` → uniform 0.5 gate at init (stable, recipe-neutral) and verify ep25 not depressed.

## Mechanism — why this is a genuinely DIFFERENT lever
Every prior experiment adjusted capacity (width/depth EXP-005/007/014), the optimizer (009/010), regularization strength (008/011/012/013/015), BN/activation noise (016/017), or the spatial downsampling operator (018). NONE changed the *functional form* of how channels interact. SE adds **content-adaptive channel gating** — an attention mechanism that conditions each channel's scale on the global image content. It is a new modeling capability, not raw width, at ~`2·C²/r` params/block (tiny: ~16K at C=256, r=16).

## Why it targets the limiter
The limiter is the generalization ceiling (~96.3–96.5; project-insights High, EXP-014; reinforced by 13 straight nulls EXP-006→018). SE is one of the few proven ways to raise a convnet's accuracy WITHOUT adding width/depth — the original paper reports consistent ImageNet top-1 gains (~0.5–1pp) at <1% extra params/FLOPs. Its compute is a GAP reduction + two 1×1 convs on a [N,C,1,1] tensor → near-throughput-free, avoiding the under-anneal trap that disqualified capacity (005/007), SAM (013), GhostBN (016), and MaxBlurPool (018).

## Throughput (the #1 failure mode — pre-registered check)
SE compute is dominated by the per-block GAP (a `mean` reduction). Two 1×1 convs on 1×1 spatial are trivial. Standard ops, no fused-kernel break. Expect ≤~3% cost (≥~145 ep). MUST run the M1 full-train-step throughput probe; if a cell drops <135 ep, restrict to layer3 only. Watch the GAP for a CUDA-sync stall.

## Design — SAME-SESSION multi-cell
- c0: unchanged baseline — full-speed same-session anchor (stored 96.38 too weak at the noise floor).
- cA: SE in layer2+layer3 residual branches, r=16 — PRIMARY.
- cB: SE in all three blocks (layer1+2+3) OR r=8 — second operating point per the throughput smoke and cA's ep25.

## Correctness / EMA / ReZero / eval
- SE `fc1/fc2` are standard 1×1 convs → trainable, kaiming-init (then `fc2` zero-init override); optimizer picks them up.
- ReZero preserved: SE inside the α-gated branch → α=0 identity at init (one-step backward smoke: α.grad≠0, SE grads finite).
- Eval/EMA: SE deterministic, train≡eval; `AveragedModel(use_buffers=True)` averages SE conv weights as params. flip-TTA unaffected (gate from GAP is flip-invariant).
- bf16/channels_last: 1×1 conv + mean reduction preserve memory format.
- Smokes: (i) gate ∈(0,1); (ii) at α=0 GatedResidual output == input (bit-exact); (iii) finite backward on SE params + α; (iv) num_params rises by exactly the SE count; (v) eval native-fp32 + flip + EMA-param coverage.

## Verification
- Best SE cell ≥ **96.48** AND > same-session c0 by >0.1pp, replicated with a mandatory confirmation re-run on any apparent win (low-c0-draw lesson, EXP-016/017/018).
- num_epochs ≥ ~135 (hard under-anneal gate); ep25 within ~0.5pp of c0; fully annealed (best≈final).
- Integrity: train.py-only; prepare.py byte-unchanged; ≤1 eval/epoch; seed 42; summary best == per-epoch max (anti-gaming); SE-disabled cell ≡ baseline.
- ON A WIN: bake SE (winning r/placement) as default.

## Hypothesis
Content-adaptive channel recalibration (SE) adds a modeling capability absent from the saturated width/depth/regularization/downsampling axes and lifts best_test_acc ≥96.48 over the same-session control at near-full epochs. If it ties at healthy epochs/ep25, channel attention is redundant with the existing representation on this 7.8M-param CIFAR net at 300s — pointing to the readout (idea-03), schedule (idea-02), or a genuine ceiling.

## Effort: low-medium. Risk: (1) SE's gains are ImageNet-scale; on a small CIFAR net with heavy aug + EMA they may sit within the ~0.1pp noise floor (the honest prior); (2) per-block GAP sync stall costing epochs (mitigated: layer3-restriction, num_epochs gate); (3) SE in the un-gated Residual blocks can disturb the validated recipe at init (mitigated: zero-init fc2 → 0.5 gate, or 2·sigmoid identity gate; verify ep25).
## Sources: Hu et al. 2018 (arXiv:1709.01507); EXP-018 proposals/idea-02.md + 01-idea-review.md (scored 6/10, init concern); knowledge/references/rezero-identity-init.md; project-insights High (ceiling); train.py:109-137.
