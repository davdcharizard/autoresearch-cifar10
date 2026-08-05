# Muon optimizer (Keller Jordan; airbench94_muon, arXiv:2404.00498 lineage)

Source: https://github.com/KellerJordan/cifar10-airbench (`airbench94_muon.py`), fetched 2026-06-28.
Muon = SGD-momentum whose update **matrix** is orthogonalized by a fixed Newton-Schulz quintic before being applied. The lever behind the newest fast-CIFAR records on the **same** whitened wide-shallow ResNet family we use.

## Exact reference implementation (airbench94_muon)
- **Newton-Schulz**: coeffs `a,b,c = (3.4445, -4.7750, 2.0315)`; `X = G.bfloat16(); X /= (X.norm()+1e-7)`; transpose if `G.size(0)>G.size(1)`; iterate `A=X@X.T; B=b*A+c*A@A; X=a*X+B@X`; **steps=3**. Pushes singular values → ~1.
- **Momentum**: nesterov, `buf.mul_(mom).add_(g); g=g.add(buf,alpha=mom)`. airbench momentum = **0.6**.
- **Scaling = weight re-normalization**: `p.data.mul_(len(p)**0.5 / p.norm())` each step (pins ‖W‖_F=√out), then `p.data.add_(update, alpha=-lr)`. **No weight decay in the Muon step** (relies on the renorm + short schedule). Safe because every conv is BN-followed → scale-invariant to ‖W‖.
- **LR / schedule**: Muon-group **lr=0.24**, linear decay from peak to 0 over **8 epochs, no warmup**. Non-Muon params (biases, head) → separate **SGD** group with its own LR.
- **Param filter**: only **4D** tensors (`len(shape)==4`, conv weights) → Muon; the 2D classifier head goes to SGD (renorm would distort the non-BN'd logit scale).

## Key mechanism notes
- NS **normalizes its input** (`X/‖X‖`) and drives singular values→1, so the **update magnitude is independent of the gradient/momentum scale** — momentum changes smoothing/direction, NOT step size. Per-step rotation ≈ `lr·‖update‖/‖p‖`; with weight-renorm ‖p‖=√out and ‖update‖≈√min(m,n), at lr 0.24 this is **~24%/step** for the 512-wide convs.

## Validated on THIS project (EXP-009 — no-improvement, 94.11%)
- **airbench's peak LR 0.24 does NOT transfer to our long one-cycle.** Our 150-epoch triangular schedule holds near-peak LR for many epochs (vs airbench's 8-epoch no-plateau sprint); the ~24%/step rotation destabilized BN and **collapsed the net to ~random** through the high-LR phase (ep25-100 ~10-20%), recovering only as LR→0 (ep138 94.11, still rising). NS numerics / weight-renorm / EMA / dual-optimizer wiring all verified correct (clean recovery).
- **Fix for next attempt**: lower `PEAK_LR_MUON` ~2-3× (≈0.08-0.12), or use the **update-scale convention** (`update *= sqrt(max(1,out/in))`, no weight-renorm) + **decoupled WD** so ‖p‖ can shrink and per-step rotation drops. Read ep10/ep25 trajectory to set retune direction (divergence → LR too high; slow monotone → too low).
- Throughput: ns_steps=3 cost is negligible here (~23.5k img/s ≈ plain-SGD recipe).
