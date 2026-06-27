# Muon Optimizer (MomentUm Orthogonalized by Newton-Schulz)

**Sources**: kellerjordan.github.io/posts/muon/ ; github.com/KellerJordan/cifar10-airbench ; deepwiki.com/KellerJordan/cifar10-airbench/2.2-muon-optimizer (hyperparameters fetched 2026-06-10, EXP-028)

## Mechanism
Orthogonalizes the (nesterov-)momentum update of each 2D weight matrix — replaces the update U with the nearest semi-orthogonal matrix (all singular values → 1) via 5 quintic Newton-Schulz iterations. Equalizes update energy across directions instead of letting a few dominant singular directions carry the step. Benefit class: per-step sample efficiency (~1.35× on NanoGPT speedrun); set the CIFAR-10 94% record (3.3 → 2.59 A100-s) on airbench. Hidden 2D layers only — first/last layers and 1D params stay on SGD/AdamW.

## Newton-Schulz-5 (bf16-stable, Jordan's coefficients)
```python
def zeropower_via_newtonschulz5(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)  # quintic, max slope at 0; converges loosely on purpose
    X = G.bfloat16()
    if G.size(-2) > G.size(-1): X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1): X = X.mT
    return X
```
Step (per matrix param): `buf.mul_(m).add_(g); u = g.add(buf, alpha=m) if nesterov else buf; O = NS5(u.reshape(p.size(0), -1)).reshape_as(p); p.add_(O, alpha=-lr * sqrt(max(1, rows/cols)))`. Conv weights (C_out, C_in, k, k) reshape to (C_out, C_in·k·k).

## airbench CIFAR-10 hyperparameters (the only conv-net anchor)
- Muon on **conv filters only**: lr 0.24, momentum 0.6, nesterov, NS-5
- Head weights SGD lr 0.67; biases/BN SGD lr 0.053, momentum 0.85 nesterov
- airbench ALSO renormalizes conv weight norms per step (`p.mul_(len(p)**0.5 / p.norm())`) — a dual-norm scheme; importing Muon without it changes the effective dynamics (flagged in EXP-028)

## In-project status
- EXP-028 tested Muon-for-convs on the EXP-006 recipe (one-cycle shape, peak 0.24, decoupled WD 5e-4): **no-improvement, 96.53 vs bar 96.81** (two clean draws 96.42/96.53 = baseline mean 96.57). See exp-report-028.
- **Measured NS-5 cost on H20 (eager, 19 conv matrices, bf16)**: +2.9ms on a 22.4ms step (~13%) → 123 vs 139 epochs. Launch-bound as predicted (est. was +2–4ms).
- Mechanism verdict: sample efficiency REAL early (ep10 85.7 vs ~78, +7pp) but decays to zero by plateau; converged basin slightly worse under this recipe (final test_loss 0.193 vs ~0.185 both draws). Arrival-time gains pay in time-to-threshold regimes (airbench's ~10 epochs), not in max-over-checkpoints plateau regimes (~123 epochs).
- Caveat confirmed relevant: airbench's per-step weight-norm renormalization was NOT imported; basin deficit may partly trace to the missing dual-norm scheme — untested, sub-bar arithmetic either way.
