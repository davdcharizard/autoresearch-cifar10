# Smooth Non-Monotonic Activations: Swish/SiLU & Mish

**Sources**:
- Ramachandran, Zoph & Le — "Searching for Activation Functions" (2017), arXiv:1710.05941 (Swish/SiLU)
- Misra — "Mish: A Self Regularized Non-Monotonic Activation Function" (2019), arXiv:1908.08681 (Mish)

## Key Insights

- **SiLU / Swish** = `x · σ(x)` (σ = sigmoid). Smooth, non-monotonic (dips slightly negative for small negative x before returning to ~0). Found by RL search over activation space. Reported small CONSISTENT top-1 gains over ReLU across architectures (ResNets, MobileNet) on CIFAR and ImageNet when used as a drop-in replacement for EVERY ReLU. Mechanism: smoothness gives a non-zero gradient for small negative pre-activations (no hard "dead-ReLU" zero region), easing optimization; the self-gating `σ(x)` acts as a soft, input-dependent gate. Core PyTorch: `F.silu` / `nn.SiLU` (no new dependency).
- **Mish** = `x · tanh(softplus(x))` = `x · tanh(ln(1+e^x))`. Similar smooth non-monotonic shape; reported marginally stronger CIFAR gains than Swish in the original benchmarks. Core PyTorch: `F.mish` / `nn.Mish` (no new dependency). COST: ~2× SiLU's pointwise cost (a `tanh` AND a `softplus`/`exp`) — a larger throughput risk under a compute-time-gated budget.
- Both are **pointwise** ops. Under `torch.compile`, pointwise activations typically FUSE into the preceding conv/BN epilogue, so under a LAUNCH-BOUND regime (GPU waiting on kernel launches, not pointwise FLOPs) the added wall-clock can be ~zero. This is the key feasibility argument for trying them at a fixed compute budget where every compute-ADDING structural change has hit an epoch wall.
- Init note: `kaiming_normal_` computes its gain for ReLU/leaky-ReLU. Swapping the activation leaves a slight init-gain mismatch, but in a fully BN'd net (BatchNorm after every conv) the per-layer scale is re-normalized, so the mismatch is second-order and usually ignored in practice.

## Relevance to this project (CIFAR-10 k=4 WRN, 300s/H20)

- The recipe is convergence-bound and the activation function (always `F.relu`, train.py L89/L92/L127) is the single largest UNTRIED orthogonal lever — flagged in the EXP-009 goal-learning ("try orthogonal axes — activation function"). Unlike capacity changes (epoch wall), regularizers (convergence-bound net rejects them), or convergence-polish (EMA/SWA/LS-down move loss not top-1), a smooth activation changes the optimization/representation landscape, the class that CAN move top-1.
- **SiLU is the safe first probe** (cheapest, best odds of staying throughput-neutral at the launch-bound 8ms/step). Mish is the follow-up ONLY if SiLU shows a positive AND throughput-neutral signal — its 2× pointwise cost risks the epoch wall that confounded EXP-024.
- CAVEAT on magnitude: on an already well-tuned shallow ResNet-20-style net the gain may be below the ~0.2pp noise floor → no-improvement. Judge on top-1 (the goal), and ALWAYS check epoch count (throughput-neutrality) before attributing any delta to the activation rather than a dt change.
- **EXP-028 RESULT — SiLU NULL (and a small throughput loss):** best_test_acc 95.98 (−0.24pp), final_test_loss FLAT 0.196≈0.195, at 88 ep. Critically, dt rose 8→9ms — SiLU's `σ(x)` did NOT fully fuse into the conv/BN epilogue under torch.compile, costing ~1ms/step (~12%) and ~3 epochs. So the launch-bound "pointwise ops are free" assumption did NOT hold for SiLU here. The flat loss (no convergence benefit) shows this well-tuned shallow net is not activation-limited — the dead-ReLU smoothing benefit is a deep/hard-to-train phenomenon that doesn't bind. Mish (2× the pointwise cost) would be strictly worse on throughput for the same null mechanism — do NOT try it. The activation axis is CLOSED for this budget.
