# ReZero — trainable identity-init for residual blocks

**Source**: Bachlechner et al. 2020, "ReZero is All You Need: Fast Convergence at Large Depth" (arXiv:2003.04887).

## Core idea
Gate a residual branch with a **learnable scalar α initialized to 0**:
```
y = x + α · F(x)
```
At init (α=0) the block is **exact identity**, so a deeper net starts bit-equivalent to its shallower base. Unlike a frozen identity, α has a live gradient (∂L/∂α = ⟨grad_out, F(x)⟩, generally ≠ 0), so the block trains and ramps its capacity in gradually as α moves off zero. Stabilizes deep-net training and removes the need to retune LR/warmup when adding depth.

## Why it matters here (validated EXP-004, +0.13pp → 96.00%)
- Adding capacity to the DavidNet (a layer2 residual block) needed an identity-init so the deeper net wouldn't disrupt the annealed low-LR tail (where all this goal's gains live) and so PEAK_LR could stay fixed (clean single-variable capacity test).
- **Critical pitfall**: the obvious "zero the final BatchNorm γ" identity-init is **DEAD** in this codebase because `conv_bn` ends in ReLU — `ReLU(0)=0` with derivative 0, so the block never receives gradient and stays identity forever. ReZero avoids this (the gate sits outside the ReLU). Always verify trainability with a one-step backward smoke (`α.grad ≠ 0`) before the official run.
- Cost: one scalar `nn.Parameter(torch.zeros(1))` per block; it joins the SGD param group (WD on it is negligible).

## Reuse
Next capacity experiments (stack a second gated block, widen block2 toward airbench96, per-channel LayerScale variant) should reuse this gate. Composes with whitening + EMA + flip-TTA (all orthogonal).
