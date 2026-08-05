# SAM — Sharpness-Aware Minimization (Foret et al., ICLR 2021, arXiv:2010.01412)

**What**: A generalization-improving wrapper around any base optimizer. Each step does TWO
forward-backward passes:
1. Compute gradient `g` at current weights `w`.
2. **Ascent** to the local worst-case neighbor: `e_w = ρ · g / ‖g‖` (global 2-norm), set `w ← w + e_w`.
3. Compute gradient at the perturbed point `w + e_w`.
4. **Restore** `w ← w` (subtract `e_w`), then `optimizer.step()` using the **perturbed-point** gradient.

This biases SGD toward FLAT minima (low worst-case neighborhood loss), which correlate with a
smaller generalization gap. It is a **loss-GEOMETRY** mechanism — orthogonal to the loss function
(label smoothing), data (augmentation), weight penalty (weight decay), and preconditioner (Muon).

**Hyperparameters**: ρ (neighborhood radius). Canonical CIFAR-10 default **ρ=0.05**; 0.05–0.1 range;
overestimating ρ is less harmful than underestimating. Reported gains +0.3–1.0pp on CIFAR ResNets
*at matched epochs* (mostly heavier backbones — WRN-28-10, PyramidNet, Shake-Shake — than ResNet-9).

**Cost & the under-anneal trap (critical for this 300s time-budgeted goal)**: 2× fwd-bwd ≈ HALVES
throughput → plain SAM ~75–80 ep (under-anneal, expected loss given EXP-005/007). Control the cost by
applying SAM only in the **low-LR tail** (where this recipe's accuracy concentrates, EXP-001), plain
SGD before. Epochs ≈ `150·(1 − f/2)` where f = SAM-active fraction. **Gate correctly**: since `progress`
is elapsed-budget fraction, "final 35%" means `progress >= 0.65` (NOT `>= 0.35`, which would run SAM
for 65% of training → ~101 ep). Periodic SAM-k / LookSAM degrade accuracy — prefer the temporal split.

**Implementation correctness traps (hand-implement, no new deps)**:
- BN running stats must update on the **1st pass only** → set BN momentum=0 on the perturbed pass
  (wrap in try/finally so a NaN/throw can't leave momentum=0); matters because the EMA averages BN
  buffers (`use_buffers=True`).
- Compute `e_w` and the grad-norm on **fp32 master params**, autocast only the forward — SAM is
  numerically touchy under mixed precision; bf16's wide exponent range helps but watch early-tail loss.
- Restore `w` BEFORE `optimizer.step()` so momentum/wd/Nesterov apply to the perturbed grad at `w`.
- Clear the per-step perturbation state (avoid stale-`e_w` subtraction).

**Efficient variants (dependency-free fallbacks)**: ESAM-style sparse weight perturbation (perturb a
random ~50% subset each step); shrink the SAM-active tail fraction.

**Status on this goal**: tried EXP-013 (tail-only, ρ=0.05, start 0.65 & 0.75) → **no-improvement**. Ran
STABLE (no NaN; fp32 perturbation works under bf16) and the gate fired correctly, but both cells lost to
the same-session baseline (96.29/96.18 vs 96.47): the 2× fwd-bwd cost removed ~26 anneal epochs
(150→124/132) and the flat-minima gain didn't offset that at 300s. The lighter cell fully annealed and
still lost → it's the epoch cost, not under-anneal. **Do NOT re-test plain/tail SAM at 300s.** Only a
near-FREE variant (ESAM sparse-perturb / LookSAM reuse) could re-enter, but EXP-013 showed zero positive
SAM signal even where it ran. The published gains are matched-epoch / heavier backbones — irrelevant
under a time budget that charges 2× compute as halved epochs. See `experiments/013/04-analysis.md`.

Sources: arXiv:2010.01412; github.com/davda54/sam (PyTorch ref, BN-stats tip); efficient-SAM survey
arXiv:2406.08001.
