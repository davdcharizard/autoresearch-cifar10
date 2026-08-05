# Idea-02: Throughput-free recipe alignment to airbench96 — GELU + cutout 12

## Summary
Two one-line edits to `train.py`, both in the parts I read directly:

1. **`conv_bn` (line 101-106)**: replace `nn.ReLU(inplace=True)` with `nn.GELU()`. This swaps the activation in every Conv-BN-act unit at once — `conv_bn` is the single helper used by `prep`, all three `layer{1,2,3}` ConvGroups, and both `c1`/`c2` inside `Residual` and `GatedResidual`. So one edit changes the activation network-wide. Ordering stays conv→BN→act, which is the documented airbench order.
2. **Augmentation site (line 211)**: change `Cutout(8)` to `Cutout(12)` inside `train_tf`. The `Cutout.__init__` default and `__call__` already parametrize on `self.size` (line 50-61), so no class change is needed — only the call-site argument.

These are the only lines touched. No schedule, optimizer, LR, or architecture-shape change → identical step count per host draw (zero throughput cost; GELU is a cheap pointwise op, larger cutout is free).

**Recommendation: test GELU + cutout12 TOGETHER as one "align to airbench96 cheap recipe" change**, not separately — see Risk.

## What it targets
The optimization-smoothness + regularization gap versus the documented 96.03% airbench96 recipe. Our net is the EXP-004 recipe (96.00%) but uses ReLU and Cutout(8); airbench96 (96.03%) and hlb-CIFAR10 both use GELU throughout and cutout=12 (`fast-cifar10-recipes.md` lines 8, 13). This is the only remaining throughput-free divergence from the reference net, so it sidesteps the capacity-vs-epochs tension that sank EXP-005 (a 4×4 block cost 11 epochs, `03-experiment-learnings.md` line 60-62).

## Reasoning
- `fast-cifar10-recipes.md` line 8: airbench "GELU ConvGroup blocks (Conv→MaxPool→BN→GELU)"; line 6: "hlb-CIFAR10... GELU activations". GELU is the activation in both nets that beat ours by 0.03pp.
- GELU has nonzero gradient everywhere, unlike ReLU. Relevance to our blocks: the ReZero `GatedResidual` (line 119-137) identity-init relies on `alpha=0` and is activation-independent, so unaffected. The "dead block" concern flagged in project-insights (zeroed-BN-γ fails because ReLU'(0)=0, line 51-53) does NOT apply here — we use ReZero, not zeroed-γ. But it is worth noting GELU would actually *relax* that constraint for any future zeroed-γ block. No interaction harms the current net.
- `_weights_init` (line 157-160) uses `kaiming_normal_(nonlinearity="relu")`. With GELU the ReLU-tuned gain is slightly off-optimal but harmless: BN immediately follows every conv and renormalizes activation scale, so the init gain only affects the very first forward, not the trained regime. I will leave it as-is to keep the change minimal (changing it would conflate a second variable).
- Cutout12 removes ~2.3× more area (12² vs 8²) → stronger regularization, the airbench value. With ~142 epochs in budget (EXP-004) convergence is fully annealed, so the slightly slower early convergence from extra occlusion is very likely absorbed in the tail (EXP-001 line 88-90: gain concentrates in the low-LR tail).

## Estimated effort
**Low** — two-line diff, one training run under `timeout 600`, GPU 1, assert `prepare.py` byte-unchanged.

## Risk assessment
The dominant risk is the **~0.1pp run-to-run noise floor** (`03-experiment-learnings.md` line 32-34): the time-budgeted loop fits a host-throughput-dependent epoch count (142/131/150 across identical-code runs), and seed re-rolling is disallowed. Each individual change (GELU alone, cutout12 alone) plausibly contributes <0.1pp and would be indistinguishable from epoch-count jitter — exactly how EXP-006's real +0.28pp TTA onset still netted −0.07pp. **This is why I recommend combining GELU+cutout12**: stacking two same-direction recipe-alignment changes raises the chance the *summed* effect clears the +0.1pp bar in a single run, at the cost of clean per-change attribution (acceptable — both are documented airbench96 values, so neither needs to be credited individually). The remaining risk is that cutout12's extra regularization slightly *under-fits* within budget and cancels GELU's gain. Worst case is a clean no-improvement with **zero baseline risk**: training stays byte-reversible, no architecture/throughput change, and the EXP-004 96.00% recipe is untouched on the integration branch. Given the modest upside, treat any sub-0.1pp single-run delta as unproven per the noise-floor finding.
