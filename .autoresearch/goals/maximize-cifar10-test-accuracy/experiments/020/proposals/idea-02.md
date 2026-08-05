# Proposal idea-02: GELU activation (replace ReLU) — adopt the fast-CIFAR-SOTA activation

## Core change (train.py only)
`conv_bn` (train.py:101-106) ends in `nn.ReLU(inplace=True)`. Replace the activation with **GELU** behind an `ACT` env (`relu` default / `gelu`). One-site change — every conv block (prep, layer1/2/3, residual branches) uses it.

```python
def _act():
    return nn.GELU() if ACT == "gelu" else nn.ReLU(inplace=True)

def conv_bn(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                         nn.BatchNorm2d(c_out), _act())
```
`ACT` env (`relu`/`gelu`). No other change.

## Mechanism — why this is a DIFFERENT (representational) lever
GELU (`x·Φ(x)`) is a smooth activation with a small negative lobe and a non-zero derivative everywhere, vs ReLU's hard zero-clamp and dead-gradient at x≤0. This changes the network's function class and gradient flow — a representation change distinct from the saturated capacity/optimizer/regularization/aug/downsampling/attention axes. Smoother activations are reported to ease optimization and slightly improve generalization on vision nets; critically, the two fastest published CIFAR recipes BOTH use GELU.

## Why it targets the limiter
The limiter is a budget-limited generalization ceiling (project-insights High, EXP-014). This attacks it at the **activation/representation** — a lever the leading fast-CIFAR nets adopted over ReLU but which THIS net never tried (we inherited DavidNet's ReLU). It is ~throughput-free (GELU is a cheap elementwise op; the tanh-approx/erf kernels are well-optimized under bf16/channels_last), so it sidesteps the #1 failure mode (under-anneal). If a smoother activation generalizes better here, it moves the metric without any capacity/epoch cost.

## External evidence
- **tysam hlb-CIFAR10** uses **GELU** (Conv→MaxPool→BN→GELU) and is one of the fastest ~94% CIFAR recipes (knowledge/references/fast-cifar10-recipes.md).
- **Keller Jordan airbench** (arXiv:2404.00498) uses **GELU** ConvGroup blocks and holds the speed records (95% ~10s, 96% ~27s). Our net descends from this lineage but kept ReLU.
- GELU (Hendrycks & Gimpel 2016, arXiv:1606.08415) is the default activation in modern vision/transformer backbones; smoothness aids optimization.

## Throughput
Near-neutral: GELU is a single elementwise op replacing ReLU; the cost difference is a few % at most on the activation kernels, dwarfed by the convs. MUST verify num_epochs ≥ 135 with a probe — if the GELU kernel is unexpectedly slow under our bf16/channels_last path, fall back (but airbench runs GELU at full speed, so the risk is low). Same num_epochs-first gate as every change.

## Correctness / EMA / eval
- Pure activation swap: no params (GELU is parameter-free), no VRAM change, deterministic, train≡eval. `AveragedModel(use_buffers=True)` and flip-TTA unaffected. bf16/channels_last preserved (GELU supports both).
- Interaction note: the `GatedResidual` ReZero comment reasons about ReLU'(0)=0 killing a zeroed-BN-γ gradient — IRRELEVANT here because (a) we use ReZero α=0 not zeroed-γ, and (b) GELU'(0)=0.5≠0 actually IMPROVES gradient flow. The frozen whitening conv has no activation. No init/LR retune needed (BN renormalizes activation scale), though a mild LR sensitivity is possible (judge via ep25/anneal).
- Smokes: (i) `ACT=relu` bit-identical to baseline (regression); (ii) num_params unchanged; (iii) finite fwd/bwd; (iv) GELU path runs and anneals (ep25 sane).

## Design — SAME-SESSION multi-cell
- c0: `ACT=relu` (baseline) — full-speed anchor + regression check.
- cA (PRIMARY): `ACT=gelu` — determines the verdict.
- (Optional cB if throughput allows: GELU only in the deeper layers vs everywhere — diagnostic; likely skip to keep the design clean, single-variable.)

## Verification
- cA(gelu) ≥ 96.48 AND > same-session c0 by >0.1pp; mandatory confirmation re-run on any apparent win.
- num_epochs ≥ 135 (probe first); ep25 within ~0.5pp of c0 (activation swap shouldn't depress early convergence); fully annealed.
- Integrity: train.py-only; prepare.py unchanged; ≤1 eval/epoch; seed 42; summary best == per-epoch max; `ACT=relu` ≡ baseline smoke. Background nvidia-smi sampling.
- ON A WIN: bake GELU as default.

## Hypothesis
Replacing ReLU with GELU (the activation used by the fastest published CIFAR recipes) yields a smoother loss landscape that generalizes ~0.1–0.3pp better, lifting best_test_acc ≥96.48 over the same-session control, throughput-free at ≥135 epochs. If it ties, the activation choice is not accuracy-limiting on this whitened ResNet-9 at 300s and the representation axis is further closed.

## Effort: low. Risk: (1) BN after the conv largely normalizes away activation-scale differences, so ReLU vs GELU often ties at convergence on small CIFAR nets (honest prior: modest EV; the airbench/hlb GELU choice may be for speed/stability, not accuracy); (2) GELU throughput could under-anneal if the kernel is slow under our exact bf16/channels_last config — mitigated by the num_epochs probe; (3) possible mild LR-sensitivity shift — judged via ep25/anneal, no retune planned (kept single-variable).

## Sources
knowledge/references/fast-cifar10-recipes.md (hlb/airbench GELU); arXiv:2404.00498 (airbench); arXiv:1606.08415 (GELU); train.py:101-106.
