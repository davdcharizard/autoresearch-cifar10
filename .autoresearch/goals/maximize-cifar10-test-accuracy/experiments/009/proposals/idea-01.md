# Proposal EXP-009 idea-01: Decoupled weight decay (no-decay on BN γ/β + ReZero α)

**Created**: 2026-06-28
**Baseline**: 96.38% (EXP-008, commit 07c3760 — cutout12 + light RandomErasing on the EXP-004 recipe)
**Bar**: ≥96.48% (+0.1pp) AND clearly above the ~0.1pp run-to-run noise floor.

## One-line summary

Split the single SGD param list into two groups — `weight_decay=5e-4` for the multi-dim weight matrices (conv 4-D, fc 2-D) and `weight_decay=0.0` for the 1-D params (BatchNorm affine γ/β and the ReZero `alpha` scalar) — holding everything else (PEAK_LR=0.4, wd=5e-4, schedule, augmentation, EMA, seeds) byte-identical to the 96.38 baseline. This is the standard "no bias/BN decay" trick (Bag of Tricks, He et al. CVPR 2019, arXiv:1812.01187). One ~10-line edit at the optimizer construction.

## Mechanism (tied to the diagnosis)

The named limiter (per `03-experiment-learnings.md` Patterns and the EXP-008 analysis) is a **saturated / regularization-bound net with a ~4× epoch surplus**. EXP-008 proved the productive lever class is **throughput-free regularization-regime change** — it converts wasted annealing epochs into a higher annealed generalization ceiling without touching the epoch count. This idea is a second, orthogonal instance of that lever class: instead of adding input-space regularization, it **redistributes** the existing L2 penalty to where it controls real model complexity.

Causal chain from the change to the metric:

1. The current optimizer (`train.py:244-250`) applies `weight_decay=5e-4` uniformly to **every** trainable param. Because all conv/fc layers are `bias=False` (see `conv_bn` at `train.py:101-106` and `nn.Linear(512, 10, bias=False)` at `train.py:153`), the ONLY 1-D params in the net are: BatchNorm affine `weight` (γ) and `bias` (β) for each `nn.BatchNorm2d`, plus the single `GatedResidual.alpha` of shape `[1]` (`train.py:134`).
2. **Decaying BN γ/β is a spurious penalty.** BatchNorm already fixes the activation scale by normalizing to unit variance before the affine; pulling γ→0 and β→0 with L2 does not reduce a meaningful notion of model complexity, it just adds a scale-distorting force that the BN forward partly cancels. The textbook result (Bag of Tricks §4, "No bias decay") is that excluding γ/β/bias from decay gives a small but reliable generalization gain (~0.1–0.3pp on ImageNet conv nets) at zero cost. fastai and timm both default to this exclusion.
3. **Decaying `alpha` actively fights the capacity ramp that delivered EXP-004's +0.13pp.** EXP-004 showed α starts at 0 and ramps off zero to "earn capacity gradually" (`α.grad = 0.0179` at init; capacity lead emerged by ep25). The L2 force on α is `lr · wd · α`, a restoring force toward 0 — i.e. it directly opposes the mechanism that made the GatedResidual block useful. Removing it lets the block reach its natural operating α.
4. With BN/α excluded, the **effective weight decay on the conv/fc weights is unchanged** (still 5e-4) — the penalty is now applied only where it controls real complexity. The fully-annealed optimum shifts slightly, and on a net whose accuracy lands almost entirely in the low-LR tail (`03-experiment-learnings.md` Medium: "most accuracy gain arrives in the low-LR tail"), a small shift in the regularized optimum can register at the tail.

**The decisive property for THIS experiment**: this change is throughput-free AND does not slow convergence. EXP-008's tail was still mildly rising at ep150 (96.32→96.38), so the harder-augmented net is slightly under-annealed; the diagnosis explicitly says to AVOID levers that further slow convergence (stronger aug, capacity). Penalty redistribution changes neither the per-step cost (param-group SGD is the same arithmetic) nor the convergence speed in a direction that worsens the under-anneal — if anything, removing a restoring force on α/γ marginally *speeds* the relevant directions. So it is the rare lever with essentially zero downside on the current operating point.

## Concrete code change (files/functions actually read)

All edits are inside `main()` in `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/train.py`.

**Replace the optimizer construction (`train.py:244-250`)**, currently:

```python
optimizer = optim.SGD(
    [p for p in model.parameters() if p.requires_grad],  # exclude frozen whitening conv
    lr=PEAK_LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=True,
)
```

with a two-group partition by dimensionality. The frozen whitening conv (`self.whiten.weight`, `requires_grad_(False)` at `train.py:147,169`) is already excluded by the `requires_grad` filter; keep that filter and partition the survivors by `p.ndim`:

```python
decay_params, no_decay_params = [], []
for p in model.parameters():
    if not p.requires_grad:
        continue                      # frozen whitening conv stays excluded
    if p.ndim <= 1:                   # BN gamma/beta (1-D) + GatedResidual.alpha (shape [1])
        no_decay_params.append(p)
    else:                             # conv weight (4-D), fc weight (2-D)
        decay_params.append(p)

# Smoke assertion: the no-decay group must be SMALL (BN affine + the single alpha only).
# Per BN layer there are 2 1-D tensors (gamma, beta); the net has one alpha scalar.
n_no_decay = len(no_decay_params)
n_decay = len(decay_params)
assert n_no_decay >= 1, "no-decay group empty — partition logic broken"
# no-decay params must be far fewer in COUNT of elements than decay params
no_decay_numel = sum(p.numel() for p in no_decay_params)
decay_numel = sum(p.numel() for p in decay_params)
assert no_decay_numel < 0.05 * decay_numel, (
    f"no-decay numel {no_decay_numel} too large vs decay {decay_numel}"
)
print(f"wd groups | decay tensors: {n_decay} ({decay_numel:,} el) | "
      f"no_decay tensors: {n_no_decay} ({no_decay_numel:,} el)")

optimizer = optim.SGD(
    [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=PEAK_LR,
    momentum=MOMENTUM,
    nesterov=True,
)
```

**No other change is needed.** The LR-schedule loop already writes the LR to every group:

```python
for g in optimizer.param_groups:
    g["lr"] = lr
```

(`train.py:291-292`) — this iterates over both new groups, so both receive the one-cycle LR each step. `momentum`, `nesterov`, and the per-step LR are set identically for both groups; only `weight_decay` differs. The EMA wrapper (`AveragedModel`, `train.py:255-257`) averages `model` params directly and is independent of the optimizer grouping — untouched.

**Expected smoke numbers** (for falsification at milestone-1, computed from the architecture):
- BN layers: `prep` (1) + layer1 [conv_bn + 2 in Residual = 3] + layer2 [conv_bn + 2 in GatedResidual = 3] + layer3 [conv_bn + 2 in Residual = 3] = **10 BatchNorm2d**, each contributing γ and β → **20 one-dim BN tensors**, plus **1 alpha** → **21 no-decay tensors**.
- no_decay numel = sum of BN channel counts ×2 + 1. BN channels: prep 64; layer1 128,128,128; layer2 256,256,256; layer3 512,512,512 → channel sum = 64 + 384 + 768 + 1536 = 2752; ×2 (γ+β) = 5504; +1 alpha = **5505 elements**.
- decay numel = `learnable_params − 5505`. `learnable_params` is printed at `train.py:241-242`; total `num_params` is 7,784,627 (EXP-008), whitening conv is `54·3·3·3 = 1458` frozen, so learnable ≈ 7,783,169 and decay numel ≈ 7,777,664. Ratio no_decay/decay ≈ 0.0007 ≪ 0.05 → assertion passes with wide margin.
- The `num_params: 7,784,627` summary line MUST stay unchanged (no architecture change).

## Whether to hold PEAK_LR / wd

**Recommend hold PEAK_LR=0.4 and wd=5e-4 unchanged**, for two reasons:
1. **Single-variable attribution.** The whole value of this experiment is a clean read of the no-decay-on-BN/α effect against the 96.38 baseline. Co-tuning LR or the base wd confounds it and re-opens the under-anneal risk that LR changes carry.
2. **The effective-wd reduction is intended and small in the directions that matter.** Removing decay from BN/α reduces the *total* L2 force, but only on 5505 of ~7.78M elements (0.07%). The conv/fc weights — which carry essentially all the model's complexity and where wd actually regularizes — keep the full 5e-4. So this is NOT a global wd reduction that would need an LR rebalance; it is a targeted removal of a penalty on params where L2 is the wrong tool. No compensating wd bump is warranted.

## HONEST magnitude assessment vs the ~0.1pp noise floor

This is the candid part the assignment demands. **The most likely outcome is sub-noise, landing in the null band [96.30, 96.45], i.e. NOT clearing the +0.1pp bar.** The reasons:

- **The base recipe is already heavily regularized.** The 96.38 net stacks LS 0.2 + cutout12 + RandomErasing + EMA + wd 5e-4. The Bag-of-Tricks ~0.1–0.3pp figure for no-bias-decay is measured on ImageNet conv nets that were NOT simultaneously carrying this much other regularization, and it is the *upper* part of a range whose lower part (≈0.1pp) sits exactly AT this experiment's noise floor. When a net is already near its regularized ceiling, the marginal correction from one more penalty-redistribution shrinks.
- **The α-decoupling benefit is weakly quantified** (the EXP-008 cross-model review, `01-idea-review.md` item 5, rated it overstated). SGD decay on α is `lr·wd·α` — proportional to α, NOT a constant force. At PEAK_LR=0.4, wd=5e-4, the per-step pull is `0.4·5e-4·α = 2e-4·α`. Unless α grows to O(1), this is a tiny fraction of the measured `α.grad ≈ 0.0179`. So the α term only matters if α actually grows large during training — which we will measure (see Verification). Realistically the α contribution is a few hundredths of a pp at most.
- **There is only ONE GatedResidual block** (`layer2`); a deeper net with many ReScale gates would amplify the α benefit, but here it is a single scalar.
- **The BN-decay removal is the real evidence-backed mechanism**, but on a 10-BN-layer net it is also the standard, modest trick — its isolated effect here is most plausibly in the +0.02 to +0.08pp range, under the floor.

**The honest case FOR running it anyway is downside asymmetry, not ceiling.** Worst realistic case ≈ baseline (penalty redistribution toward the theoretically-correct allocation should not regress; it cannot under-anneal because epochs are budget-protected and convergence is not slowed). It is the lowest-regret throughput-free rider available. But on its own, as a single seed-fixed run against a bar that sits at the noise floor, it is **<35% to clearly clear 96.48**. A prior cross-model review rated its evidence 7/10 and ceiling 4/10; reading the actual code (single α, 0.07% of params un-decayed, already-saturated recipe) corroborates the low-ceiling call rather than overturning it.

**Implication for the loop**: this idea's best role is arguably as a *free rider folded into a future training-side win that itself clears the bar* (mirroring the EXP-006 TTA learning: "fold in as a free rider on a future training-side win"), OR as a deliberately-accepted low-ceiling/zero-downside probe if the loop wants to bank the standard trick. If the loop is choosing among candidates for the single 96.38-beating shot, a higher-ceiling regularization lever (e.g. pushing the proven EXP-008 augmentation axis further) is the better bet, and this should be honestly flagged as the conservative option.

## Evidence

- **Bag of Tricks (He et al., CVPR 2019, arXiv:1812.01187), §4 "No bias decay"**: exclude BN γ/β and biases from weight decay; one of the more reliable "free" tricks, ~0.1–0.3pp isolated on conv nets. Cited in this goal's `knowledge/references/fast-cifar10-recipes.md:27` and the EXP-008 brainstorm (`008/01-brainstorm.md:10`). fastai and timm ship this as a default.
- **EXP-004 analysis (`experiments/004/04-analysis.md`)**: α is live and accuracy-bearing — `α.grad = 0.0179` at init, capacity lead emerged by ep25 (92.63 vs 88.84). This establishes that α matters AND that the L2 restoring force on it is opposing a useful mechanism. It also bounds the α-decoupling magnitude: the per-step decay `2e-4·α` is small vs `α.grad ≈ 0.0179` unless α grows.
- **EXP-008 analysis (`experiments/008/04-analysis.md`)**: confirmed throughput-free regularization is the productive lever class (+0.38pp from CPU-side augmentation, epochs/total_seconds unchanged). This idea is in the same class but is honestly the lower-ceiling member of it. EXP-008 §Unexplored Avenues / Next Steps #2 explicitly names "compose decoupled weight decay (idea-01) with the EXP-008 augmentation" as a follow-up.
- **Code facts (read directly in `train.py`)**: all conv/fc are `bias=False`, so the only 1-D params are 20 BN tensors + 1 alpha; the LR loop already writes both groups; the frozen whitening conv is `requires_grad=False` and excluded by the existing filter.
- **EXP-006 learning (`03-experiment-learnings.md` Low / Failed Approaches)**: a sub-noise eval-side win peaked −0.07pp below baseline purely from epoch-count jitter — the cautionary precedent for why a sub-0.1pp expected change is unlikely to register as a clean win on one fixed-seed run.

## Verification / falsification

Pre-register these checks (mirroring EXP-008's protocol):
- **Throughput unchanged**: `num_epochs` stays in the ~142–150 band (expect ~150, matching EXP-008); `total_seconds` ~440–448s; `img/s` ~25k. Param-group SGD is the same arithmetic, so any material epoch drop would be host-contention, not the change — and would confound the read (treat as inconclusive, not a verdict).
- **num_params unchanged**: `7,784,627` (no architecture change). `learnable` line unchanged.
- **Smoke assertion fires correctly**: the printed `wd groups` line shows ~21 no-decay tensors / ~5,505 elements and ~7.78M decay elements (ratio ≪ 0.05). If the no-decay group captured a conv/fc weight (logic bug), the numel ratio assertion trips.
- **Record final α if cheap**: add a one-line `print(f"final_alpha: {model.layer2[2].alpha.item():.4f}")` after the loop (`layer2[2]` is the `GatedResidual`; index = [conv_bn, MaxPool2d, GatedResidual] per `train.py:150`). This quantifies the α-decoupling: if α stayed small (≪1), the α benefit was negligible and any gain is the BN-decay part; if α grew, the α-decoupling contributed. Cheap, off the hot path, no eval-budget impact.
- **Improvement verdict**: best_test_acc ≥ 96.48 AND clearly above the ~0.1pp floor. A 96.40–96.47 single-run result is a NULL (sub-noise), per the goal definition and `03-experiment-learnings.md` noise-floor finding — do NOT bank it as a win.
- **Falsified if**: best < 96.48 with normal epochs (the most likely outcome — penalty redistribution is sub-noise on this already-regularized recipe), OR best regresses below ~96.30 (would indicate the removed penalty was actually load-bearing — theoretically unexpected; would be an informative surprise).

## Strongest risk

The dominant risk is **sub-noise**: on a net already at 96.38 with LS 0.2 + cutout12 + RandomErasing + EMA + wd 5e-4, removing decay from 0.07% of the params (BN γ/β + one α) most plausibly moves the annealed accuracy by a few hundredths pp, landing in the null band against a bar that sits AT the noise floor. The assumption that most needs to hold for a *win* — that the BN/α decay was a materially harmful penalty on THIS recipe — is exactly the one the already-heavy regularization makes shaky. It is theoretically very unlikely to *hurt* (so worst case ≈ baseline, not a regression), but its appeal is zero-downside, not a high ceiling.

Secondary risk: epoch-count jitter from shared-host contention could produce a ±0.1pp swing that masks or fakes a small true effect (the EXP-006 precedent) — read `num_epochs` before trusting any small delta.

## Effort

**Low.** One ~10-line edit at the optimizer construction plus a 1-line final-α print; one milestone-1 in-process smoke (assert the partition counts) and one 300s training run. Reuses the EXP-008 verification protocol verbatim. No architecture, schedule, augmentation, or seed change.
