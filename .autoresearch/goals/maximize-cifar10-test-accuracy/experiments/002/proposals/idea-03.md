# Proposal idea-03: Scale the DavidNet/ResNet-9 wider (and slightly deeper) under the same time-based one-cycle

## Summary

EXP-001's ResNet-9 (6.57M params) fit **192 epochs** of a 24-epoch-designed net into
the 300s budget while using only **1.6 GB of 98 GB** VRAM (`experiments/001/04-analysis.md`
§ Execution/Results). 192 epochs is ~8× the canonical schedule for this net — the
extra epochs are deep in diminishing returns. The diagnosis names two limiters:
under-annealing (fixed by EXP-001's completing one-cycle) and **massively
underutilized compute / under-capacity model** (`experiments/001/01-brainstorm.md`
§ Diagnosis point 2). This idea attacks the second: **trade excess epochs for a
higher-capacity net** by widening (and optionally lightly deepening) the proven
DavidNet, keeping the entire EXP-001 recipe (time-based one-cycle, SGD+Nesterov,
Cutout, label smoothing, bf16+channels_last, `scale_out=0.125`) unchanged. Because
the LR schedule is keyed on `total_training_time / TIME_BUDGET_S` (train.py lines
188-192), it **auto-adapts** to the new (lower) epoch count with no schedule
retuning — this is the property that makes a pure capacity swap clean.

The concrete primary variant is a **1.5× width** DavidNet: prep 3→96, stages
96→192(+Residual)→384→768(+Residual), ~14.8M params. Estimated ~2.25× FLOPs/step →
~27 steps/s → **~80-85 epochs** in 300s — still many multiples of the ~24 epochs one
cycle needs to converge. Expected `best_test_acc` **~95.4-95.9%** (central ~95.6%),
clearing the +0.1pp bar (≥95.32%). This is the lowest-risk, lowest-effort of the
loop-2 ideas: a few-line architecture edit on a validated base, no new mechanisms.

## What it targets (limiter from the diagnosis)

Named limiter: **"Compute massively underutilized ... the net is UNDER-capacity for
the available compute, not over-trained"** (`01-brainstorm.md` § Diagnosis; idea
brief). EXP-001's own analysis flags this as a top next step:
`04-analysis.md` § Unexplored Avenues — *"Wider/deeper net: only 1.6 GB VRAM used —
widths could grow substantially"* — and § Next Steps item 3.

Causal chain to the metric:
1. The metric is accuracy under **fixed training time**, and EXP-001 spent that time
   on ~192 epochs of a small net — well past where a one-cycle net saturates (most of
   the gain already arrives in the low-LR tail of a *single* completing cycle, per
   `03-experiment-learnings.md` Patterns/Medium). Extra epochs of the same net yield
   near-zero marginal accuracy.
2. A wider net has strictly higher representational capacity per the WideResNet result
   (Zagoruyko & Komodakis, arXiv:1605.07146): *"decrease depth and increase width ...
   a 16-layer-deep wide residual network outperforms ... thousand-layer-deep
   networks"* — width converts compute into accuracy more efficiently than depth on
   CIFAR.
3. The wider net is ~2.25× slower/step → ~80-85 epochs fit instead of 192. Since one
   cycle converges by ~24 epochs, 80+ epochs is still ample headroom for the anneal to
   complete; we move from the *flat* part of the epochs→accuracy curve (192 ep, small
   net) to a *higher-asymptote* curve (80+ ep, big net). The peak of that higher curve
   sits above 95.22%.

In short: spare FLOPs that were buying redundant epochs are redirected into parameters
that raise the achievable accuracy ceiling.

## Exact `train.py` changes

All edits are in `train.py`; `prepare.py` (eval + 300s budget) untouched. Normalization
stays `EVAL_MEAN=(0.4914,0.4822,0.4465), EVAL_STD=(1,1,1)` (train.py line 30) — no
change, so train/eval stay synced. The recipe constants (PEAK_LR 0.4, MOMENTUM 0.9,
WEIGHT_DECAY 5e-4, LABEL_SMOOTHING 0.2, PCT_START 0.15, SCALE_OUT 0.125, Cutout 8) are
**kept verbatim**; only the architecture (and possibly batch size) changes.

### Architecture (primary variant: 1.5× width DavidNet)

Replace the `ResNet9.__init__` body (train.py lines 84-99) with a width-parameterized
version. The structure, residual placement, pooling, and head are identical to
EXP-001; only channel counts scale. `conv_bn` (lines 66-71) and `Residual` (lines
74-81) are reused unchanged.

```python
class ResNet9(nn.Module):
    def __init__(self, num_classes=10, scale_out=SCALE_OUT, w=(96, 192, 384, 768)):
        super().__init__()
        self.scale_out = scale_out
        c0, c1, c2, c3 = w
        self.prep   = conv_bn(3, c0)                                          # 32x32
        self.layer1 = nn.Sequential(conv_bn(c0, c1), nn.MaxPool2d(2), Residual(c1))  # ->16x16
        self.layer2 = nn.Sequential(conv_bn(c1, c2), nn.MaxPool2d(2))               # ->8x8
        self.layer3 = nn.Sequential(conv_bn(c2, c3), nn.MaxPool2d(2), Residual(c3)) # ->4x4
        self.pool   = nn.MaxPool2d(4)                                               # ->1x1
        self.fc     = nn.Linear(c3, num_classes, bias=False)
        self.apply(self._weights_init)
```

`forward` (lines 101-107) is unchanged. The `512`→`c3` change in the `nn.Linear` is the
only head edit. Update the print on line 153 to report the new param count (cosmetic).

**Spatial-dim check (input 32×32):** prep @32 → layer1 conv@32, MaxPool2→16, Residual@16
→ layer2 conv@16, MaxPool2→8 → layer3 conv@8, MaxPool2→4, Residual@4 → `MaxPool2d(4)`@4
→ **1×1** → flatten → Linear(768→10). Identical spatial schedule to EXP-001 (which
ended 32→1 correctly), so the global `MaxPool(4)` still collapses to 1×1 for any width
— width does not touch spatial dims. ✓

**Param count:** convs scale ~quadratically in width. EXP-001 = 6.57M; 1.5× width →
~6.57M × 1.5² ≈ **14.8M params** (the prep-input and head terms are tiny, so the
quadratic estimate dominates). At ~14.8M params, fp-state for SGD+momentum is trivial
on 98 GB — even 4× this is < 5 GB activations+params at batch 512.

### Batch size and LR

**Keep BATCH_SIZE = 512 and PEAK_LR = 0.4 unchanged.** Rationale: the one-cycle peak,
wd, and scale_out convention is a *coupled* set validated at 95.22% (`03-experiment-learnings.md`
Patterns/High); width does not change the loss reduction (mean) or the logit scale, so
the same peak LR remains in the stable regime — widening a residual net under a fixed
recipe does not require LR rescaling (WideResNet uses the *same* LR schedule across
width multipliers). Batch 512 already gives full H20 utilization at 1.6 GB; a wider net
raises arithmetic intensity, which *helps* throughput efficiency, so there is no reason
to shrink batch. (If epoch-1 loss diverges — watch the `debiased` print — the single
safe fallback is PEAK_LR 0.4→0.3; do not change other constants.)

### Estimated epochs in budget (the load-bearing arithmetic)

EXP-001 measured: 18,529 steps / 300s = **61.8 steps/s**; 192 epochs; 97 batches/epoch
(50,000 train imgs, batch 512, drop_last). Per-conv FLOPs ∝ C_in·C_out·H·W. I summed
the 8 conv terms of the EXP-001 net (units of C_in·C_out·H·W, ×9 common factor
dropped): prep 0.20M, layer1-conv 8.39M, layer1-res×2 8.39M, layer2-conv 8.39M,
layer3-conv 8.39M, layer3-res×2 8.39M → **≈42.1M units** total, dominated by the
32×32 and 16×16 stages.

Scaling all stage widths by 1.5×: every conv except the prep-input term scales by
1.5²=2.25 (both C_in and C_out scale). So FLOPs/step ≈ **2.25×**. Throughput is
compute-bound here (per-step ~16 ms, the H20 is busy), so steps/s ≈ 61.8 / 2.25 ≈
**27.5 steps/s** → ~8,250 steps / 97 ≈ **~85 epochs** in 300s. Even allowing for
imperfect kernel scaling (wider channels improve utilization, so this is conservative;
real slowdown is often < FLOP ratio), expect **~75-90 epochs** — comfortably ≥ the
~24-40 epochs a completing one-cycle needs.

### Optional secondary lever (only if the width run leaves margin): light depth

The current layer2 (256→512... here 192→384 stage) is the **one stage with no
residual block** (train.py line 90). Adding `Residual(c2)` after its MaxPool — i.e.
`nn.Sequential(conv_bn(c1, c2), nn.MaxPool2d(2), Residual(c2))` — mirrors layer1/layer3
and adds 2 convs at 8×8 (cost 2·c2² ·64 units). At c2=384 that is 2·384²·64 ≈ 18.9M
units (+45% FLOPs) — too expensive to combine with full 1.5× width. **Treat depth as
mutually exclusive with the full width bump**; the primary deliverable is the width
variant. A documented fallback ladder (below) lets the executor pick a smaller width if
85 epochs proves too few.

### Width ladder (executor picks one; do NOT sweep seeds, just one architecture)

| variant | widths (prep, s1, s2, s3) | ~params | ~FLOP× | ~steps/s | ~epochs/300s |
|---|---|---|---|---|---|
| EXP-001 base | 64,128,256,512 | 6.57M | 1.0 | 61.8 | 192 |
| **1.25× (safe)** | 80,160,320,640 | ~10.3M | ~1.56 | ~40 | ~120 |
| **1.5× (primary)** | 96,192,384,768 | ~14.8M | ~2.25 | ~27 | ~85 |
| 1.75× (aggressive) | 112,224,448,896 | ~20M | ~3.06 | ~20 | ~62 |

Recommended primary: **1.5×**. It is the capacity sweet spot — large enough to plausibly
add capacity headroom, small enough to keep ~85 epochs (3.5× the canonical 24). The
1.25× variant is the conservative fallback if 1.5× under-converges; 1.75× is the upside
probe only if 1.5× clearly helps and epochs stay ≥60.

## Reasoning with cited pointers

- **Headroom is real and measured.** `experiments/001/04-analysis.md` § Results: "The
  6.5M-param net used only 1.6 GB of 98 GB — VRAM is nowhere near a constraint, leaving
  large headroom for wider/deeper nets." § Unexplored Avenues + Next Steps item 3
  explicitly name the wider net. `03-experiment-learnings.md` Patterns/Medium: "VRAM is
  NOT a constraint here ... free to widen/deepen the net substantially."
- **Excess epochs are wasted, so trading them is cheap.** EXP-001 ran 192 epochs of a
  24-epoch net; "returns may diminish past ~192 epochs of one cycle"
  (`04-analysis.md` § Unexplored Avenues, More throughput). The single-completing-cycle
  gain saturates well before 192 epochs, so dropping to ~85 epochs of a *bigger* net
  loses little schedule benefit while adding capacity.
- **Width > depth for CIFAR residual nets.** WideResNet (Zagoruyko & Komodakis,
  arXiv:1605.07146, fetched): widening outperforms deepening in accuracy *and*
  efficiency; a shallow-wide ResNet beats much deeper ones. This is exactly the lever
  here — keep the proven 9-layer skeleton, grow width.
- **The fast-CIFAR record family reaches higher acc by widening+deepening.** airbench96
  (Keller Jordan, arXiv:2404.00498; DeepWiki architectures search) hits 96% by widening
  ConvGroups, adding an extra conv per block, and adding residuals — direct precedent
  that more capacity on this net class buys the last 1-2pp. (airbench also adds
  whitening; that is idea-02's territory, deliberately *not* combined here to keep this
  a clean capacity-only test.)
- **No LR retune needed.** The schedule is `progress = total_training_time /
  TIME_BUDGET_S` (train.py lines 188-192) — independent of step count — so fewer epochs
  still anneal to ~0 by 300s. WideResNet uses the same LR schedule across width
  multipliers, supporting keeping PEAK_LR=0.4.
- **Throughput model.** EXP-001 log facts (`04-analysis.md` § Execution): 61.8 steps/s,
  ~16 ms/step, 192 epochs, 1.6 GB. FLOP ratio computed from the net's conv geometry
  (above). Compute-bound regime → steps/s ≈ baseline / FLOP-ratio.

## Estimated effort

**Low.** A width-parameterized `ResNet9.__init__` (≈8 lines changed) plus the head
`512`→`c3` edit and a cosmetic print. `conv_bn`, `Residual`, `forward`, the training
loop, schedule, augmentation, optimizer, and eval are all reused verbatim. One run,
one architecture. The only iteration risk is picking a width that under-converges,
mitigated by the ladder (start 1.5×, fall back to 1.25× if needed).

## Risk assessment

**Strongest risk / assumption that most needs to hold: that ~85 epochs of the 1.5× net
out-converges 192 epochs of the base net within 300s.** The capacity gain must exceed
the epochs lost. Two ways this fails:

1. **Too big → too few epochs → one-cycle under-converges (worst case).** If the wider
   net is FLOP-heavier than estimated (kernel inefficiency at odd widths) or needs more
   than ~40 epochs to converge, it may finish the cycle still climbing and land *below*
   95.22%. Mitigations: (a) 1.5× targets ~85 epochs, ~3.5× the canonical 24, large
   margin; (b) `best_test_acc` is best-across-epochs, capturing the peak even if the
   final epoch isn't optimal; (c) the 1.25× fallback (~120 epochs) if 1.5×
   under-converges; (d) the time-based schedule guarantees the anneal *completes*
   regardless of epoch count, so the failure is "still rising at LR→0," detectable from
   the per-epoch eval trace (acc still increasing steeply at the last epoch).
2. **Capacity wasn't the binding limiter / diminishing returns near 95%.** ResNet-9 may
   already be near its generalization ceiling for this augmentation budget, so extra
   width buys < 0.1pp (or even slightly overfits, partly offset by Cutout + label
   smoothing 0.2). This is the honest weak spot: 95.22% is already in airbench-95
   territory, and pure width without whitening/TTA may yield only a few tenths.
   **Honest expectation: modest (~+0.2-0.6pp), not dramatic.** The bar is +0.1pp, so
   even a modest gain clears it, but this is not a high-ceiling idea — idea-02
   (whitening) has more headroom. If this returns < +0.1pp, the learning is "capacity
   is not the binding limiter at 95%; schedule/data/eval tricks are," which usefully
   redirects the next loop.

Lower risks: divergence at higher width (unlikely — same LR/scale/wd convention,
fallback PEAK_LR 0.3); VRAM (non-issue, ~3-5 GB of 98); throughput estimate off by
~30% (covered by the ladder + best-across-epochs scoring). No new deps, no eval/seed
surface touched (seed `torch.manual_seed(42)` and the single `evaluator.evaluate`/epoch
preserved).

## Expected accuracy estimate vs 95.22%

- **Primary (1.5× width, ~85 epochs, recipe unchanged): ~95.4-95.9%, central ~95.6%
  (≈ +0.4pp).** Justification: width adds capacity (WideResNet); ~85 epochs comfortably
  completes the one-cycle; the rest of the validated 95.22% recipe is untouched. The
  gain is bounded because we add only capacity (no whitening, no TTA, no new aug).
- **Conservative floor (correct run, capacity not very binding): ~95.2-95.4%** — a
  wash-to-marginal pass; relies on best-across-epochs to clear +0.1pp.
- **Upside (1.5× or 1.75× hits a real capacity sweet spot, ≥60 epochs): up to ~96.0%.**

Net: a **low-risk, low-effort** experiment with a **modest expected gain (~+0.4pp,
clears the +0.1pp bar)** and a small chance of a wash. Its main value beyond the metric
is a clean read on whether model capacity (vs schedule/data/eval tricks) is still a
binding limiter at ~95% — directly informing whether to invest in the higher-ceiling
whitening idea next.

## Sources

- `experiments/001/04-analysis.md` (EXP-001 results: 95.22%, 192 ep, 61.8 steps/s, 1.6 GB, next-steps)
- `experiments/001/01-brainstorm.md` § Diagnosis (under-capacity / underutilized-compute limiter)
- `goals/.../03-experiment-learnings.md` (validated base recipe; VRAM non-constraint; low-LR-tail gain)
- `train.py` lines 18-30 (recipe constants), 66-107 (conv_bn/Residual/ResNet9/forward), 188-192 (time-based schedule)
- [Wide Residual Networks, Zagoruyko & Komodakis, arXiv:1605.07146](https://arxiv.org/abs/1605.07146) (width > depth on CIFAR)
- ["94% on CIFAR-10 in 3.29 Seconds", Keller Jordan, arXiv:2404.00498](https://arxiv.org/abs/2404.00498) and [airbench architectures (DeepWiki)](https://deepwiki.com/KellerJordan/cifar10-airbench/2.3-neural-network-architectures) (airbench96 widens+deepens to 96%)
