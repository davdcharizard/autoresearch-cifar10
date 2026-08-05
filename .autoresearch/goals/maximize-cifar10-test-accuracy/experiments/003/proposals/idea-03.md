# Proposal idea-03 (EXP-003): Increase DavidNet capacity (width multiplier) within the 300s budget

## One-line summary
Widen the DavidNet/ResNet-9 channel counts by a uniform multiplier (primary variant **1.25×**), keeping the proven EXP-001+EXP-002 recipe byte-identical, on the bet that at 95.72% the binding limiter has shifted from optimization/regularization to representational **capacity**. Honest verdict up front: this is a genuinely 2-sided bet — width buys capacity but spends update count, and our entire EXP-001/002 gain history lives in the low-LR tail, so under-annealing is the real failure mode. I recommend 1.25× (not 1.5×) as the first variant and gate the decision on a throughput smoke test.

## Limiter targeted
From the diagnosis chain: EXP-001 (architecture swap + completing one-cycle) gave +3.65pp and showed the bulk of accuracy arrives in the low-LR tail (under-annealing was the limiter). EXP-002 (weight-EMA + flip-TTA) gave +0.50pp by **denoising the evaluated tail iterate** — i.e. it attacked the *tail-noise / robustness* limiter, and that limiter is now largely spent (EMA averages the tail; TTA averages flip variance). The remaining limiters the EXP-002 idea-review named are "robustness, capacity, and whitening" (idea-review item 8). EMA+TTA addressed the robustness slice. This proposal targets the **capacity** slice: a 6.57M-param wide-shallow net may have a representational ceiling below the ~96% airbench-95/96 regime that the EXP-001 report explicitly noted we are sitting on the edge of.

The mechanistic claim: more channels → a richer per-layer feature basis → a lower achievable train/test loss floor at the *same* annealed schedule, provided the wider net still completes enough low-LR steps to reach its (lower) floor. The causal chain to the metric is "more channels → lower loss floor → higher `best_test_acc`", but it is conditional on "enough tail steps survive the throughput hit". That conditionality is the whole experiment.

## Why this is promising (evidence)
- **WideResNet (Zagoruyko & Komodakis 2016)**: width is a first-class accuracy lever on CIFAR — WRN-28-10 (widen factor 10) substantially outperforms thin-deep ResNets at matched-or-lower depth, and the paper's central thesis is that for CIFAR, *widening* a shallow residual net is a more compute-efficient way to add capacity than deepening. Our net is already the wide-shallow archetype the paper favors; nudging the widen factor up is exactly the lever they validate.
- **airbench (Keller Jordan, arXiv:2404.00498)**, per `knowledge/references/fast-cifar10-recipes.md`: the airbench family *scales the network by accuracy target* — the 96% config uses a larger net than the 94% config. We are explicitly trying to climb from the 95% plateau toward 96%, which in that lineage's own design corresponds to spending some of the compute envelope on a bigger net.
- **Project insight (EXP-001, High importance)**: "Compute envelope is huge: H20 98 GB, baseline uses ~1.6 GB — memory & throughput are free levers." VRAM is a non-binding constraint, so a 1.25× width net (~1.56× params, ~10M; ~1.56× conv FLOPs) stays trivially inside VRAM. The only real cost is throughput → epochs.
- **EXP-001 report (§Observations)** itself flags "wider/deeper net" as a top unexplored avenue precisely because "only 1.6 GB VRAM used."
- **EXP-002 report §Unexplored Avenues** notes EMA+TTA is orthogonal and "should compose additively with architectural upgrades... The next capacity experiment can keep EMA+TTA on for free." So we inherit the +0.50pp EMA/TTA scaffolding for free and test capacity on top of it.

## Why 1.25× and not 1.5× (the throughput/epoch trade-off)
The cost is conv FLOPs, which scale ~quadratically in the width multiplier `w` for the conv layers (both `c_in` and `c_out` scale by `w`). The dominant FLOP cost in DavidNet is the early/mid stages at high spatial resolution:
- prep (3→64 @ 32²) — `c_in` is fixed at 3, so this stage scales only ~linearly in `w`.
- layer1 conv (64→128 @ 32²) + Residual(128) two convs @ 16² — scales ~`w²`.
- layer2 conv (128→256 @ 16²) — scales ~`w²`.
- layer3 conv (256→512 @ 8²) + Residual(512) two convs @ 4² — scales ~`w²` but at small spatial size, cheaper per channel.

So total conv compute scales a bit below `w²` (prep is linear, but it is a small fraction). Empirically WRN/DavidNet timing tracks ≈`w²`:
- **1.25×** → ≈1.56× FLOPs → throughput ≈ 1/1.56 ≈ 0.64× → epochs ≈ 183 × 0.64 ≈ **~115 epochs**.
- **1.33×** → ≈1.77× → epochs ≈ **~103**.
- **1.5×** → ≈2.25× → epochs ≈ 183 × 0.44 ≈ **~80 epochs**.

EXP-001 reached 95.05% by ep187 and the EMA+TTA tail in EXP-002 was still climbing at ep178. The schedule is *time-based*, so it always completes its anneal — but a heavier net does fewer total optimizer steps, so each phase of the cycle (including the critical low-LR tail) gets fewer updates. At ~115 epochs the net still gets ~60% of the baseline's steps, which historically (DavidNet canonical recipe is only ~24 epochs at 94%) is well above the floor needed to anneal. At ~80 epochs (1.5×) the risk that the wider net is read *under-converged* — washing out or reversing the capacity gain — is materially higher and is exactly the EXP-002 idea-review's warning (item 8: "pure width trades away the update count that produced EXP-001's late gain"). So **1.25× first**; promote to 1.5× only if a smoke test shows it still lands ~100+ epochs.

## Concrete code change to `train.py`
The change is localized to the `ResNet9` module (lines 88–119) plus its `conv_bn`/`Residual` helpers (lines 70–85), all of which I read in full. Everything else — the training loop, LR schedule, EMA wrapping, TTA gating, optimizer, augmentation — stays byte-identical for clean attribution.

**1. Add a width constant** next to the other hyperparameters (after `NUM_CLASSES = 10`, line 19):
```python
WIDTH_MULT = 1.25  # channel width multiplier on the DavidNet base widths (64/128/256/512)
```

**2. Parameterize `ResNet9.__init__`** (currently lines 88–99). Replace the hard-coded channel literals with multiplier-scaled, round-to-even widths so BN/conv channel counts stay integers and channels_last stays efficient:
```python
class ResNet9(nn.Module):
    def __init__(self, num_classes=10, scale_out=SCALE_OUT, width_mult=WIDTH_MULT):
        super().__init__()
        self.scale_out = scale_out

        def w(c):  # scaled width, rounded to a multiple of 8
            return max(8, int(round(c * width_mult / 8)) * 8)

        c1, c2, c3, c4 = w(64), w(128), w(256), w(512)
        self.prep = conv_bn(3, c1)
        self.layer1 = nn.Sequential(conv_bn(c1, c2), nn.MaxPool2d(2), Residual(c2))
        self.layer2 = nn.Sequential(conv_bn(c2, c3), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(conv_bn(c3, c4), nn.MaxPool2d(2), Residual(c4))
        self.pool = nn.MaxPool2d(4)
        self.fc = nn.Linear(c4, num_classes, bias=False)
        self.tta = False
        self.apply(self._weights_init)
```
At `width_mult=1.25`: `w(64)=80, w(128)=160, w(256)=320, w(512)=512` (round-to-8: 512*1.25/8=80 → 640; recompute below). Note rounding: 64*1.25=80, 128*1.25=160, 256*1.25=320, 512*1.25=640 — all already multiples of 8 at 1.25×, so `c1,c2,c3,c4 = 80,160,320,640` and `fc` becomes `Linear(640, 10, bias=False)`.

`conv_bn`, `Residual`, `_weights_init`, `_forward_once`, and `forward` need **no change** — they already take channel counts as arguments and `Residual` keys off a single `c`. The EMA wrapper (`AveragedModel(model, ...)`, line 178) wraps whatever `model` is, so it inherits the new widths automatically. The frozen eval (`prepare.py` `Eval.evaluate` → `model(inputs)`) is untouched and width-agnostic.

**3. Nothing else changes.** `PEAK_LR=0.4`, `WEIGHT_DECAY=5e-4`, `LABEL_SMOOTHING=0.2`, `Cutout(8)`, `PCT_START=0.15`, `EMA_DECAY=0.998`, `TTA_START_FRAC=0.8`, batch 512 — all held fixed (see "Risk" for why holding them is the right default but also a risk).

Param count at 1.25× (80/160/320/640): the dominant terms scale ≈1.56×, giving ≈10.3M params (vs 6.57M). VRAM at batch 512 should land ≈2.5–3.5 GB — far under 98 GB.

## Recommended first variant + smoke gate
Run **1.25× (80/160/320/640)** as the primary. Before committing the official 300s run, do a *throughput-only* smoke probe (NOT a full run, NOT a seed search): launch the modified `train.py` and read the live `img/s` and per-epoch wall from the first ~2 epochs of stdout, then kill it. If projected epochs ≥ ~100, 1.25× is safe; if the throughput hit is milder than the `w²` estimate (e.g. memory-bandwidth-bound rather than FLOP-bound on this H20, which is plausible at only 1.6 GB working set) and projected epochs ≥ ~140, consider promoting to 1.33× or 1.5× in a follow-up. Keep `manual_seed(42)`; the smoke probe is for throughput calibration only and its accuracy numbers are ignored.

## Strongest risk
**Under-annealing washes out the capacity gain.** The assumption that most needs to hold: at ~115 epochs the 1.25× net still reaches a *lower* annealed loss floor than the 6.57M net does at 183 epochs. If the wider net needs proportionally more steps to converge (plausible — more parameters, same per-step gradient signal), the tail may be read under-trained and `best_test_acc` lands *at or below* 95.72%, i.e. no-improvement. This is the EXP-002 idea-review's explicit caution (item 8) and contradicts the safe-bet framing of EMA+TTA. Mitigation is the smoke gate and the conservative 1.25× choice; but it cannot be fully retired without running it.

**Secondary risk — LR/WD mismatch at higher width.** `PEAK_LR=0.4` and `wd=5e-4` were tuned on 6.57M params (idea-review item 9). Wider nets sometimes prefer a slightly lower peak LR or different effective wd. I recommend holding them fixed for clean single-variable attribution (so a win/loss is unambiguously "capacity"), but flag that a sub-bar result could be a *tuning* artifact rather than a true capacity ceiling — so a no-improvement here should NOT be read as "capacity doesn't help", only as "capacity at this fixed recipe and this step budget doesn't help." Do **not** stack the optional layer2-residual depth change in the same run (item 9) — that would confound width vs depth.

**Tertiary risk — wall-clock.** Eval is outside the 300s training budget but counts against the 600s wall cap. Fewer epochs (~115 vs 183) means *fewer* evals, and the per-eval forward is wider but still cheap, so wall-clock risk is *lower* than EXP-002, not higher. Not a real concern.

## Honest expected-magnitude estimate vs the 95.82% bar
This is the shakiest of the EXP-003 candidate ideas on expected value, and I will not inflate it. Realistic outcome distribution centered on the 1.25× variant:
- **~40%**: lands in 95.7–95.9% — i.e. roughly flat to a marginal win, often *inside* the +0.1pp bar's noise. Capacity helps a little but the lost ~70 epochs of annealing eat most of it.
- **~30%**: clears the bar with margin (95.9–96.2%) — capacity was genuinely binding and 115 epochs was enough to anneal. This is the upside the WRN/airbench evidence supports.
- **~30%**: no-improvement (≤95.72%) — under-annealing dominates, or LR/wd mismatch caps the tail.

Central estimate ≈ **95.85%**, i.e. *just* at the bar with high variance. Honest framing: the expected delta is smaller and far higher-variance than EXP-002's was, because we are spending a proven lever (update count / annealing) to buy an unproven one (capacity at this scale). If the brainstorm has a lower-variance candidate (e.g. whitening front-end, or a cheap EMA-decay/TTA-gate sweep on the now-higher 95.72% base), that may be the better EXP-003 pick on expected value — but this idea has the highest *ceiling* (a clean path toward 96%) and is the natural test of whether capacity is now the limiter. Recommend running it only if the loop wants to probe the capacity ceiling specifically, with 1.25× and the smoke gate.

## Effort
**Low.** ~6 lines changed in one module, no new dependencies, no loop/schedule/eval changes, no new control flow. One throughput smoke probe (~30s) plus one official 300s run. The only non-trivial care is the round-to-8 width helper to keep channel counts integer and channels_last-friendly.
