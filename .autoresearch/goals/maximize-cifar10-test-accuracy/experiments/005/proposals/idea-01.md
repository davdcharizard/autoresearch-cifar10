# Proposal idea-01 — Capacity via WIDTH: widen block2 256 → 384 (airbench96 width)

## One-line
Widen the layer2 stage from 256 to 384 channels (matching airbench96's `block2=384`), keeping the EXP-004 ReZero gate and `PEAK_LR=0.4` fixed, to add representational capacity at the cheap-resolution 8×8 stage and push `best_test_acc` past 96.10%.

## Mechanism (causal chain to the metric)
EXP-004 established that **representational capacity is a binding lever** at this scale: adding one `GatedResidual(256)` to layer2 lifted 95.87 → 96.00 (+0.13pp), and the capacity gain *outran* a 32-epoch (174→142) throughput cost. The diagnosis names capacity as the limiter; EXP-004 spent it on **depth**. This idea spends the *same lever on width*, the specific next step the airbench96 reference documents.

Causal chain:
1. Layer2 operates at 8×8 (the GatedResidual) and 16×16 (its input conv) — the cheapest spatial resolutions in the net for adding channels (FLOP ∝ H·W). At 8×8 there are only 64 spatial positions, so a channel-count increase there is far cheaper per added parameter than the same increase at 16×16 or 32×32.
2. Going 256→384 makes block2's two residual convs 2.25× wider (channel² scaling) and the two bracketing convs (layer2-input and layer3-input) 1.5× wider. Net forward FLOPs rise ~34% (498M→668M, computed below), giving the block ~+2.2M params (7.78M→~10.0M).
3. Wider channels at the block that already proved capacity-hungry (EXP-004's gate ramped α off zero and the lead emerged exactly there) give the network a richer 8×8 feature basis. More mid-level features → lower annealed train/test loss floor → higher `best_test_acc`, *provided the one-cycle still anneals far enough* within the reduced epoch count.
4. The ReZero gate (α=0 init) on the now-wider `GatedResidual(384)` keeps the wider block an exact identity at init, so the net starts bit-equivalent to a proven net and earns the extra width gradually — no LR retune, no early-trajectory disruption (the EXP-004 lesson, validated: ep10 matched within noise).

## Why width (not more depth) at this 8×8 stage — per-FLOP efficiency
EXP-004 added **depth** (2 convs, +75.5M FLOPs, +1.18M params, +0.13pp). This adds **width** to the same stage. Per-FLOP comparison at the 8×8 residual convs:
- Depth (EXP-004): 2 new 256×256 convs = +75.5M FLOPs for +1.18M params.
- Width (this): widening the existing 2 residual convs 256→384 = +94.3M FLOPs for +1.47M params, *plus* the two bracket convs widen for "free-ish" capacity (+0.74M params, +75.5M FLOPs).

Width is slightly *less* FLOP-efficient per residual-conv parameter (wider convs cost C² while deeper convs add a fixed C² each), BUT width also widens the **bracket convs** (layer2-input, layer3-input), broadening the information channel *into and out of* the block — depth cannot do that. The literature view: at a fixed shallow ResNet-9 depth, the airbench lineage scaled the 95→96 step via **both** an extra conv/residual per group AND wider groups (128/384/512). airbench96 chose 384 for block2 specifically, suggesting width at this stage is on the documented frontier. The honest read: width and depth are comparable per-FLOP here; airbench96's choice of 384 is the concrete evidence that *this particular* width is productive, which is why it's worth a dedicated probe distinct from EXP-004's depth.

## Concrete change (THIS codebase)
Single architectural edit in `train.py`, in `ResNet9.__init__` (lines 149–151):

Current (line 150–151):
```python
self.layer2 = nn.Sequential(conv_bn(128, 256), nn.MaxPool2d(2), GatedResidual(256))
self.layer3 = nn.Sequential(conv_bn(256, 512), nn.MaxPool2d(2), Residual(512))
```
Change to:
```python
self.layer2 = nn.Sequential(conv_bn(128, 384), nn.MaxPool2d(2), GatedResidual(384))
self.layer3 = nn.Sequential(conv_bn(384, 512), nn.MaxPool2d(2), Residual(512))
```

That's it. Three numbers change: `conv_bn(128,256)→(128,384)`, `GatedResidual(256)→(384)`, and layer3's input `conv_bn(256,512)→(384,512)`.

Spatial/channel plumbing verified against the actual code:
- `conv_bn` (lines 101–106) is `Conv2d(c_in,c_out,3,padding=1)` + BN + ReLU; padding=1 preserves spatial dims → no pool-chain change.
- `GatedResidual` (lines 119–137) is channel- and spatial-preserving (`c1=conv_bn(c,c)`, `c2=conv_bn(c,c)`, `x+α·c2(c1(x))`), so 384→384 is a drop-in.
- layer3's `Residual(512)` (line 151, via `Residual` lines 109–116) is unchanged; its input conv just consumes 384 instead of 256.
- **layer3 still outputs 512** → `self.pool = MaxPool2d(4)` (line 152) on the 4×4 map and `self.fc = Linear(512,10)` (line 153) are byte-untouched. whiten/prep/layer1 untouched.
- `_weights_init` (lines 157–160) kaiming-inits the new wider convs automatically via `self.apply`. The ReZero α stays `nn.Parameter(torch.zeros(1))` → identity at init.
- Optimizer (lines 243–249) builds its param group from `model.parameters()` after the architecture is defined, so the wider params are picked up automatically. EMA `AveragedModel` (lines 254–256) wraps `model` → also automatic. **No other line in `train.py` needs to change.**

Everything else held fixed for clean single-variable attribution: `PEAK_LR=0.4`, schedule, `WEIGHT_DECAY=5e-4`, `LABEL_SMOOTHING=0.2`, `Cutout(8)`, `BATCH_SIZE=512`, EMA 0.998, flip-TTA gate, `torch.manual_seed(42)`, whitening front-end — all unchanged.

## Throughput / epoch estimate (the decisive number)
Forward-FLOP accounting (FLOP ∝ H²·C_in·C_out·9; resolutions from tracing the pool chain: prep/whiten 32², layer1-conv 32² then pool→16², residual1 16², layer2-conv 16² then pool→8², gatedres2 8², layer3-conv 8² then pool→4², residual3 4²):

| Term | 256 (current) | 384 (new) |
|---|---|---|
| whiten 32²·3·54 | 13.4M | 13.4M |
| prep 32²·54·64 | 31.9M | 31.9M |
| layer1 conv 32²·64·128 | 75.5M | 75.5M |
| residual1 ×2 @16² | 75.5M | 75.5M |
| **layer2 conv 16²·128·W** | 75.5M | **113.2M** |
| **gatedres2 ×2 @8²·W·W** | 75.5M | **169.8M** |
| **layer3 conv 8²·W·512** | 75.5M | **113.2M** |
| residual3 ×2 @4² | 75.5M | 75.5M |
| **Total** | **498.3M** | **668.1M** |

Ratio = **1.341** (forward FLOPs +34%).

Epoch translation, anchored to EXP-004's measured FLOP→epoch elasticity: EXP-004 added +75.5M to a 422.8M base (+17.9% FLOPs) and epochs fell 174→142 (−18.4%) — i.e. epochs scale ~inversely with FLOPs (the loop is compute-bound: ~26k img/s, VRAM 1.6GB, good overlap). Applying that elasticity from the 96.00% baseline (498.3M / 142 ep):

**142 × (498.3 / 668.1) ≈ 106 epochs.**

This lands **below the ~110–120 under-annealing danger zone** the idea itself flagged. That is the central risk (next section). For reference: EXP-001 ran 192 ep, EXP-003 174 ep, EXP-004 142 ep — accuracy held while epochs fell because each step added capacity, but we are now extrapolating *past* the lowest epoch count yet tried.

## VRAM
Params +2.2M (→~10.0M) and activations for the wider 8×8/16×16 maps add a few hundred MB at most. EXP-004 ran at 1635MB of 98GB. Estimate ~1.9–2.1GB peak — far inside budget. VRAM is a non-issue (confirmed pattern, EXP-001).

## Evidence
- **airbench96 (Keller Jordan, arXiv:2404.00498; legacy/airbench96.py)**: `'widths': {'block1':128, 'block2':384, 'block3':512}`, 10 conv layers, reports **96.03% avg over n=400 runs**. This is direct evidence that block2=384 is a productive width on the *exact* DavidNet-lineage architecture we run. (Caveat: airbench96 uses peak LR 9.0/1024, batch 1024, 37 epochs, GELU — a fully different schedule/width-everywhere net; the 384 number transfers, the LR does not — see risks.) Verified via WebFetch of the raw source this session.
- **EXP-004 (experiments/004/04-analysis.md §Results)**: capacity is binding at 95.87%; the layer2 GatedResidual lead emerged *at this very block* (ep25 92.63 vs 88.84) and outran a 32-epoch cost. Widening the same block is the most-supported next capacity move.
- **ReZero (rezero-identity-init.md; Bachlechner 2020, arXiv:2003.04887)**: α=0 identity-init lets us add capacity without LR retune or early-trajectory disruption — validated in EXP-004 (ep10 within noise of EXP-003). The wider `GatedResidual(384)` reuses this exact gate, so `PEAK_LR=0.4` stays fixed for clean attribution.
- **EXP-001 patterns**: VRAM is free; "most accuracy arrives in the low-LR tail of a *completing* one-cycle" (03-experiment-learnings.md, Medium). This last pattern is precisely what the epoch drop threatens.

## Strongest risk (and the central tradeoff)
**Under-annealing eats the capacity gain.** The estimate puts this run at **~106 epochs**, below the ~110–120 zone where the completing-one-cycle tail still does its work. The goal's most-cited mechanism is "most accuracy is in the low-LR tail." At 106 epochs the same wall-clock 300s budget is split over fewer, more-expensive steps; the schedule still *anneals to ~0* (it is time-keyed, lines 285–289, so the LR always reaches ~0 at the budget end), but the net sees **fewer gradient steps in the low-LR tail**, so the wider net may not converge far enough into its (larger) loss basin to realize its capacity. EXP-004's gain depended on the capacity advantage *exceeding* the lost-annealing cost; here the FLOP hit (+34%) is ~2× EXP-004's (+18%), so the margin is thinner and the bar (96.10) is +0.10 above an already-tight 96.00.

Assumption that most needs to hold: **the wider block's lower achievable loss floor more than compensates for ~36 fewer epochs of annealing.** EXP-004 cleared a similar bet at half the FLOP cost; doubling the cost makes it materially less certain.

Secondary risks: (a) the time-keyed schedule's single-step overshoot is negligible; (b) wd 5e-4 is unchanged on +2.2M params — fine, the recipe was robust to the EXP-004 param add; (c) BN stats EMA at 0.998 over fewer steps — minor.

## Honest quantitative estimate
- **Central**: ~96.02–96.08%. The capacity is real and at the right block, but I expect under-annealing at ~106 epochs to **partially cancel** it, landing near or just below the bar.
- **Optimistic** (if the wider basin anneals fast, like EXP-004's mid-training lead): ~96.10–96.18%.
- **Pessimistic** (under-annealing dominates, as the 34% FLOP hit is large): ~95.90–96.00% — capacity gain fully eaten, no improvement.

**Probability of clearing 96.10: ~30–35%.** This is lower than EXP-004's bet because the FLOP cost roughly doubled while the headroom to the bar shrank. The honest read: **this is shakier than the idea framing suggests** — the ~106-epoch estimate sits in the danger zone the idea itself names, and the central estimate straddles the bar rather than clearing it. A more conservative width (e.g. 320) would land ~118 epochs (safer annealing, less capacity) and might be the better risk-adjusted probe; but 384 is the *documented* airbench96 width and gives the cleanest "does the reference width transfer" answer. If this run lands ~96.0–96.08 (capacity visible but annealing-starved), the natural follow-up is to *recover epochs* (cheaper stem / smaller width like 320) rather than abandon width.

## Effort
**Low.** Three integer edits to two lines (149–151) of `train.py`; no new class, no schedule change, no new deps. One training run (~7.5 min wall under the 600s timeout: ~106 ep × eval + 300s training + startup). Identical run/verify protocol as EXP-004 (`CUDA_VISIBLE_DEVICES=1 uv run train.py`). Pre-run smoke worth doing (cheap): confirm forward shapes (512×4×4 pool input, 384 channels through layer2), `α.grad ≠ 0` after one backward on the wider gate, and learnable param count ≈10.0M.
