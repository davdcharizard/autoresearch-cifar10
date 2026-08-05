# Proposal idea-01 (EXP-004): Add a residual block to layer2 + an extra conv to the layer2/3 stems (airbench96-style capacity bump), with the prescribed ~0.78× LR retune

## One-line summary
Bring OUR DavidNet a step toward the airbench96 architecture by (a) adding a `Residual` block to **layer2** — currently the only stage without one — and (b) reducing `PEAK_LR` by the documented airbench94→96 factor of **0.78×** (0.40 → 0.31) to keep the now-deeper net stable. Keep whitening + EMA + flip-TTA byte-identical (they are free and orthogonal). This is a *minimal, well-motivated* capacity increase (+2 conv layers, 8→10 learnable convs, matching airbench96's conv count) plus the single LR change a bigger net needs — not a width-and-depth-and-schedule kitchen sink. Honest framing: this is a genuine 2-sided bet (capacity vs. fewer annealed steps), and the analysis below puts the central estimate just at/over the 95.97% bar with meaningful downside.

## Limiter targeted
From the EXP-003 diagnosis chain and the EXP-002 idea-review's named remaining limiters ("robustness, capacity, whitening"): robustness was spent by EMA+TTA (EXP-002, +0.50pp), input conditioning was spent by whitening (EXP-003, +0.15pp), and the *one explicitly-deferred lever* is **representational capacity**. EXP-003's "Next Steps" lists the capacity probe directly, and the airbench lineage's own design (`knowledge/references/fast-cifar10-recipes.md` line 13: "the airbench family scales the network by accuracy target — the 96% config uses a larger net than the 94% config") says the canonical 95→96 step *is* a bigger net. We are sitting at 95.87% on the documented ~95/96 boundary; the airbench96 reference (arXiv:2404.00498) reaches 96.05% precisely by adding a third conv + residual per block.

Mechanistic chain to the metric: **more conv layers in the mid stages → a richer, deeper feature hierarchy at 16×16/8×8 resolution → a lower achievable train/test loss floor under the same annealed one-cycle → higher `best_test_acc`** — *conditional on* the deeper net still completing enough low-LR tail steps to reach that floor. That conditionality is the experiment. The LR reduction is part of the mechanism, not a free knob: a deeper residual stack has larger gradient-path gain, and airbench measured that the 10-conv net needs ~0.78× the 8-conv net's peak LR to avoid the tail destabilizing.

## The exact architectural change (which blocks/convs/widths)

### Reference (airbench96, fetched from `legacy/airbench96.py`)
- 3 convs per ConvGroup, **residual connection over the last two convs** of each block (`x = x + x0`, x0 saved after conv2).
- Widths: whiten=24 (kernel 2), block1=128, block2=384, block3=512; head `Linear(512,10)` × (1/9).
- 10 conv layers total (whiten + 9 learnable) vs the 94-variant's shallower net.
- lr 9.0/1024, epochs 37, batch 1024, warmup 10%, cutout 12, TTA-2, wd 0.012/1024, momentum 0.85. The 96-config's lr is **0.78×** the 94-config's lr; warmup shortened; decay-to-zero.

### Our current net (`train.py` lines 119–132, read in full)
Learnable convs today: `prep`(1) + layer1[`conv_bn`(1) + `Residual(128)`(2)] + layer2[`conv_bn`(1)] + layer3[`conv_bn`(1) + `Residual(512)`(2)] = **8** learnable convs (plus the frozen whiten conv). Spatial map: whiten/prep 32×32 → layer1 conv 32×32, MaxPool→16×16, Residual@16×16 → layer2 conv 16×16, MaxPool→8×8 → layer3 conv 8×8, MaxPool→4×4, Residual@4×4 → MaxPool(4)→1×1.

**layer2 is the only stage with no Residual block.** That is exactly the asymmetry airbench96 removes (it puts a residual in *every* block). It is also the cheapest place to add depth in terms of the capacity/throughput trade (see cost analysis): its two extra convs run at 8×8 resolution, 4× cheaper per channel than the 16×16 convs in layer1.

### The change (localized to lines 128–130 of `train.py`)
Replace the `layer2` definition so it gains a `Residual` block, matching the layer1/layer3 pattern exactly:

```python
self.layer1 = nn.Sequential(conv_bn(64, 128), nn.MaxPool2d(2), Residual(128))
self.layer2 = nn.Sequential(conv_bn(128, 256), nn.MaxPool2d(2), Residual(256))  # CHANGED: + Residual(256)
self.layer3 = nn.Sequential(conv_bn(256, 512), nn.MaxPool2d(2), Residual(512))
```

That single added `Residual(256)` adds **2 conv_bn layers** (8→10 learnable convs, matching airbench96's conv count), both at 8×8 resolution. No other module changes: `conv_bn`, `Residual`, `_weights_init`, `_forward_once`, `forward`, the whiten front-end, `pool` (still sees 512×4×4 → 512×1×1), and `fc=Linear(512,10)` are all untouched because the residual is channel-preserving (256→256) and spatial-preserving. The EMA `AveragedModel` wrapper (line 233) wraps whatever `model` is, so it inherits the new block automatically. `prepare.py`/`Eval.evaluate` is width/depth-agnostic (`model(inputs)`), untouched.

### The LR retune (line 21)
```python
PEAK_LR = 0.31  # CHANGED from 0.40: airbench94->96 reduces peak LR ~0.78x for the deeper residual net (0.40 * 0.78 = 0.312)
```
Rationale grounded in the reference: airbench96's lr is explicitly **0.78× the 94-config's lr**, and that reduction is paired with *exactly this kind* of depth+residual addition. A deeper residual stack increases the effective gradient gain through the network, so the same peak LR that was stable on the 8-conv net risks a noisier mid-training and a worse-annealed tail on the 10-conv net. 0.40 × 0.78 = 0.312 → use 0.31. This is the *one* required co-change; everything else (PCT_START=0.15 warmup, wd 5e-4, LS 0.2, Cutout 8, EMA 0.998, TTA gate 0.8, batch 512, time-based schedule, seed 42) stays fixed for clean attribution of "capacity + its required LR".

I deliberately do **not** also widen channels, shorten warmup, change cutout to 12, or switch to multi-crop TTA in this run — those are additional airbench96 deltas, but stacking them would confound the capacity test and is the "kitchen sink" the brief warns against. Depth-via-one-residual-block is the cleanest single capacity lever (it is the named airbench96 mechanism — "a residual connection over the last two convs of each block"), and it adds the *least* throughput cost per unit capacity by living at 8×8.

## Expected epochs vs current 174, and throughput/VRAM cost
The two added convs are `Conv2d(256,256,3)` at 8×8 = 64 spatial positions. Per-step added MACs ≈ 2 convs × 256×256×9×64 ≈ 0.6 GMAC/img forward (≈1.8 GMAC/img incl. backward). For calibration against the existing net's dominant terms: layer1's `conv_bn(64,128)`@32² is 128×64×9×1024 ≈ 0.075 GMAC and `Residual(128)`@16² is 2×128×128×9×256 ≈ 0.075 GMAC; layer3's `Residual(512)`@4² is 2×512×512×9×16 ≈ 0.075 GMAC. The layer2 Residual@8² is 2×256×256×9×64 ≈ 0.075 GMAC — i.e. it adds roughly **one more "0.075 GMAC unit"** to a forward pass that already has ~6–7 such units plus the whiten conv. So the FLOP increase is on the order of **~12–18%**, not a doubling.

- **Throughput**: expect ~0.83–0.88× of current img/s → **~145–155 epochs** (from 174). This is well above the ~100-epoch floor that EXP-003's analysis treated as safe for annealing (DavidNet's canonical recipe anneals in ~24 epochs). The time-based schedule (lines 264–268) guarantees the anneal *completes* regardless; the cost is fewer total optimizer steps in each LR phase.
- **VRAM**: the extra block's activations at 8×8 are tiny; expect peak VRAM to rise from ~1.61 GB to ~1.8–2.0 GB — trivially inside the 98 GB / non-binding soft constraint (EXP-001 insight: VRAM is free).
- **Wall-clock**: fewer epochs → *fewer* evals → wall-clock risk is *lower* than baseline, comfortably under the 600s kill.

**This must be confirmed by a throughput-only smoke probe** (launch the modified `train.py`, read live `img/s` + per-epoch wall from the first ~2 epochs, kill it). Gate: if projected epochs ≥ ~130, proceed with the official 300s run; if the hit is unexpectedly severe (<110 epochs), the under-annealing risk dominates and the run should be reconsidered. The smoke probe is throughput calibration only; its accuracy numbers are ignored and seed stays 42.

## Why this should lift best_test_acc past 95.97% (evidence)
- **airbench96 (arXiv:2404.00498, fetched source)**: the *documented, validated* 95→96 step is exactly "add a third conv + residual per block" with ~0.78× LR. The 96-config reaches 96.05% where its 94/95-configs plateau lower — direct evidence that this specific capacity addition is what carries the last point. We are porting the cheapest slice of that change (one residual block, matching the conv count) onto a recipe that already has whitening + EMA + TTA.
- **Whitening offsets the per-epoch cost (EXP-003, directly measured)**: EXP-003 showed whitening's large early-epoch lead (ep10 85.5% vs 81.6%) *survived running 9 fewer epochs*. The same mechanism applies here: a deeper net runs fewer epochs, but whitening accelerates its early convergence, partially buying back the lost annealing budget. This is the explicit reason the brief pairs capacity with the already-present whitening.
- **WideResNet (Zagoruyko & Komodakis 2016)** and the EXP-003 idea-03 capacity rationale: added capacity lowers the loss floor on CIFAR for wide-shallow residual nets; depth-via-residual is the airbench-preferred form here (residual eases the gradient flow so the added depth trains).
- **Free scaffolding (EXP-002/003)**: EMA (+0.50pp) and TTA and whitening (+0.15pp) all stay ON and are orthogonal to architecture; the EXP-002 report explicitly states they "compose additively with architectural upgrades." So this run inherits ~+0.65pp of denoising/conditioning for free and tests capacity on top of 95.87%.

## Strongest risk
**Under-annealing washes out the capacity gain** — the assumption that most needs to hold is that the 10-conv net at ~150 epochs reaches a *lower* annealed loss floor than the 8-conv net does at 174 epochs. Every gain in this goal's history (EXP-001/002/003) lives in the low-LR tail; spending ~20–25 epochs of that tail to buy 2 conv layers is the core gamble. If the deeper net needs proportionally more steps to converge (plausible — more parameters, same per-step signal, and the residual block initializes near-identity so it contributes little early), the tail is read under-trained and `best_test_acc` lands at or below 95.87% (no-improvement). This is the exact caution EXP-003 idea-03 flagged for capacity. Mitigation: the change is the *minimum* capacity bump (one 8×8 residual block, ~12–18% FLOPs, not a width multiplier that would cost ~56%+), and the smoke gate ensures ≥~130 epochs before committing.

**Secondary risk — the LR retune is itself a confound / could be mistuned.** I am changing two things (depth + LR), so a no-improvement cannot cleanly separate "capacity doesn't help" from "0.31 was the wrong LR for *our* net". I judge the LR co-change *necessary* (the reference prescribes it for this exact architecture delta, and holding 0.40 risks tail instability on the deeper net), but 0.78× is airbench's number for a *full* block-triple-ing, not our single added block — our smaller depth change might prefer a milder reduction (e.g. 0.36). The clean-attribution cost is real; I accept it because shipping a deeper net at a known-unstable LR is the worse failure mode. A no-improvement here should be read as "this depth+LR point didn't clear the bar", not "capacity is dead".

**Tertiary risk — residual-block init contributes little within budget.** `_weights_init` (line 137) kaiming-inits the new convs; the `Residual` is `x + c2(c1(x))`, so early in training the block is a near-no-op plus noise and only earns its capacity late. Under a shortened (fewer-epoch) budget the new block may not "warm up" enough to pay off. (airbench96 uses Dirac/identity init to mitigate exactly this — an untried future rider, deliberately out of scope to keep the change minimal.)

## Honest expected-magnitude estimate vs the 95.97% bar
I will not inflate this. Outcome distribution, central case = the layer2-residual + LR 0.31 variant:
- **~35%**: clears with margin (95.97–96.2%) — capacity was binding, ~150 epochs annealed enough, whitening bought back the early epochs. This is the airbench96-supported upside.
- **~35%**: lands 95.85–95.97% — marginal, often inside the bar's noise; capacity helps a little but the lost ~20 epochs eat most of it.
- **~30%**: no-improvement (≤95.87%) — under-annealing or LR mismatch dominates.

Central estimate ≈ **95.95–96.0%**, i.e. *right at* the +0.1pp bar with meaningful variance. This has a **higher ceiling** than a pure LR sweep (a clean, reference-backed path toward 96%) but **higher variance** than EXP-003's whitening was, because it spends a proven lever (annealed step count) to buy an unproven one (capacity at this scale). The +12–18% FLOP cost is materially gentler than the 1.25× width idea's ~56% (EXP-003 idea-03), so the epochs-vs-capacity trade is more favorable here than in the width variant — which is the main reason I prefer one-residual-block depth over a width multiplier.

## Effort
**Low.** ~1 line added in `ResNet9.__init__` (the `Residual(256)`) + 1 line changed (`PEAK_LR`), no new modules, no new deps, no loop/schedule/eval/EMA/TTA changes, no new control flow. One ~30–60s throughput smoke probe to confirm ≥~130 projected epochs, then one official 300s run on GPU 1 (`CUDA_VISIBLE_DEVICES=1`). The only care needed is confirming the smoke-probe epoch projection before committing.
