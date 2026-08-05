# Idea-03: Capacity via more depth — a SECOND ReZero-gated residual block in layer3 (4×4)

**Goal**: maximize `best_test_acc` on CIFAR-10 in a fixed 300s training-time budget, editing only `train.py`. Baseline 96.00% (EXP-004). Bar ≥96.10% (+0.1pp).

**One-line claim**: Stack the just-validated EXP-004 lever — add ONE more ReZero-gated residual block — placing it in `layer3` at 4×4 resolution, where it is the cheapest-throughput place to add capacity (smallest activation footprint of the three stages) while costing the same FLOPs as an 8×8 block. Honest central estimate ~96.05–96.10%, i.e. **right at the hard bar**; this is a genuine but diminishing-returns extension.

---

## 1. Mechanism — how the change moves the metric, tied to the limiter

**Named limiter (from EXP-004 diagnosis + result)**: representational capacity. EXP-004 confirmed capacity is *still binding* at this scale/budget — adding one ReZero-gated `Residual(256)` to layer2 moved 95.87→96.00 (+0.13pp), and the trajectory showed the gain came from a genuinely lower annealed loss floor (ep25 92.63% vs 88.84% for the 8-conv net), not from a schedule artifact. The block's capacity uptake (α ramping off 0) is the mechanism, and it outran the 32-epoch throughput cost.

**Causal chain for this idea**:
1. Add a second ReZero-gated 2-conv residual block → net goes from 10 to 12 learnable convs.
2. ReZero init (`α=0`) means the deeper net starts **bit-identical** to the proven 96.00% net (EXP-004 `_forward_once` is unchanged at init), so it inherits the full proven early-convergence trajectory and needs **no LR retune** (PEAK_LR stays 0.4). This is exactly the property EXP-004 validated on its trajectory (ep1/ep10 matched EXP-003 within noise).
3. As `α` moves off zero, the extra block adds a deeper non-linear feature transform → lower achievable annealed loss floor → higher `best_test_acc`, **conditional** on the deeper net still completing enough low-LR tail steps (the under-annealing risk).
4. The live gradient path `∂L/∂α = ⟨grad_out, c2(c1(x))⟩ ≠ 0` (ReZero, Bachlechner et al. 2020) is what makes step 3 happen at all — the block trains rather than staying a frozen identity.

**Why layer3 (4×4) is the chosen placement — the capacity-per-throughput argument**:

The decisive, non-obvious fact is that **all three candidate placements cost identical FLOPs** because DavidNet's channel²·spatial product is invariant across stages (channels double each time spatial halves twice):
- GatedResidual(128) @ 16×16: 2·128²·9·256 = 75.5M MAC/img
- GatedResidual(256) @ 8×8: 2·256²·9·64 = 75.5M MAC/img  ← the EXP-004 block
- GatedResidual(512) @ 4×4: 2·512²·9·16 = 75.5M MAC/img  ← **this proposal**

So FLOPs do not differentiate the placements. What *does* differ is the **activation footprint** (and therefore the memory-traffic / kernel-launch cost that dominates these small convs at batch 512):
- layer1 16×16×128 = 32768 elem/img/tensor (largest)
- layer2 8×8×256 = 16384 elem/img/tensor (EXP-004's block)
- layer3 4×4×512 = 8192 elem/img/tensor (**smallest — half of layer2**)

At 4×4 the convs touch the fewest activation bytes and run on the largest channel count (best arithmetic intensity / GPU utilization). EXP-004's 8×8 block cost ~11% throughput (29.3k→26.3k img/s). A 4×4 block has the same FLOPs but ~half the activation traffic, so its throughput hit should be **≤ the 8×8 block's**, plausibly notably less — buying back annealing budget exactly where the diagnosis says gains live. Capacity-wise, a 512-channel block is the **widest** transform we can add, i.e. the most representational power per block. layer3 currently has only a kaiming-init `Residual(512)` and no second block — there is real room.

This makes layer3 the best capacity-per-throughput buy of the three depth placements.

---

## 2. Concrete change in THIS codebase

All edits in `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/train.py`. The `GatedResidual` class already exists (lines 119–137) and is reused verbatim — **no new class needed**.

**Single line change — `ResNet9.__init__`, line 151**:
```python
# current (line 151):
self.layer3 = nn.Sequential(conv_bn(256, 512), nn.MaxPool2d(2), Residual(512))
# proposed:
self.layer3 = nn.Sequential(conv_bn(256, 512), nn.MaxPool2d(2), Residual(512), GatedResidual(512))
```

That is the entire diff. Channel/spatial trace confirming downstream is untouched:
- `conv_bn(256,512)` outputs 512×8×8 → `MaxPool2d(2)` → 512×4×4 → `Residual(512)` → 512×4×4 → **`GatedResidual(512)` → 512×4×4** (channel- and spatial-preserving, `forward: x + alpha*c2(c1(x))`, c1/c2 are `conv_bn(512,512)` pad-1 so spatial is preserved).
- `self.pool = nn.MaxPool2d(4)` (line 152) still receives **512×4×4 → 512×1×1** ✓, `flatten(1)` → 512, `self.fc = nn.Linear(512,10)` (line 153) unchanged ✓.

Everything else byte-identical: `PEAK_LR=0.4` (line 21), schedule (lines 285–289), EMA (lines 254–256, 307–309), flip-TTA gate (lines 340–347), whitening (lines 146–147, 234–235), `torch.manual_seed(42)` (line 199), Cutout 8, batch 512, bf16. The new `α` scalar auto-joins the SGD param group via the `requires_grad` filter at line 244 (WD on a scalar is negligible). The frozen whiten conv stays excluded (`requires_grad_(False)`).

**Param/VRAM delta**: +2·(512·512·9) conv weights + 2·512 BN params + 1 α scalar ≈ +4,720,640 params → ~12.5M total (from EXP-004's 7,784,627). VRAM: layer3 activations are the smallest per-stage tensor; EXP-004 was 1635 MB; expect ~1.7–1.9 GB, non-binding vs 98 GB.

**Required smoke before the official run** (the EXP-004 lesson — verify the gate is alive, do NOT trust filenames):
1. One-step backward: instantiate `ResNet9`, forward a random batch, `loss.backward()`, assert the **new** layer3 `GatedResidual.alpha.grad` is not None and ≠ 0 (proves the block is trainable, not a dead identity).
2. Identity-at-init: with α=0, assert `_forward_once` output equals the EXP-004 net's (the block is exact identity at init).
3. Shape check: assert pool input is 512×4×4 and logits are [B,10].
4. Throughput read: launch the modified `train.py`, read live `img/s` over the first ~2 epochs, kill it. Gate: if projected epochs ≥ ~120, proceed; if < ~110, flag under-annealing risk for analysis (do not auto-abort — the time-based schedule still completes the anneal). Seed stays 42; smoke accuracy numbers are ignored.

---

## 3. Evidence

- **EXP-004 (direct, strongest)** (`experiments/004/04-analysis.md`): the *same lever* (one ReZero-gated block) gave +0.13pp and the analysis explicitly pre-registers "a second gated block ... ReZero init makes added depth safe to stack without LR retune" as the natural next probe (Unexplored Avenues, Next Steps). The trajectory mechanism (identity start → mid-training capacity lead → higher tail) is documented and is the template for this run's falsification check.
- **ReZero (Bachlechner et al. 2020, arXiv:2003.04887)** (`knowledge/references/rezero-identity-init.md`): the gate sits outside the post-ReLU, giving identity-at-init *with* a live gradient — the property that makes stacking safe and LR-retune-free. The reference explicitly names "stack a second gated block" as a sanctioned reuse.
- **airbench / fast-cifar10 lineage (arXiv:2404.00498)** (`knowledge/references/fast-cifar10-recipes.md`, EXP-004 brainstorm §lit-review): the documented 94→96 step is a *bigger network* (extra conv + residual per block, 10 convs); airbench96 reaches 96.05% at this scale. We are pushing past airbench96's conv count (12 vs 10), which is consistent with capacity still helping but also flags that we are near the documented ceiling for ~minute-scale training.
- **FLOP-invariance computation (code-grounded, this proposal §1)**: traced from the actual `ResNet9` stage definitions (lines 149–153) and conv shapes — 256²·64 = 512²·16 = 128²·256 = 4.19M, so layer3 placement is FLOP-equal to EXP-004's proven 8×8 block but has half the activation traffic → throughput hit ≤ EXP-004's measured 11%.

---

## 4. Contrast with widening (idea-01) — depth-at-4×4 vs width-at-8×8

idea-01 explores adding **width**. The capacity-per-FLOP comparison:
- Widening a stage's channels by factor `w` multiplies that stage's conv FLOPs by `w²` (both `c_in` and `c_out` grow). A 1.25× width bump on layer2 is ~1.56× that stage's FLOPs — a heavier, non-identity-initializable change (you cannot byte-match the proven net, so early convergence is disrupted and an implicit LR interaction appears).
- This depth idea adds a **fixed +75.5M MAC block that is FLOP-equal to EXP-004's proven block**, is **ReZero-identity-initializable** (zero early disruption, no LR retune, bit-equivalent start), and at 4×4 has the **lowest activation traffic** of any placement.

So on capacity-per-FLOP *and* safety-of-init, **depth-at-4×4 dominates width-at-8×8 here**: equal/known FLOP cost, a proven init recipe, and the smallest throughput penalty. Width's only edge is that it grows capacity without adding depth (avoiding any deep-net optimization difficulty), but ReZero already neutralizes that concern. The honest caveat is that both are fighting the same diminishing-returns ceiling (§5).

---

## 5. Strongest risk + honest estimate

**Strongest risk — diminishing returns at the hard bar.** The single assumption that most needs to hold: *the second capacity block adds nearly as much as the first.* It almost certainly does not. The first block bought +0.13pp; capacity curves are concave, so the second plausibly buys **<0.10pp** — and the bar is a hard +0.10pp at a 96.00% baseline that already matches airbench96 (96.05%). We are adding a 12th conv to a net where airbench's documented 96-config uses 10. A run-to-run schedule-noise floor of ~±0.05–0.10pp (single fixed seed, time-based schedule) could mask a true small positive or fake a marginal pass — so the **trajectory check (does the deeper net lead EXP-004 mid-training and settle higher in the tail?) must be the real evidence, not the final number alone.**

**Secondary risk — under-annealing.** A second block costs throughput. If layer3's 4×4 block is *not* as cheap as the activation-traffic argument predicts (e.g. cuDNN picks a poor kernel for 4×4/512), epochs could drop below ~110 and the low-LR tail (where all prior gains live) under-trains, landing ≤96.00%. Mitigation: the throughput smoke gates this, and the time-based schedule (lines 285–289) guarantees the anneal *completes* regardless of epoch count. ReZero init further mitigates by never disrupting the tail. Note EXP-004 ran 142 epochs with one 8×8 block; a 4×4 block should be cheaper, so ~120–130 epochs is the central expectation, comfortably above the ~110 floor.

**Honest quantitative estimate**:
- Central `best_test_acc`: **~96.05%** (+0.05pp over 96.00).
- Range: 95.98% (under-annealing / second block ~no-op) to 96.15% (capacity still meaningfully binding).
- **Probability of clearing 96.10%: ~30–40%.** This is genuinely shakier than EXP-004's ~60%-at-96.0% call, because (a) returns are concave and the first block already spent the easy capacity, (b) we are pushing past airbench96's depth, and (c) the bar is hard and noise-sized. The mechanism is sound and the change is safe (cannot regress badly thanks to ReZero), but the *magnitude* clearing a hard +0.1pp bar is the coin-flip-or-worse part.

**Effort**: **low** — one-line edit (reuses the existing `GatedResidual` class), one alpha-grad/identity/shape smoke, one throughput smoke, one 300s run on GPU 1 (`CUDA_VISIBLE_DEVICES=1`). No new deps, no new code paths.

---

## 6. Falsification criteria (for the analysis phase)

- **Confirms mechanism**: ep1/ep10 within noise of EXP-004 (identity start intact); ep25–ep50 the 12-conv net leads EXP-004; tail `best_test_acc` settles > 96.00%.
- **Refutes (capacity saturated)**: tail lands ≤96.00% despite ≥120 epochs and a live α.grad → the second block adds no usable capacity at this scale/budget; depth is no longer binding and the next loop should pivot to the orthogonal eval-side lever (multi-crop TTA, EXP-004 idea-02) rather than stacking a third block.
- **Refutes (throughput)**: epochs <110 and tail < EXP-004's trajectory at matched epochs → under-annealing, not capacity, is the cause; re-evaluate placement (fall back to a second 8×8 block) or abandon depth.
