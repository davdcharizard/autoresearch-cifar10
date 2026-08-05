# Proposal — EXP-013 idea-01: Mild capacity step, widen layer2 / 8×8 stage 256→320

## 1. Summary (concrete train.py change)

Widen the layer2 (8×8) stage from 256 to 320 channels — a ~1.25× width step, deliberately
milder than EXP-007's 256→384 (1.5×). This touches exactly three sites in `ResNet9.__init__`
(`train.py`, lines 149–151), because layer2's output width feeds layer3's input stem:

- `layer2 = nn.Sequential(conv_bn(128, 256), MaxPool2d(2), GatedResidual(256))`
  → `conv_bn(128, 320)` and `GatedResidual(320)` (the inner ReZero block's two `conv_bn(c,c)`
    become `conv_bn(320,320)`; `alpha` stays a scalar, init 0 — identity-at-init preserved).
- `layer3 = nn.Sequential(conv_bn(256, 512), MaxPool2d(2), Residual(512))`
  → `conv_bn(320, 512)` for the stem; `Residual(512)` is unchanged (consumes layer3[0]'s 512 out).

The `fc(512,10)`, whitening conv, `prep`, and `layer1` are all untouched. No hyperparameter
changes (PEAK_LR 0.4, wd 5e-4, LS 0.2, EMA 0.998, schedule, batch 512 all held) — this is a clean
single-variable capacity test, exactly as EXP-007 was, so the result is directly comparable to it.

Mechanically this is a ~3-line edit to integer literals plus a same-session baseline cell.

## 2. What it targets (the named limiter)

The limiter is the **capacity-vs-epoch tradeoff at the 300s training-time budget**, instantiated at
the layer2/8×8 stage. EXP-004 established the 8×8 stage is the *productive* capacity location:
adding a ReZero block there won +0.13pp (95.87→96.00), the last capacity win, while the same
construction at layer3/4×4 (EXP-005) and a coarse deepen both lost. The open question EXP-007 left
is purely one of *magnitude*: width at 8×8 is not discredited, only the 1.5× step that under-annealed.

This experiment targets the hypothesis that the net is *mildly* capacity-bound at 8×8 and that a
smaller width step buys realizable capacity without truncating the low-LR tail past the under-anneal
cliff. The metric path: more 8×8 channels → richer mid-level features at the stage where capacity
already paid off (EXP-004) → higher annealed-tail accuracy, *provided* the smaller per-step FLOP cost
keeps `num_epochs` in the clean/mild zone (≥135) rather than EXP-007's truncated 94.

## 3. Reasoning — why 320 should land differently from 384 (quantitative)

**Parameter delta (verified against EXP-007's reported +2.21M).**
The widen affects four conv/BN weight tensors. With conv weight = c_out·c_in·9:

| site | 256→384 (EXP-007) | 256→320 (this) |
|---|---|---|
| layer2[0] conv(128,·) | 128·128·9 = 147,456 | 128·64·9 = 73,728 |
| GatedResidual c1 (c²·9) | (384²−256²)·9 = 737,280 | (320²−256²)·9 = 331,776 |
| GatedResidual c2 | 737,280 | 331,776 |
| layer3[0] stem conv(·,512) | 128·512·9 = 589,824 | 64·512·9 = 294,912 |
| **total (≈, +BN)** | **≈ +2.21M** ✓ | **≈ +1.03M** |

So 256→320 costs **+1.03M params (~47% of EXP-007's +2.21M)**, taking the net from 7,784,627 to
~8.82M (+13%). The dominant FLOP term is the two 8×8 GatedResidual convs, which scale with c²: their
combined cost grows 1.56× at 384 (384²/256²) but only **1.56→1.25²=1.56... → actually (320/256)²=1.5625**
— correction: (320/256)² = 1.5625 for those two convs, vs (384/256)²=2.25 at 384. The layer2[0]
(16×16, c_in fixed at 128) and layer3[0] stem (8×8→ pooled, c_out fixed at 512) scale *linearly*
in the added width, so their added FLOPs are 64/128 = 50% of EXP-007's added FLOPs at those sites.

**Epoch prediction.** EXP-007's 256→384 cut epochs 150→94 (−56 epochs, −37%). That cost is driven
by added per-step FLOPs, which at the two c²-dominated 8×8 convs scale as Δ(c²): 384 added
384²−256²=81,920 units/conv; 320 adds 320²−256²=36,864 units/conv = **45% of the 384 increment** at the
heaviest sites; the linear sites add 50%. Taking ~45–50% of EXP-007's epoch cost gives an added-FLOP
cost of roughly −25 to −28 epochs from the 150-epoch base, i.e. a **predicted ~122–128 epochs** for a
clean (no extra shared-host contention) run. That lands *below* the ≥135 "mild" threshold and *above*
the <110 "abort" threshold — i.e. squarely in the ambiguous middle. This is the honest read: 320 is
*better-positioned* than 384 but is **not guaranteed** to clear the clean-anneal bar. The throughput
itself (img/s) should stay near the full ~26k (8×8 convs run at full speed, no 4×4 cuDNN penalty like
EXP-005), so the epoch loss is FLOP-driven and predictable, not kernel-pathological.

For the win to materialize, the realizable capacity gain at ~122–128 epochs must exceed the tail
accuracy lost to ~25 fewer anneal epochs. EXP-004 is the existence proof that 8×8 capacity *can* win
even at reduced epochs (it won at 142 vs EXP-003's 174). The bet is that 320's smaller epoch hit
(~125) stays on the right side of that same curve where 384's 94 did not.

## 4. Sources (cited evidence)

- **EXP-007** (`experiments/007/04-analysis.md`; TSV row 007, 95.85, no-improvement): 256→384 cut
  epochs 150→94, best==final (monotone rise to ep94, still climbing) → under-anneal, NOT
  capacity-saturation. Explicitly pre-registers "try a milder 256→320." This is the direct parent.
- **EXP-004** (`experiments/004/04-analysis.md`; TSV row 004, 96.00, improvement): ReZero block at
  layer2/8×8 → +0.13pp; capacity lead emerges by ep25 (92.63 vs 88.84). Establishes 8×8 as the
  productive capacity location and that capacity can win despite an epoch cost.
- **Under-anneal failure entry** (`03-experiment-learnings.md`, Medium-Importance, count 2,
  EXP-005+EXP-007): capacity adds that cost too many epochs under-anneal because the low-LR tail is
  truncated; prescribes SMALL steps and reading `num_epochs` first.
- **EXP-005** (TSV row 005): 4×4 capacity both unused AND ~10% slower (cuDNN at small spatial). Source
  of the expectation that 8×8 widening avoids the kernel-speed penalty — the epoch cost here is pure
  FLOPs, hence predictable.
- **Noise-floor entry** (`03-experiment-learnings.md`, High-Importance, EXP-006/004/005): ~0.1pp
  run-to-run jitter from time-budgeted epoch-count variation at fixed seed → mandates a same-session
  baseline cell for attribution.

## 5. Estimated effort

**Low–medium.** The code change is ~3 integer literals (256→320, twice, plus the layer3 stem
in-channels 256→320). One full-throughput run (~300s training + eval, ~7–8 min wall) plus one
same-session baseline cell for the noise floor. No new deps, no schedule retune, no new modules.

## 6. Risk assessment

**Dominant risk — under-anneal (recurring failure, count 2).** The predicted ~122–128 epochs is in
the ambiguous band between the ≥135 "mild/clean" threshold and the <110 "abort" threshold. If
shared-host contention stacks on top of the FLOP cost (as it did in EXP-007, where contention helped
push 150→94), epochs could fall toward ~110 and the net would finish still-climbing — a repeat of
EXP-007's best==final signature, just milder. This is the single assumption that most needs to hold:
that the FLOP-driven epoch cost is ~45–50% of EXP-007's and that the host is not heavily loaded.

**Secondary risk — capacity simply isn't binding at 8×8 anymore.** EXP-012 found the ReZero gate is
"not accuracy-limiting at any magnitude" (decoupling wd halved |α| with zero accuracy cost),
suggesting the existing 8×8 capacity may already be sufficient. If so, even a cleanly-annealed 320
run ties baseline (added width unused, like EXP-005's 4×4) rather than winning. This makes the upside
genuinely uncertain — honestly, this idea is *shakier than the "last capacity win was here" framing
suggests*, because EXP-012's α-magnitude evidence points at a partially-saturated 8×8 stage. The width
axis (more channels) is mechanistically distinct from the depth/gate axis EXP-012 probed, so it is not
strictly refuted — but the prior should be tempered.

**Pre-registered decision gate (read num_epochs FIRST, per the learnings):**
1. Inspect `num_epochs` from the run summary BEFORE judging accuracy.
   - `num_epochs < 110` → **ABORT / under-anneal verdict** (matches EXP-007 cliff); do not credit any
     accuracy number, report as under-anneal, the width premise remains untested at full anneal.
   - `110 ≤ num_epochs < 135` → **MILD zone**: check the tail trajectory. If `best == final` (monotone
     to the last epoch), declare under-anneal even if accuracy is near baseline — the capacity is
     unrealized. Only count a win if best occurs strictly before the final epoch AND clears the bar.
   - `num_epochs ≥ 135` → **CLEAN zone**: accuracy is the verdict.
2. **Win bar:** best_test_acc ≥ 96.48 (baseline 96.38 + 0.10pp) AND ≥ same-session baseline cell + 0.10pp
   (to clear the ~0.1pp epoch-jitter noise floor; the stored 96.38 baseline alone is insufficient).
3. If the run lands in the MILD zone with best==final and accuracy ≈ baseline, the honest verdict is
   "inconclusive on capacity, lost on anneal" — the same EXP-007 outcome at lower severity — and the
   width axis should then be considered closed (two under-anneal datapoints would promote the failure
   to High importance and a stronger do-not-retry signal).

## 7. Concrete train.py code sketch

In `ResNet9.__init__` (`train.py`, currently lines 149–151), change the layer2 width and the layer3
input stem:

```python
# BEFORE (current, lines 149-151):
self.layer1 = nn.Sequential(conv_bn(64, 128), nn.MaxPool2d(2), Residual(128))
self.layer2 = nn.Sequential(conv_bn(128, 256), nn.MaxPool2d(2), GatedResidual(256))
self.layer3 = nn.Sequential(conv_bn(256, 512), nn.MaxPool2d(2), Residual(512))

# AFTER (widen layer2 8x8 stage 256 -> 320; layer3 stem in-channels follow):
self.layer1 = nn.Sequential(conv_bn(64, 128), nn.MaxPool2d(2), Residual(128))
self.layer2 = nn.Sequential(conv_bn(128, 320), nn.MaxPool2d(2), GatedResidual(320))
self.layer3 = nn.Sequential(conv_bn(320, 512), nn.MaxPool2d(2), Residual(512))
```

No other edits. `GatedResidual(320)` automatically builds `conv_bn(320,320)` ×2 with `alpha`
init 0 (identity at init, ReZero invariant preserved). `conv_bn(320, 512)` keeps c_out=512 so
`fc(512,10)`, `pool`, and the whitening front-end are untouched. Expected reported
`num_params` ≈ 8.82M (vs current 7,784,627). Confirm `num_params` and `num_epochs` in the run
summary, then apply the decision gate in §6.
