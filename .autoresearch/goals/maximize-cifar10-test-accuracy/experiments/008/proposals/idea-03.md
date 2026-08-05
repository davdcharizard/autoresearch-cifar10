# Proposal EXP-008 idea-03: Milder widen layer2 (8×8 stage) 256→320

## One-line summary
Widen the proven 8×8 middle stage `layer2` 256→320 (the pre-registered EXP-007 fallback): ~half the capacity of the 384 widen at ~half the throughput cost, aiming to land on the right side of the capacity-vs-epochs tradeoff (target ~120–135 epochs) and clear 96.10%.

## Mechanism — how this advances best_test_acc, tied to the named limiter

The diagnosis from EXP-007 named the binding constraint precisely: **the capacity/epoch balance, not capacity itself, is the limiter**. EXP-004 proved capacity is still a live lever at 96.0% (adding one ReZero block at 8×8 bought +0.13pp). EXP-007 tried to push that lever harder (256→384, +2.21M params, 1.5× layer2 cost) and *under-annealed* — it fit only 94 epochs vs EXP-004's 142, and the net was monotonically still climbing at budget exhaustion (ep90 95.67 → ep94 95.85, best==final). The capacity premise was not refuted; the net simply never reached the flat, fully-annealed low-LR tail where the ~96.0 floor is set.

The causal chain for 256→320:
1. A 320 widen adds **+1,032,576 params** (hand-computed below) — 47% of the 384 widen's +2,212,608, i.e. roughly half the extra capacity.
2. The throughput cost scales with the *added FLOPs*, which are dominated by the two 8×8 GatedResidual convs (∝ channels²). At 320 those convs are (320/256)² = 1.5625× the 256 cost; at 384 they were (384/256)² = 2.25×. So the *incremental* compute over baseline at 320 is roughly (1.5625−1)/(2.25−1) = 0.5625× ≈ 56% of the 384 widen's incremental compute. The 16×16 `layer2[0]` stem and 8×8 `layer3[0]` stem add to this but are smaller terms.
3. EXP-007 lost 142→94 = 48 epochs to the 384 widen. If the epoch loss scales roughly with incremental compute (≈56%), 320 should lose ~27 epochs → land near **115–122 epochs**, possibly higher (120–135) on a less-contended host. That is plausibly enough epochs to reach the annealed tail, unlike 94.
4. With enough epochs to anneal AND extra capacity, the annealed loss floor should settle at or below baseline — the EXP-004 mechanism (capacity lifts the floor) applies, but now at a capacity step chosen to *not* starve annealing.

This is the single most evidence-backed next step: it is the explicit pre-registered fallback recorded in three places (EXP-007 analysis §Key Learning, §Unexplored Avenues #1, §Next Steps #1; EXP-007 plan §Abort Criteria).

## Concrete change — exact edits in this codebase

All edits are in `ResNet9.__init__`, `train.py` lines 150–151 (the only lines touched), mirroring the EXP-007 diff but with 384→320:

- **Line 150** `self.layer2`:
  `nn.Sequential(conv_bn(128, 256), nn.MaxPool2d(2), GatedResidual(256))`
  → `nn.Sequential(conv_bn(128, 320), nn.MaxPool2d(2), GatedResidual(320))`
- **Line 151** `self.layer3` stem input only:
  `nn.Sequential(conv_bn(256, 512), nn.MaxPool2d(2), Residual(512))`
  → `nn.Sequential(conv_bn(320, 512), nn.MaxPool2d(2), Residual(512))`

`layer3` OUTPUT stays 512, so `self.pool = nn.MaxPool2d(4)` and `self.fc = nn.Linear(512, 10)` are untouched (line 152–153). `whiten`/`prep`/`layer1` untouched. No HP/schedule/optimizer/augmentation/loop/`forward` change. `GatedResidual(320)` reuses the proven class unchanged — `self.alpha = nn.Parameter(torch.zeros(1))` keeps the branch exact-identity at init (the `layer2[2]` index is unchanged), so the deeper-width net starts bit-equivalent on that branch and ramps capacity in via α, with the live gradient path `∂L/∂α ≠ 0` intact.

### Exact parameter count (hand-computed, method validated against the known 384 = 9,997,235)

Baseline total `num_params` (incl. frozen whiten conv) = **7,784,627** (EXP-004; learnable 7,783,169 + 1,458 whiten = 54·3·3·3).

Delta for 256→320 (Δc = 64):
- **layer2[0]** conv_bn(128, 256→320): conv 64·128·9 = 73,728; BN 64·2 = 128 → **73,856**
- **layer2[2]** GatedResidual two conv_bn(c,c): each conv 9·(320²−256²)=9·36,864=331,776; BN 128 → 331,904 per block; ×2 = **663,808** (α scalar unchanged)
- **layer3[0]** conv_bn(256→320, 512): conv 64·512·9 = 294,912; BN(out=512) unchanged → **294,912**
- **Total Δ = 1,032,576**

**New num_params = 7,784,627 + 1,032,576 = 8,817,203.** (Cross-check: the same method gives the 384 widen's published 2,212,608 / 9,997,235 exactly, so this figure is trustworthy. Verify in the Milestone-1 smoke as `num_params == 8,817,203`.) This is +1.03M over baseline vs EXP-007's +2.21M — sitting roughly halfway, exactly the "~half the capacity" the diagnosis called for.

## Throughput / epoch projection

- EXP-004 (256): 142 epochs / 13,704 steps, steady ~26.3k img/s.
- EXP-007 (384): 94 epochs / 9,069 steps, img/s oscillating 6–8k dips to ~20k peaks (host contention + intrinsic cost), still climbing at the end.
- 320 incremental compute ≈ 56% of the 384 widen's (channel² scaling of the dominant 8×8 convs, §Mechanism point 2) → projected **~115–122 epochs** under EXP-007's contention, **~125–135** if the host is quieter. The target ~120–135 from the diagnosis is realistic but not guaranteed.

**Host-contention caveat (load-bearing):** GPU 0 is shared (hard constraint: runs go on GPU 1, but the host/PCIe/CPU dataloader workers contend). EXP-007 explicitly attributed part of its 94-epoch shortfall to contention-driven img/s dips, and EXP-007 plan §Unexplored-Avenues notes epoch count is "partly host-load-dependent." This means num_epochs is a *noisy* diagnostic: a 320 run that lands at ~112 could be either intrinsic cost or a transiently busy host. The 320 step is deliberately more robust to this than 384 because its throughput hit is smaller, but a single run cannot fully disentangle contention from intrinsic cost — weigh num_epochs together with the *trajectory shape* (still-climbing vs flattened), not in isolation.

## PEAK_LR — retune or hold?

**Hold PEAK_LR = 0.4** (as EXP-007 did), for single-variable attribution. The honest caveat (from EXP-007 plan §Code-Changes): the GatedResidual(320) branch is identity-init (α=0, inert at start, no LR sensitivity), but the **widened main-path convs `layer2[0]` (128→320) and `layer3[0]` (320→512) are NOT identity-preserving** — they are kaiming-initialized at the new width and active from step 1. A wider main path can shift the optimal LR (more channels → different gradient-noise/curvature scale). So holding 0.4 is a deliberate attribution choice, not a guarantee the optimum is unchanged. It admits a third failure mode (LR/optimization mismatch) alongside under-anneal and saturation. I recommend holding for this run to keep it a clean capacity A/B and to preserve the direct comparability to EXP-004/EXP-007; if 320 lands with adequate epochs (≥125) but flat accuracy, an LR retune (e.g. 0.3–0.5 sweep) becomes the indicated follow-up rather than a confound to fix mid-experiment. Note EXP-004's ReZero result showed the recipe is fairly LR-robust at this scale; the width change here is modest (1.25× channels), so a large LR mismatch is unlikely but cannot be excluded.

## Evidence

- **EXP-004 (in-repo, the strongest evidence):** capacity is a *binding* lever at 96.0% — one ReZero block at the 8×8 stage = +0.13pp (95.87→96.00), and the capacity gain outran a 32-epoch throughput loss (142 vs 174). This is the direct demonstration that *adding* 8×8 capacity helps when epochs remain sufficient.
- **EXP-007 (in-repo, defines the operating point):** 256→384 under-annealed (94 epochs, still climbing, −0.15pp). Crucially it showed the failure was *epochs, not capacity* (monotone-rising tail, best==final). This is the precise reason a milder step is the indicated move, and it bounds the cost: 384 lost 48 epochs, so a 56%-incremental-compute step should lose proportionally fewer.
- **airbench96 (arXiv:2404.00498, cited across EXP-004/007 brainstorms):** the documented 95→96 step is a bigger net with a ~384 middle width; 320 moves toward that width while staying inside the epoch budget. airbench reaching 96 in 37 epochs is also the basis of the saturation counter-argument (below).
- **Param accounting (code fact):** my hand-computed delta reproduces EXP-007's published 9,997,235 exactly, so the 8,817,203 figure and the "half the capacity" framing are reliable, not estimated.

## Honest headroom assessment vs the saturation counter-argument

This is the crux and I will not inflate it. There are three live hypotheses for a 320 run:

1. **Win (capacity realized):** epochs recover to ~120–135, the wider net anneals fully, floor lifts to ≥96.10. Supported by EXP-004's direct +0.13pp capacity result and EXP-007's "still climbing" evidence that the 384 net *would* have done better with more epochs.
2. **Under-anneal again:** epochs ≤110, still-climbing tail, sub-96.10. Possible if 320's throughput hit is larger than the channel²-scaling estimate (the 16×16 stem and layer3 stem add compute) or the host is busy.
3. **Saturation:** epochs ≥125 but accuracy flat ≤~96.0. This is the counter-argument's prediction: EXP-004's 142 epochs gave the same ~96.0 as ~150; airbench hits 96 in 37 epochs — the recipe may be at a ~96.0 ceiling that *more capacity at fixed recipe* cannot break.

**My honest read:** this is a **marginal-to-moderate** bet, not a high-confidence one. The saturation counter-argument has real force — if 142 epochs already saturates the *recipe* (not just this net's capacity), then a 320 net that fits ~125 epochs might net the same ~96.0 ± noise, landing in the sub-0.1pp dead zone exactly as 384 did (but for the *opposite* reason). The counter-evidence is that EXP-004 was NOT saturated by capacity at 142 epochs (adding capacity moved it +0.13pp), and EXP-007's 384 tail was still rising — both suggest the floor is set by *capacity-given-enough-epochs*, and 320 is a deliberate attempt to get both. But the expected effect size is small: half of EXP-004's +0.13pp capacity contribution is ~+0.06pp, which is *below* the +0.1pp bar and inside the ~0.1pp noise floor. To clear the bar, 320 needs the capacity gain to be *more* efficient per-param than a linear interpolation between EXP-004 (256+1 block) and EXP-007 (384), OR to benefit from being a width step (which EXP-007 showed the architecture wants) rather than another depth block. I would call this **<50% to clearly clear 96.10**, with a meaningful chance of a sub-noise null. It is still the right experiment to run because (a) it is the pre-registered, lowest-regret way to resolve the EXP-007 ambiguity (was it capacity or epochs?), and (b) the num_epochs diagnostic makes even a null *informative* — it cleanly partitions the remaining hypothesis space and tells the loop whether to abandon capacity-scaling entirely (saturation) or push a cheaper capacity placement.

## Verification / falsification — num_epochs as first-class diagnostic

Primary metric: `best_test_acc ≥ 96.10` AND clearly above the ~0.1pp noise floor → improvement.

Diagnostic partition (record `num_epochs`, `num_steps`, `img/s`, and the per-epoch tail trajectory):
- **best ≥ 96.10:** improvement — capacity realized at the milder step (the hypothesis).
- **best < 96.10 AND num_epochs ≤ 110:** **under-annealed again** (same failure as 384, milder). Read the tail: if still monotonically climbing (best==final), capacity is *still* not refuted — pivot to a *cheaper* capacity add (e.g. 256→288, or widen only the GatedResidual branch leaving layer2[0] stem at 256, or the 2nd-ReZero-block depth option at ~1.18M). Caveat: attribute partly to host contention — check img/s dips.
- **best < 96.10 AND num_epochs ≥ 125 with a FLAT tail (not climbing):** **capacity-saturated at fixed recipe** OR LR mismatch. This is the decisive saturation signal — it would refute the "more 8×8 capacity helps" thesis at this recipe and redirect the loop to orthogonal levers (TTA composition, schedule/LR retune, augmentation) rather than more capacity. Distinguish LR-mismatch from saturation via a follow-up small LR sweep only if the tail is flat with adequate epochs.

Genuineness/scope (per EXP-007 protocol): diff confined to lines 150–151; `prepare.py` byte-unchanged; `num_params == 8,817,203`; PEAK_LR still 0.4; one `evaluator.evaluate(` site (≤1 eval/epoch); seeds `torch.manual_seed(42)`/`torch.cuda.manual_seed(42)` unchanged; summary `best_test_acc` == max per-epoch `best:`.

Note the unavoidable noise-floor confound (same as every arch change here): the width change shifts the CPU RNG stream consumed by `self.apply` (kaiming) and the DataLoader shuffle/worker seeds, so same-shaped downstream modules (`layer3` Residual(512), `fc`) and the data stream differ slightly — part of the ~0.1pp floor. Only a clearly-greater-than-noise gain registers; a +0.06pp-class result is not cleanly attributable.

## Strongest risk

**The assumption that most needs to hold:** that ~120–135 epochs is *both* achievable (throughput) *and* sufficient (the recipe is not already saturated at 96.0). The single most likely failure is that this bet falls into the sub-noise dead zone — either under-annealing recurs because the throughput hit (compounded by shared-host contention) is larger than the channel²-scaling estimate, OR epochs are adequate but the marginal capacity at fixed recipe nets ≈+0.06pp, inside the noise floor. In both cases best < 96.10 and the run reads as no-improvement. The mitigant is that 320 is specifically the smallest-regret step to *resolve* that ambiguity, and num_epochs makes the outcome diagnostic regardless of sign.

## Effort

**Low.** A two-token edit on two lines (384→320 relative to the already-specified EXP-007 diff, or 256→320 relative to baseline), no new code paths, reuses the existing GatedResidual class and the EXP-007 verification protocol verbatim with one constant changed (num_params 9,997,235 → 8,817,203). One training run within the 300s budget, ~430–490s wall. Smoke check is a CPU forward + param-count assert.
