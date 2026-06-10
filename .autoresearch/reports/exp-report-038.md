# Experiment Report EXP-038

**Date**: 2026-06-09
**Verdict**: no-improvement
**Metric**: best_test_acc = 95.47% (baseline 96.22%, bar 96.32%; Δ −0.75pp)

## Goal
Maximize CIFAR-10 ResNet-20 `best_test_acc` (%) within a fixed 300s training-compute budget on a
single H20, editing only `train.py`. Higher is better. Baseline 96.22% (EXP-012, commit 6c417a4);
bar = baseline + 0.1 = 96.32%.

## Idea & Hypothesis
**Chosen idea**: compute-neutral **fat-head width reallocation** — change per-stage widths from the
uniform `{64,128,256}` (k=4) to a fat-head schedule that narrows the spatially-expensive stage1
(32×32) and widens the spatially-cheap discriminative stage3 (8×8) at ≈constant total FLOPs. Motivated
by the project's two HIGH walls: top-1 needs CAPACITY (not polish), but ANY FLOP add trips the epoch
wall. Because a stage3 channel costs ~16× fewer FLOPs than a stage1 channel, reallocation was
hypothesized to add net capacity (more channels/params) where it's cheap WITHOUT adding FLOPs/epochs —
the only route thought to thread both walls.

**Hypothesis**: reallocating capacity to stage3 at constant FLOPs lifts best_test_acc above 96.32
without the epoch-wall under-training of uniform widening, at throughput-neutral ~91 ep.

## Approach
Single edit to `train.py` L100-101: replaced `w1,w2,w3 = 16k,32k,64k` ({64,128,256}) with explicit
widths. Two width values were used across runs:
- Run 1: `{44,128,320}` — discarded (see Execution): w1=44 is not a multiple of 8 → stage1 convs fall
  off the tensor-core/channels_last fast path; plus the GPU was externally contended.
- Run 2 (accepted): `{48,128,304}` — tensor-core-aligned (multiples of 16), FLOP-matched (≈−0.9% by
  the w²·area proxy), +32 net channels concentrated in stage3. params 5,468,154 (vs baseline
  4,299,866). The `_make_layer`/projection shortcuts and `fc=Linear(w3,10)` + adaptive avg-pool adapt
  to the new widths with no other change.

## Execution
Heavily complicated by **shared-node GPU contention**: both H20s were under volatile external load
(another user's Protenix `batch_inference.py` cycling jobs onto GPU 0 and 1). The benchmark is
wall-clock-dt-gated, so contention inflates per-step dt and collapses the epoch count → confounded
results. Sequence:
- Run 1 {44,128,320}: dt ~36–38ms (contention + 44-channel misalignment) → killed.
- Run 2 {48,128,304} manual attempts: dt ~24ms (mid-run contention) → confounded (52 ep → 94.98).
- Built an automated fair-run launcher: poll for an idle GPU, launch, early-abort if contended by 90s,
  accept only a completed run reaching ≥64 ep. It caught a fully-clean window on GPU 0 — **dt uniform
  10–11ms (446×10ms + 114×11ms, no spikes), 73 epochs, accepted.**

The accepted clean run: best_test_acc 95.47%, final_test_acc 95.40%, final_test_loss 0.2082,
num_epochs 73, num_steps 28087, total_seconds 390.4, peak_vram_mb 425.0.

## Results
The clean, uncontended run gives a fair verdict: **95.47% (−0.75pp), under-trained at 73 epochs.**

The decisive observation is the **clean dt: ~10.5ms vs baseline's 8ms (+31%) despite ≈FLOP-neutral
widths.** The fat-head reallocation is FLOP-neutral but **NOT wall-clock-neutral** — the wider 304-ch
stage3 is memory-bandwidth-bound (more activation/weight traffic at 8×8) and/or fuses less efficiently
under torch.compile, so each step costs ~31% more wall-clock than the FLOP count predicts. Under the
fixed 300s budget that yields only 73 epochs (vs ~91), a ~20% epoch handicap → genuine under-training
(final_test_loss 0.208 > baseline 0.195) → regression. The added stage3 capacity did NOT overcome the
epoch loss.

This **falsifies the central premise** that capacity could be reallocated "for free" by exploiting the
per-stage FLOP asymmetry: the asymmetry is real in FLOPs but the wall-clock budget does not track FLOPs
(it tracks memory-bound execution time). It is a direct instance of the project-insights EXP-015 lesson
("FLOPs-neutral architecture changes are NOT necessarily wall-clock-neutral under torch.compile") and
reinforces the HIGH compute-wall entry: even FLOP-neutral capacity placement trips the epoch wall via a
wall-clock premium. Combined with uniform widening (EXP-004/009), the capacity axis is now closed from
both the uniform and the non-uniform/reallocation directions.

A secondary, durable lesson: this shared H20 node is intermittently saturated by another user's jobs;
the wall-clock-dt-gated budget makes ANY run launched during contention invalid, so a fair run requires
explicit idle-GPU gating + epoch-count sanity (the launcher pattern).

## Verification
- **Cond 1 — primary metric clears bar (`>=96.32`)**: FAILED — 95.47%.
- **Cond 2 — clean completion within budget**: PASSED — total_seconds 390.4 < 600, exit 0.
- **Cond 3 — no hard-constraint violations**: PASSED — diff = train.py only; eval-line count 73 ==
  num_epochs 73 (≤1 eval/epoch); no new deps; seed 42; prepare.py/eval untouched. num_params changed
  (5,468,154) — expected, capacity is not constrained.

Results trustworthy: the accepted run had uniform clean dt (no contention spikes), the 73-ep count
matches the clean ~10.5ms dt exactly, loss/acc are on a plausible (under-trained) trajectory. Verdict:
**no-improvement** (condition 1 failed on its merits; the result is a fair test of the fat-head at its
real wall-clock cost).

## Unexplored Avenues
- **Gentler reallocation** (e.g. {56,128,288}): smaller stage3 widening → smaller wall-clock premium →
  more epochs, but also less added capacity. Likely still net-negative (the premium-vs-capacity
  trade-off is unfavorable on this memory-bound path); LOW value — the reallocation axis is effectively
  closed by this result's mechanism.
- **Depth reallocation instead of width** (more blocks in stage3, fewer in stage1 at matched FLOPs):
  deepening is even more launch/wall-clock-costly than widening on this launch-bound net → worse epoch
  wall. Not promising.
- The capacity bound appears GLOBAL and inseparable from the wall-clock/epoch wall at this budget —
  there is no free capacity to add. Future capacity attempts are very unlikely to pay off.

## Next Steps
1. **Abandon capacity/architecture levers entirely** (high confidence) — both uniform (EXP-004/009) and
   FLOP-neutral-reallocated (EXP-038) capacity adds trip the wall-clock epoch wall; the k=4 {64,128,256}
   allocation is at the compute-optimal frontier. Stop probing capacity.
2. **Re-examine whether the net is truly capacity-bound vs data/label-bound** (medium confidence) — every
   compute-neutral polish lever moved loss not top-1, and every capacity/regularizer add regressed. The
   remaining unprobed direction is the TRAINING TARGET/DATA the net sees within budget that is neither
   "more augmentation" (closed) nor "an optimizer tweak" (closed) — e.g. a curriculum/sample-weighting
   that is strictly compute-neutral and does not add FLOPs. Low-confidence but one of few unclosed angles.
3. **Accept the plateau may be at/near the true ceiling for ResNet-20-k4 @ 300s** (high confidence) — after
   39 experiments the 96.22 plateau is bounded by capacity (closed both ways), the compute wall, and the
   polish-vs-top1 wall. Document candidly; continue probing only genuinely-novel compute-neutral,
   non-polish, non-capacity levers.

## Key Learning
A FLOP-neutral fat-head width reallocation ({64,128,256}→{48,128,304}) is NOT wall-clock-neutral: the
wider memory-bound stage3 raised dt 8→10.5ms (+31%) → 73 ep → under-trained to 95.47% (−0.75pp). Capacity
cannot be added "for free" via the per-stage FLOP asymmetry; the capacity axis is now closed from both the
uniform and reallocation directions. (Also: this shared H20 node requires explicit idle-GPU gating for a
fair dt-budgeted run.)
