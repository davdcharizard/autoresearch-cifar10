# Report EXP-004: Increase width to k=6
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-004.md
- **Plan**: plans/plan-004.md
- **Log**: logs/exp-log-004.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%) under a fixed 300s budget, editing only `train.py`. Higher is better.
Baseline at experiment time: **96.00%** (EXP-003). Success bar: ≥ 96.10.

## Idea & Hypothesis
Widen the network further (WIDTH_MULT 4→6, ~9.7M params), recipe + Cutout fixed. Hypothesis: capacity has been
the dominant lever and per-step time barely grew k=1→k=4 (memory-bound), so k=6 would add capacity at low
wall-clock cost (~55–70 epochs) and lift acc toward ~96.3–96.5%.

## Approach
One-line change: `WIDTH_MULT` 4→6 (stages {96,192,384}, 9,659,802 params). All else fixed. Single run.

## Results
- **Primary metric**: **95.26%** (baseline 96.00%, delta **−0.74 pp** — regression)
- **Observations**: Only **35 epochs / 13,314 steps** fit (vs k=4's 77 / 29,931). Per-step time jumped to
  ~22 ms (vs k=4 ~10 ms) — **the k=4→k=6 step crossed from memory-bound into compute-bound**, so throughput did
  NOT stay flat as the k=1→k=4 data had suggested. final_test_loss 0.223 (worse than EXP-003's 0.204). The run
  was clean and in budget; it simply underfit.
- **Analysis**: Hypothesis refuted in its central assumption. The "width is nearly free" pattern held only while
  the model was memory/launch-bound (k≤4); at k=6 the H20 becomes compute-bound and the epoch budget collapses
  (77→35). 35 epochs of a 9.7M model generalizes worse than 77 epochs of a 4.3M model under this 300s budget —
  the capacity/epoch trade-off has turned. This locates the width sweet spot at **k=4** for the current recipe
  and budget. Not a capacity-is-useless result — rather, *more* capacity needs either more epochs (not available)
  or a faster bigger model. Trajectory unchanged: best stays 96.00% (EXP-003).
- **Key Learning**: Width stops being "free" past k=4 — k=6 turns compute-bound (~22ms/step), fits only 35
  epochs, and underfits (95.26 < 96.00). k=4 is the capacity sweet spot at the 300s budget; further pure widening regresses.

## Verification
- **Conditions**: Condition 1 (clean completion) PASS; Condition 2 (≥96.10) **FAIL** (95.26); Condition 3 skipped.
- **Review Notes**: Result trustworthy — clean fixed-seed run, eval frozen, metric genuinely below baseline.
  Not invalid (no constraint breach) and not a crash (ran to completion) — a legitimate no-improvement.
- **Verdict**: no-improvement
- **Verdict Basis**: valid run but primary metric fell below baseline (verification condition 2 failed).

## Unexplored Avenues
- **Regularization at k=4 (current best)**: WD 1e-4→5e-4, tune Cutout size, or mixup — push the *k=4* model
  rather than adding capacity it can't train. Most promising direction now.
- **A "wider but cheaper" capacity add**: e.g. widen only the last stage, or a moderate k=5, to gain some
  capacity without halving epochs — but EXP-004 suggests diminishing returns; lower priority.
- **Reduce per-epoch cost to afford a bigger model**: larger batch + LR scaling (more GPU utilization/parallelism)
  could raise steps/s so a wider model fits more epochs — speculative.
- **Depth instead of width at k=4 (e.g. NUM_BLOCKS 4)**: small capacity add that may be cheaper than k=6.

## Next Steps
1. **Tune regularization on the k=4 model (WD 5e-4, and/or Cutout size, and/or mixup)** — *medium-high
   confidence*; k=4 is the sweet spot, so squeeze it rather than widen. (Revert to k=4 first — done in housekeeping.)
2. **Recipe tuning on k=4 (peak LR sweep)** — *medium confidence*; cheap, the LR was set in the k=1 era.
3. **Larger batch + LR scaling** to raise throughput, possibly re-enabling a bigger model — *low-medium confidence*.

## Exit Action Results
- None defined for this goal — skipped.
