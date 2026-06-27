# Report EXP-005: Width 5x on the doubly-regularized recipe
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-005.md
- **Plan**: plans/plan-005.md
- **Log**: logs/exp-log-005.md

## Goal

Maximize CIFAR-10 test accuracy (best_test_acc, %, higher is better) of train.py within the fixed 300s training budget (≤10 min total wall clock). Baseline at experiment start: **96.23%** @ 1174e0d (4x-wide ResNet-20 + one-cycle + TrivialAugmentWide + RandomErasing). Question tested: does a moderate capacity step (5x width, ~80 predicted epochs) beat 4x under the doubly-regularized recipe?

## Idea & Hypothesis

Chosen over 6x width and torch.compile as the largest width step predicted to stay above the epoch-starvation floor (8x failed at 40 epochs in EXP-002). Capacity was the rotated bottleneck after augmentation returns collapsed (RE +0.83pp, then TA only +0.17pp).

**Hypothesis**: WIDTH_MULT 4 → 5 (~6.7M params) raises best_test_acc from 96.23% to ≥96.4%, with ~75–85 epochs remaining sufficient for convergence.

## Approach

Single constant change: `WIDTH_MULT = 4` → `5` (stage widths 80/160/320). Recipe byte-identical to baseline 1174e0d. No deviations from plan.

## Execution

Single run, no retries. Completed cleanly (exit 0) in 362.8s total. Early signals normal: params 6,693,850 (matching prediction), epoch-1 35.79%. No errors of any kind — this is a pure research failure.

## Results

- **Primary metric**: best_test_acc = **95.12%** (baseline: 96.23, delta: **−1.11pp**, −1.15%)
- **Observations**:
  - **The epoch prediction failed badly**: 52 epochs, not 75–85. Effective throughput ~8,600 img/s vs ~18,700 at 4x — a 2.19x slowdown for only 1.56x FLOPs.
  - **Root cause of the throughput collapse (high confidence)**: 5x stage widths 80/160/320 are not multiples of 32/64. The 4x widths (64/128/256) map cleanly onto H20 tensor-core tile sizes and cuDNN kernel selection; 80/160/320 do not, so time-per-FLOP degraded ~40% on top of the FLOP increase. The prior scaling datapoint (8x = 512/256/... wait — 8x widths 128/256/512 ARE aligned, which is why its 2.85x-for-4x-FLOPs scaling looked sublinear) made alignment an invisible confounder in the prediction.
  - final = best (95.12) — the EXP-002 undertraining signature, now reproduced at 52 epochs WITH regularizers.
  - final_test_loss 0.2216 vs 0.1947 at 4x: generalization got worse, consistent with undertraining rather than over-capacity.
- **Analysis**: The hypothesis failed for a compound reason: (1) an unmodeled hardware-alignment effect halved throughput, and (2) the resulting 52 epochs sit in the starvation regime even under augmentation. The failure does NOT cleanly kill "width at current throughput" as planned, because the throughput itself was anomalously bad — a 6x net (96/192/384, all multiples of 32) could plausibly get BETTER time-per-FLOP than 5x. However, even at ideal aligned scaling (~2.25x FLOPs → ~60 epochs), the 52-epoch result here (−1.11pp) plus the 40-epoch result at 8x (−0.82pp pre-regularizers) strongly suggest anything below ~70 epochs loses more to starvation than width gains. Conclusion: the width direction now requires a throughput unlock (torch.compile) first; alignment is a hard design rule for any future width choice.
- **Key Learning**: On H20 tensor cores, channel counts must be multiples of 32/64 — 5x width (80/160/320) paid 2.19x time for 1.56x FLOPs, turning a moderate capacity step into epoch starvation.

## Verification

- **Conditions**:
  1. Clean completion within budget: total_seconds 362.8 ≤ 600, exit 0 — PASS
  2. best_test_acc ≥ baseline + 0.1pp (≥ 96.33): 95.12 — **FAIL**
  3. Eval at most once per epoch: skipped per protocol (informally compliant: 52 = 52)
- **Review Notes**: failure is genuine, not infrastructural — the run completed cleanly and the metric came from the frozen prepare.py Eval; the depressed value is fully explained by the measured epoch deficit.
- **Verdict**: no-improvement
- **Verdict Basis**: necessary condition 2 failed (metric −1.11pp below baseline)

## Unexplored Avenues

- **6x width (96/192/384 — all 32-aligned) AFTER a throughput unlock**: alignment removes this run's hidden penalty, and torch.compile could lift the epoch count above the ~70-epoch viability floor. The width idea is not exhausted; this specific (unaligned, no-throughput-headroom) approach is.
- **torch.compile at 4x**: now upgraded from "weak standalone" to "the gating enabler" — every capacity move is blocked on img/s.
- **Wider-but-shallower (NUM_BLOCKS 3 → 2 with 6x width)**: trades depth for width at roughly constant FLOPs with aligned channels; untried axis if compile proves fiddly.

## Next Steps

1. **torch.compile on the 4x baseline** (medium-high confidence as an enabler, medium as a standalone gain: even +20% img/s → ~137 epochs likely converts under the augmented recipe; watch wall-clock inflation from compile time, cap is 600s).
2. **6x width with compile in place** (medium confidence; only after (1) lands, so epochs stay ≥ ~70).
3. **Channels-last + aligned-width audit as a design rule** (no experiment needed — encode in goal-learnings; done in this loop's distillation).

## Exit Action Results
