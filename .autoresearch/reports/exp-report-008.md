# Report EXP-008: Wider-but-shallower — ResNet-14 at 6x width
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-008.md
- **Plan**: plans/plan-008.md
- **Log**: logs/exp-log-008.md

## Goal

Maximize `best_test_acc` (%) of the CIFAR-10 ResNet within the fixed 300s training budget (higher is better). Baseline at experiment time: **96.71%** @ 1990397 (4x ResNet-20, compiled TA+RE recipe, 139 epochs). Specific question: after uniform widening was closed by three epoch-starvation failures, can capacity still pay if depth FLOPs fund the width — ResNet-14 (NUM_BLOCKS 2) at 6x width, keeping projected epochs above the ~70 floor?

## Idea & Hypothesis

Chosen idea: trade depth for width at roughly constant time — NUM_BLOCKS 3→2 and WIDTH_MULT 4→6 (96/192/384, ~6.15M params = 1.43x baseline capacity at ~1.46x FLOPs). The brainstorm rated this the only capacity move compatible with the High count-3 epoch-floor constraint, with WRN (arXiv 1605.07146; WRN-16-8 within 0.3pp of WRN-28-10) as direct external evidence that shallow-wide matches deep-thin on CIFAR-10. Hypothesis: best_test_acc ≥ 96.85, predicated on ~80–85 epochs from measured-regime dt (36–40ms) clearing the starvation floor. Runner-up candidates: Mixup at 139 epochs, EMA eval.

## Approach

Two constant changes in train.py: `NUM_BLOCKS = 3` → `2` (ResNet-14) and `WIDTH_MULT = 4` → `6`; comments updated; recipe otherwise byte-identical to baseline 1990397. The plan carried a Failed-Approach Retry Justification (depth pays for width; projection from measured compiled scaling per the EXP-007 learning) and — new this loop — a **binding dt gate**: kill at step ~100 if dt ≥ 47ms (projecting < 65 epochs). No deviations from plan.

## Execution

One run, no retries (task bglu7etqy, GPU 0). All early signals green: params 6,149,866 (≈ prediction), epoch-1 eval 39.58% (strongest epoch-1 of any experiment — shallow-wide optimizes fast early), and the dt gate PASSED with dt 38ms steady (~13.5k img/s), projecting ~80 epochs. The run completed cleanly: 83 epochs, total 411.1s, startup 19.0s, VRAM 1709.7 MB. The throughput model was exact (predicted 36–40ms / 80–85 epochs; got 38ms / 83).

## Results

- **Primary metric**: best_test_acc = 95.76% (baseline: 96.71, delta: −0.95pp, −0.98%)
- **Observations**: The failure signature is **convergence to a lower ceiling, not starvation**: epochs 79–83 sat flat at 95.64–95.76 (best 95.76 @ ep 80, final 95.73). This is the opposite of EXP-002/005/007, which were still climbing at cutoff. Epochs were sufficient; the architecture itself is worse. Also notable: fastest early optimization of any run (39.58% at epoch 1) yet lowest ceiling of any capacity experiment — early speed and final quality anti-correlated here.
- **Analysis**: Every input to the plan was validated except the central scientific bet: at this depth regime, depth is not fungible with width. Dropping from 20 to 14 layers (18 → 12 stage convs) removed more representational power than 1.43x params at 6x width restored — the WRN equivalence (16 vs 28 layers at width 8–10) did not transfer down to 14 layers on this pad-shortcut ResNet variant under a heavy-augmentation 300s schedule. Combined with EXP-007, the capacity axis is now closed from BOTH directions: more FLOPs starves epochs (EXP-002/005/007), and FLOPs-neutral reshaping loses representation (EXP-008). The 4x ResNet-20 at 139 epochs sits at a genuine local optimum in the time-constrained architecture space. The search must rotate to levers that act at fixed architecture: regularization that raises signal per epoch (Mixup/CutMix), optimization-quality tricks (EMA), or pure throughput (max-autotune, GPU-side augmentation).
- **Key Learning**: Capacity is closed bidirectionally — wider starves epochs, shallower-wider converges lower; at this budget the deep-thin-enough ResNet-20 4x is the topology optimum, so future gains must come from per-epoch signal or optimization quality, not architecture.

## Verification

- **Conditions**: condition 2 failed (best_test_acc 95.76 < 96.81 = baseline + 0.1pp); condition 1 passed (clean exit, 411.1s ≤ 600); condition 3 skipped per first-failure stop (informally compliant: 83 eval lines = 83 epochs)
- **Review Notes**: results confirmed trustworthy — metric read from run.log summary and cross-checked against the per-epoch eval trail; training_seconds exactly 300.0 (budget integrity); params match the printed ResNet-14 6x header; no constraint violations (GPU 0 only, one eval/epoch, only train.py modified)
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (valid run, metric below baseline + 0.1pp)

## Unexplored Avenues

- **Milder depth cut**: the jump 20→14 may have overshot; there is no NUM_BLOCKS between 2 and 3 in this code's uniform-stage design, but non-uniform stages (e.g. blocks 3/3/2 or 3/2/2 with a width bump) could probe the depth-width frontier more finely. Low priority: the plateau signature suggests depth ≥ ~20 genuinely matters here, and the search has better levers now.
- **Non-uniform stage widths at depth 20** (e.g. 64/128/320 — widen only the cheap 8x8 stage): capacity concentrated where FLOPs are cheapest; the one capacity variant not covered by the bidirectional closure, but expected gain is small given both closures.
- The dt-gate protocol worked exactly as designed and should be standard for any future run whose epoch count is uncertain.

## Next Steps

1. **Mixup or CutMix on the 139-epoch 4x recipe** — now clearly the lead: capacity is closed, regularization is the remaining axis with published CIFAR gains (+0.5–1pp on WRNs), and EXP-006's 139 epochs is the longest schedule yet to absorb it. Confidence: medium.
2. **EMA of weights for eval** — orthogonal optimization-quality lever at near-zero cost; gains under cosine-to-zero schedules are small but the bar is only +0.1pp. Confidence: low-medium.
3. **compile mode="max-autotune"** — throughput probe worth ~3–7% img/s → ~4–9 extra epochs at fixed architecture; likely sub-bar standalone but compounds with any regularization win. Confidence: low standalone.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
