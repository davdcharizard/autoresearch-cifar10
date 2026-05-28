# Report EXP-006: Schedule Optimization (0.35, 0.55) for AMP
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-006.md
- **Plan**: plans/plan-006.md
- **Log**: logs/exp-log-006.md

## Goal
Maximize best_test_acc (%). Baseline: 94.44% (EXP-005). Threshold: >= 94.54%.

## Idea & Hypothesis
Shift LR drops from (0.5, 0.75) to (0.35, 0.55) to minimize time in FP16-unstable LR=0.01 and maximize time at stable LR=0.001. Expected 94.8-95.5%.

## Approach
Changed two constants in `_wall_clock_fractional_step_decay`: 0.5→0.35, 0.75→0.55.

## Execution
Single run, 106 epochs, 300.0s training, no errors.

## Results
- **Primary metric**: 94.17% (baseline: 94.44%, delta: -0.27pp)
- **Observations**:
  - The 0.01 phase (epochs 35-58) was much more stable than EXP-005's — reached 92.29% best vs EXP-005's 82.40%. The instability was dramatically reduced.
  - BUT the model peaked at 94.17% vs EXP-005's 94.44% despite having more epochs at LR=0.001 (~48 vs ~26).
  - The high-LR phase only had 35 epochs (vs 53 in EXP-005), reaching 82.40% pre-drop (same as EXP-005's). So the exploration quality was similar but the shorter duration means fewer gradient updates for feature discovery.
  - The 0.001 phase converged to a lower ceiling (94.17% vs 94.44%) — the reduced high-LR exploration limited the basin of attraction the low-LR phase could polish.
- **Analysis**: The hypothesis was wrong. The FP16 instability at LR=0.01 was a symptom, not the root cause of the accuracy gap. The real mechanism is that the high-LR exploration phase (LR=0.1) is where the model discovers good features and basins of attraction — reducing it from 50% to 35% of budget gave the model less time to explore, resulting in a lower convergence ceiling that the extended 0.001 phase couldn't overcome. The original (0.5, 0.75) schedule is near-optimal for this model despite the FP16 instability.
- **Key Learning**: High-LR exploration time is the primary driver of final accuracy ceiling; the unstable LR=0.01 phase in AMP is a necessary cost of adequate exploration, not a waste to be eliminated.

## Verification
- **Conditions**: Condition 1 FAILED — 94.17% < 94.54%
- **Verdict**: no-improvement

## Unexplored Avenues
- BFloat16 instead of Float16 — may be stable at LR=0.01 without sacrificing exploration time
- Width-4x with AMP — increase capacity to push the ceiling higher
- Batch size scaling with AMP — different throughput path

## Next Steps
1. **Width-4x with AMP** — the VRAM headroom (266 MB of 98 GB) and AMP throughput make a wider model feasible. High confidence this is the highest-impact direction.
2. **BFloat16** — may avoid the LR=0.01 instability entirely. Medium confidence.
3. **Batch size 256 + LR scaling** — throughput without changing precision dynamics. Medium confidence.

## Exit Action Results
(no exit actions defined)
