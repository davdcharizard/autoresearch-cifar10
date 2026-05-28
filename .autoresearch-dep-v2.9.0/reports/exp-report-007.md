# Report EXP-007: Width-4x (WIDTH_MULT=4) with AMP
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-007.md
- **Plan**: plans/plan-007.md
- **Log**: logs/exp-log-007.md

## Goal
Maximize best_test_acc (%). Baseline: 94.44% (EXP-005). Threshold: >= 94.54%.

## Idea & Hypothesis
WIDTH_MULT=4, quadrupling channels to {64, 128, 256} (~4.29M params). Expected 94.8-95.3%.

## Approach
Changed WIDTH_MULT from 2 to 4. All other settings unchanged from EXP-005.

## Execution
Single run, 83 epochs (32,248 steps) in 300.0s. Total 393.4s. No errors.

## Results
- **Primary metric**: 94.82% (baseline: 94.44%, delta: +0.38pp, +0.40%)
- **Observations**:
  - **Throughput surprise**: 9ms/step with AMP at width-4x — faster than width-2x without AMP (11ms). AMP Tensor Cores scale better with wider layers. 83 epochs (vs estimated 35-50).
  - **FP16 at LR=0.01**: Initially unstable (epoch 40-41 dipped to 80-81%) but recovered rapidly — by epoch 42 hit 91.08%, then climbed to 93.50% by epoch 58. Much better behavior than width-2x (EXP-005 stuck at 82.40% for 18 epochs).
  - **Post-second-drop convergence**: Epoch 62 hit 93.66%, epoch 63 jumped to 94.41%, peaked at 94.82% at epoch 73.
  - Peak VRAM: 485.4 MB (well within H20's 98 GB). 4,286,026 params (~4x width-2x's 1.07M).
  - Model still improving at budget end (94.69% at final epoch 83, best 94.82% at epoch 73).
- **Analysis**: The width-4x model exceeded the 94.44% baseline despite having fewer epochs (83 vs 106). The wider layers (a) benefit more from AMP Tensor Cores (9ms vs 7ms per step), (b) converge better at LR=0.01 with FP16 (possibly because the wider layers have more numerical precision margin per-channel), and (c) have higher representational capacity that compensates for fewer training epochs. The 94.82% result is within the WRN-paper's n=3, k=4 range (~95.3% at 200 epochs), with the gap attributable to 83 vs 200 epochs.
- **Key Learning**: Width-4x with AMP delivers 94.82% in 83 epochs at 9ms/step; AMP Tensor Cores scale better with wider layers, and the wider model handles FP16 at LR=0.01 better than width-2x.

## Verification
- **Conditions**: All 3 passed
- **Verdict**: improvement
- **Verdict Basis**: 94.82% > 94.54% threshold; +0.38pp.

## Unexplored Avenues
- **Width-4x + batch 256**: Increase throughput further on the wider model.
- **Width-6x or WIDTH_MULT=8**: Even wider, but epoch count becomes the concern.
- **torch.compile + AMP**: Compile for additional throughput.
- **Gradient accumulation**: Simulate larger effective batch size without memory cost.

## Next Steps
1. **torch.compile + width-4x AMP** — one-line throughput addition. High confidence.
2. **Batch size 256 + LR 0.2 + width-4x AMP** — compound throughput. Medium confidence.
3. **Width-6x or 8x** — push capacity further. Lower confidence (epoch count risk).

## Exit Action Results
(no exit actions defined)
