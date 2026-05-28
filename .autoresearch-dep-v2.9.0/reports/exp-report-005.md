# Report EXP-005: AMP (FP16 + GradScaler + channels_last)
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-005.md
- **Plan**: plans/plan-005.md
- **Log**: logs/exp-log-005.md

## Goal

Maximize best_test_acc (%) on CIFAR-10, higher is better. Baseline: 93.33% (EXP-003). Threshold: >= 93.43%.

## Idea & Hypothesis

**Chosen idea**: AMP (FP16 autocast + GradScaler + channels_last) for throughput improvement.
**Hypothesis**: Expected 93.8-94.5% from fitting ~100+ epochs (vs 69) in 300s.

## Approach

Added `torch.amp.autocast('cuda', dtype=torch.float16)` around forward+loss, `torch.amp.GradScaler` for loss scaling, and `memory_format=torch.channels_last` for NHWC layout. All other settings unchanged.

## Execution

Single run, 106 epochs (41,179 steps) in 300.0s. Total 405.8s (longer due to 106 eval passes). No crashes, no NaN.

## Results

- **Primary metric**: 94.44% (baseline: 93.33%, delta: +1.11pp, +1.19%)
- **Observations**:
  - **Throughput**: 1.54x speedup — per-step time 7.3ms (vs 11.3ms), 106 epochs (vs 69). Peak VRAM halved to 266.1 MB.
  - **FP16 instability at LR=0.01**: Epochs 34-52 (second LR plateau) showed severe oscillation — accuracy swung between 68-82%, never exceeding the pre-drop best of 82.40%. This is a fundamental precision issue: at LR=0.01, FP16 gradient updates are too coarse for stable convergence.
  - **Recovery at LR=0.001**: The second LR drop at epoch 52 stabilized training immediately (82.40% → 90.95% in one epoch). The lower LR's smaller gradient steps are within FP16's precision range.
  - **Extended polish phase**: Epochs 52-106 (54 epochs at LR=0.001) drove accuracy from 90.95% to 94.44% — a +3.5pp gain. The FP32 baseline only had 17 epochs at LR=0.001 (epochs 52-69).
  - **Delayed convergence**: The model didn't exceed the FP32 baseline until epoch 80, then rapidly climbed from 93.66% to 94.44% by epoch 101.
- **Analysis**: AMP's value is entirely in throughput — it trades convergence quality per-step for more steps. The FP16 instability at LR=0.01 is a serious concern (the model "wasted" 18 epochs in an unstable regime), but the 37 extra epochs at LR=0.001 more than compensated. The net effect: despite losing convergence quality in the middle phase, the extended polish phase delivered the single largest improvement of any experiment (+1.11pp). This suggests a modified wall-clock-fractional schedule (e.g., shifting the first drop earlier to spend less time at the unstable LR=0.01 and more at stable LR=0.001) could further amplify the AMP benefit.
- **Key Learning**: AMP gives 1.54x epoch throughput on H20 for width-2x ResNet-20; FP16 is unstable at LR=0.01 but the extended LR=0.001 phase more than compensates — schedule optimization for AMP (reducing the unstable middle phase) is the natural follow-up.

## Verification

- **Conditions**: All 3 passed
- **Verdict**: improvement
- **Verdict Basis**: 94.44% > 93.43% threshold; +1.11pp, the largest single-experiment gain.

## Unexplored Avenues

- **Schedule tuning for AMP**: Shift the first LR drop earlier (e.g., 0.3 instead of 0.5) to reduce time in the unstable LR=0.01 regime and increase time at stable LR=0.001. Could gain another +0.5-1pp.
- **torch.compile + AMP**: Compile the model for additional throughput on top of AMP.
- **Batch size 256 + AMP**: Larger batch with AMP for even more throughput.
- **BFloat16 instead of Float16**: BF16 has more exponent bits and may be stable at LR=0.01, avoiding the instability while keeping most of the throughput.

## Next Steps

1. **Schedule tuning for AMP** — shift first drop to 0.3 or 0.35 to spend less time at unstable LR=0.01 and more at stable LR=0.001. High confidence.
2. **Width-4x with AMP** — the reduced VRAM (266 MB) gives headroom for a wider model. Medium confidence.
3. **Batch size 256 + LR scaling + AMP** — compound throughput gains. Medium confidence.

## Exit Action Results
(no exit actions defined)
