# Report EXP-001: Widen ResNet-20 4x (WRN-style) on the validated recipe
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-001.md
- **Plan**: plans/plan-001.md
- **Log**: logs/exp-log-001.md

## Goal
Maximize best_test_acc (%) on CIFAR-10 within the fixed 300s training budget, modifying only train.py. Direction: higher. Baseline at experiment start: 93.16% (EXP-000, commit be45820). Question: with the recipe bottleneck removed in EXP-000, is model capacity now the binding constraint?

## Idea & Hypothesis
Chosen over a ResNet-9 architecture swap and Cutout augmentation: widen the existing ResNet-20 4× (stage widths 16,32,64 → 64,128,256, ~4.3M params) keeping the EXP-000 recipe byte-identical. Evidence: WRN (arXiv 1605.07146) shows consistent width gains at 16-layer depth up to 12×, training much faster than deep-thin equivalents; every CIFAR speedrun uses wide-shallow nets. Hypothesis: best_test_acc ≥ 93.8%, and total_seconds drops below ~500s as epoch count falls.

## Approach
Three-line diff in train.py: `WIDTH_MULT = 4` constant; `ResNet.__init__(..., width_mult=1)` scaling conv1/bn1, the three stages, and fc in-features; params print includes the multiplier. BasicBlock's zero-pad shortcut handled arbitrary widths unmodified. Recipe (time-keyed one-cycle peak 0.4, bf16/TF32/channels_last, batch 512 nesterov, selective WD 5e-4, label smoothing 0.1, eval once/epoch, seed 42) untouched — single-variable experiment. No deviations from plan.

## Execution
Single run, no retries. GPU 0 confirmed free; background launch with output to run.log. Early signal healthy (4,286,026 params confirmed the change; epoch-1 acc 39.20%). Completed cleanly, exit 0. No errors or dead ends.

## Results

- **Primary metric**: best_test_acc 95.23% (baseline: 93.16%, delta: +2.07 pp, +2.22%)
- **Observations**:
  - Hypothesis target (≥93.8%) exceeded by 1.4pp — width was worth far more than the conservative estimate.
  - Throughput landed mid-prediction: 114 epochs (10,965 steps at ~27ms/step ≈ 18.7k img/s); total_seconds 395.8 — comfortably under the 600s cap, confirming the "wider model relieves eval-overhead pressure" mechanism from goal-learnings.
  - final_test_acc 95.18% ≈ best (95.23%) — stable convergence, no late-run noise harvesting; final_test_loss 0.2447 (vs 0.2998 at EXP-000).
  - peak_vram_mb 1620.7 — still trivial against 98GB; width can go higher.
- **Analysis**: Capacity was indeed the post-recipe binding constraint, and the time-keyed schedule self-adapted to 3× lower throughput with zero retuning — validating the recipe's composability claim. The trajectory (91.97 → 93.16 → 95.23) shows recipe and capacity contributions are roughly additive so far. The gradient of accuracy vs width is still steep at 4×; WRN evidence says gains continue to 8–12× width at this depth, and VRAM/wall-clock headroom both allow it.
- **Key Learning**: After fixing the schedule, accuracy scales strongly with width under a fixed time budget — and heavier models *reduce* wall-clock risk by cutting eval count; the next capacity step (8×) is the obvious move with peak-LR caution.

## Verification
- **Conditions**: all passed — (1) clean completion, 395.8s ≤ 600s; (2) 95.23 ≥ 93.26 (baseline+0.1); (3) 114 evals = 114 epochs
- **Review Notes**: results confirmed trustworthy — single-variable diff in train.py (intended intervention class), seed 42 unchanged, eval harness untouched, num_params printout confirms the architecture change, final_test_acc corroborates.
- **Verdict**: improvement
- **Verdict Basis**: all conditions passed + meaningful improvement (+2.07 pp ≫ 0.1 pp bar)

## Unexplored Avenues
- **Wider still (8×, stage widths 128/256/512, ~17M params)**: WRN gains persist to 8–12× at 16 layers; throughput would drop to ~5-8k img/s → ~30-50 epochs — may start to undertrain; peak LR may need lowering (0.3) for stability.
- **Width + regularization**: at 4.3M params and 114 epochs, train/test gap is growing (train loss EMA well below test) — Cutout/RandomErasing now has capacity to regularize, unlike at 270k params.
- **ResNet-9-style topology**: still untried; could trade depth for width more efficiently (fewer sequential layers → higher img/s at same params).
- **Larger batch (1024) at higher width**: fewer steps/epoch, even fewer eval boundaries, better GPU utilization for the wide net.

## Next Steps
1. **8× width (or 6×) with peak LR 0.3–0.4** — high confidence: the width gradient is steep, VRAM/wall-clock headroom is large; watch for undertraining at ~30–50 epochs.
2. **Add RandomErasing/Cutout on top of 4× width** — medium-high confidence: capacity now exists to benefit from stronger regularization (knowledge: arXiv 1708.04552).
3. **ResNet-9 topology at matched params** — medium confidence: speedrun-proven, but recipe transfer risk; consider after width is exhausted.

## Exit Action Results
