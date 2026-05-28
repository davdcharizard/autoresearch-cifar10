# Report EXP-008: torch.compile with warmup pass
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-008.md
- **Plan**: plans/plan-008.md
- **Log**: logs/exp-log-008.md

## Goal
Maximize best_test_acc (%) on CIFAR-10, higher is better. Baseline: 94.82% (EXP-007, WIDTH_MULT=4 + AMP + aug + WD=5e-4). Threshold for improvement: > 94.92% (baseline + 0.1pp).

## Idea & Hypothesis
torch.compile with a warmup pass was chosen as the highest evidence-to-risk option. Published benchmarks showed 16% speedup on ResNet CIFAR-10 training via graph-level kernel fusion. The hypothesis: reducing per-step time from 9ms to ~7.5-8ms would increase epoch count from 83 to ~91-96, with the additional LR=0.001 epochs raising accuracy to 95.0-95.2%.

## Approach
Added `model = torch.compile(model)` after `.to(device, memory_format=torch.channels_last)`. Inserted a warmup block (dummy forward+backward under AMP autocast) before the training timer to trigger compilation outside the 300s budget. Used `mode="default"` (implicit). Two implementation bugs were fixed: (1) `torch.randn` doesn't accept `memory_format` kwarg — used `.to(memory_format=...)` after creation, (2) `scaler.update()` requires prior `scaler.step()` — used plain `loss.backward()` for warmup instead.

## Execution
Single local run. Warmup compilation succeeded in ~15s. Training ran 82 epochs in 300.0s at 9-10ms/step — identical throughput to the EXP-007 baseline (9ms/step, 83 epochs). No adjustments were made mid-run since the throughput signal was clear early: torch.compile was not providing any speedup.

## Results
- **Primary metric**: 94.75% (baseline: 94.82%, delta: -0.07pp, -0.07%)
- **Observations**: Per-step time with torch.compile was 9-10ms, identical to the 9ms baseline. Epoch count dropped by 1 (82 vs 83) likely due to slight per-step overhead from Inductor dispatch. A TensorFloat32 warning from Inductor suggested the model may not be leveraging TF32 matmul. Peak VRAM unchanged at 484 MB.
- **Analysis**: The hypothesis was invalidated. The published 16% speedup was on an A100 — the H20 (Hopper architecture) appears to already be well-optimized for this workload via AMP + channels_last. Inductor found no additional fusion opportunities. The slight accuracy drop (-0.07pp) is within run-to-run variance but also consistent with 1 fewer epoch of LR=0.001 training.
- **Key Learning**: torch.compile provides zero measurable speedup on H20 for a small ResNet with AMP+channels_last — the kernel overhead is already minimized by AMP and the model is too small for graph-level fusion to find wins.

## Verification
- **Conditions**: Condition 1 (best_test_acc > 94.92%) FAILED — 94.75%. Conditions 2-3 PASSED.
- **Review Notes**: Results confirmed trustworthy. The metric is plausible given zero throughput improvement and 1 fewer epoch.
- **Verdict**: no-improvement
- **Verdict Basis**: Primary verification condition failed — 94.75% < 94.92% threshold.

## Unexplored Avenues
- `mode="reduce-overhead"` uses CUDA graphs and may help by eliminating kernel launch overhead, though it increases memory usage. However, given the model's small size, kernel launch is unlikely to be the bottleneck.
- `mode="max-autotune"` lets Inductor try more CUDA kernel configurations. May find better kernels on H20 specifically, but adds significant compilation time.
- The idea of throughput improvement through compiler optimization may be fundamentally limited for this model size on H20. Throughput gains should be pursued through batch size increases or larger models that better utilize the GPU.

## Next Steps
1. **Batch size 256 + LR 0.2 (linear scaling)** — (high confidence) — directly increases GPU utilization by amortizing per-batch overhead. With 484 MB VRAM at batch 128, there's massive headroom. Expected ~30% throughput increase yielding ~108 epochs.
2. **CutMix batch-level augmentation** — (medium confidence) — improves accuracy per-epoch rather than adding epochs. Well-evidenced at 200 epochs but untested at 83 epochs. Risk of over-regularization with existing TrivialAugmentWide.
3. **Combined batch size + CutMix** — (medium confidence) — if batch size succeeds, CutMix on top could compound the gains. The additional epochs would also give CutMix more time to take effect.

## Exit Action Results
