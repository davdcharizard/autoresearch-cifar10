# Brainstorm EXP-008
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **"94% on CIFAR-10 in 3.29 Seconds on a Single GPU"** (https://arxiv.org/html/2404.00498v1)
  torch.compile delivers ~16% wall-clock speedup on ResNet CIFAR-10 training (3.29s compiled vs 3.83s non-compiled on A100). The speedup comes from graph-level fusion and CUDA kernel optimization. Compilation happens lazily on first forward pass.

- **PyTorch torch.compile docs — compilation overhead** (https://pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html)
  Compilation latency scales with graph size. For small ResNets, regional compilation or a warmup pass amortizes the one-time cost. `mode="reduce-overhead"` uses CUDA graphs for additional speedup but increases memory; `mode="default"` is the safest starting point.

- **OpenMixup CIFAR benchmarks** (https://github.com/Westlake-AI/openmixup/blob/main/docs/en/mixup_benchmarks/Mixup_cifar.md)
  CutMix and Mixup improve WRN CIFAR-10 accuracy by ~0.3-0.8pp over vanilla training at 200 epochs. CutMix outperforms Cutout by replacing removed regions with useful content from another image.

## Experimental History Review

Current baseline: **94.82%** (EXP-007, width-4x + AMP + aug + WD=5e-4 + (0.5, 0.75) schedule, 83 epochs at 9ms/step).

Key trajectory:
- BASE (91.72%) → EXP-001 width-2x (+0.57pp) → EXP-002 aug (+0.63pp) → EXP-003 WD (+0.41pp) → EXP-005 AMP (+1.11pp) → EXP-007 width-4x (+0.38pp) = 94.82%.
- Failed: EXP-000 (cosine T_max mismatch), EXP-004 (Nesterov+LS per-step overhead), EXP-006 (earlier LR drops reduced exploration).

Key patterns:
- **Epoch count is the binding constraint**: With 83 epochs vs WRN paper's 200 epochs, the model has 58% fewer training iterations. Every technique that adds epochs in the LR=0.001 phase compounds the accuracy ceiling.
- **AMP Tensor Cores scale with width**: Width-4x at 9ms/step with AMP (vs 11ms without AMP at width-2x). The H20 GPU is not compute-saturated.
- **FP16 instability at LR=0.01 recovered better with width-4x**: Epochs 40-41 dipped to 80-81% but recovered by epoch 42. Width-2x was stuck for 18 epochs.
- **Peak VRAM = 485 MB**: Massive headroom on H20's 98 GB. Memory is not a constraint.

Untried:
- torch.compile for throughput (recommended in EXP-007 report)
- Batch-level augmentation (CutMix / Mixup)
- Larger batch size for throughput
- Cosine annealing to smooth the FP16-unstable LR drops

## Candidate Ideas

### 1. torch.compile with warmup pass

**Summary**: Add `model = torch.compile(model)` after model creation, with a dummy forward+backward warmup pass before the training loop. The warmup triggers compilation outside the per-step time accumulation (`total_training_time` only counts `dt` inside the training loop), so compilation overhead doesn't eat into the 300s training budget. Expected 10-15% per-step speedup from graph-level kernel fusion, yielding ~91-96 epochs (vs 83 currently).

**Reasoning**: The "94% in 3.29s" benchmark shows torch.compile delivers ~16% speedup on ResNet CIFAR-10 training. Our model already uses AMP + channels_last, so the marginal benefit may be slightly lower (~10-15%), but even 10% means ~8-13 additional epochs. These extra epochs all fall in the LR=0.001 phase (the current 83 epochs already cover the full 0.5/0.75 schedule), and the goal-learning "AMP extended LR=0.001 phase is where throughput gains pay off" (EXP-005) directly predicts this will improve accuracy. The mechanism is clear: more optimization steps at the convergent LR regime.

**Sources**: arxiv.org/html/2404.00498v1 (16% compile speedup), EXP-005 report (LR=0.001 phase delivers gains), EXP-007 report "Next Steps" item 1.

**Estimated Effort**: Very low — one `torch.compile()` call + ~5 lines of warmup code.

**Risk Assessment**: Low. If torch.compile doesn't help on H20 (Hopper arch), we get a no-improvement, not a crash. The warmup pass ensures the first training step isn't penalized by compilation. Worst case: compilation overhead leaks into the first few backward passes despite the warmup, costing ~5-10s of budget and netting fewer new epochs than expected. The `mode="default"` avoids the memory growth issues reported with `mode="reduce-overhead"`.

### 2. CutMix batch-level augmentation

**Summary**: Add CutMix augmentation that operates on batches in the training loop. For each batch, with probability 0.5, sample lambda from Beta(1.0, 1.0), cut a rectangular region from one image and paste it onto another, and mix the labels proportionally. This is complementary to TrivialAugmentWide (which operates per-image) and RandomErasing (which removes information — CutMix replaces removed regions with useful content).

**Reasoning**: OpenMixup benchmarks show CutMix improves WRN CIFAR-10 accuracy by ~0.3-0.8pp at 200 epochs. CutMix's mechanism is well-understood: it forces the model to learn from partial object views while preserving the information content that Cutout/RandomErasing destroys. At 83 epochs, the improvement may be smaller (~0.2-0.4pp) but the technique is strongly evidenced. CutMix's computational cost is minimal (tensor indexing + label interpolation), so epoch count should be unaffected.

**Sources**: CutMix paper (Yun et al., 2019), OpenMixup benchmarks, EXP-002 (TrivialAugmentWide +0.63pp with negligible overhead).

**Estimated Effort**: Low-medium — ~15-20 lines in the training loop (cutmix function + conditional application + mixed label cross-entropy).

**Risk Assessment**: Medium. The interaction between CutMix and TrivialAugmentWide is not well-studied — both modify images, and compound augmentation can over-regularize (making training harder without compensating gains). With only 83 epochs, over-regularization is more dangerous than with 200 epochs. The mixed-label cross-entropy changes the loss landscape, which could interact with FP16 stability. Fallback: if combined augmentation hurts, CutMix alone (removing TrivialAugmentWide) may still improve.

### 3. Batch size 256 + LR 0.2 (linear scaling)

**Summary**: Double BATCH_SIZE from 128 to 256 and LR from 0.1 to 0.2 (linear scaling rule, Goyal et al. 2017). The larger batch improves GPU utilization — at width-4x with AMP, per-step VRAM is only 485 MB on a 98 GB GPU, and batch 128 likely underutilizes the H20's compute. Expected ~30% throughput increase (fewer steps per epoch, slightly longer per-step time), yielding ~108-115 epochs.

**Reasoning**: The H20 has massive compute headroom. At batch 128 with 9ms/step, the GPU is spending time on kernel launch overhead relative to compute. Doubling the batch amortizes this overhead, and the linear scaling rule preserves SGD optimization dynamics. More epochs mean more LR=0.001 training. EXP-005 showed that going from 69 to 106 epochs yielded +1.11pp; going from 83 to ~110 epochs via batch size (while keeping width-4x capacity) could yield ~0.3-0.5pp from the same mechanism.

**Sources**: Goyal et al. 2017 (linear scaling rule), EXP-005 (throughput → epochs → accuracy), EXP-007 (485 MB VRAM at batch 128).

**Estimated Effort**: Very low — two constants (BATCH_SIZE=256, LR=0.2).

**Risk Assessment**: Medium. LR=0.2 with FP16 may be less stable than LR=0.1 (FP16 gradients are already coarse). The generalization gap from larger batches is well-documented — accuracy can drop ~0.2-0.5pp from weaker stochastic regularization. The linear scaling rule is an approximation; the actual optimal LR for batch 256 may differ. Additionally, the WEIGHT_DECAY effective strength changes with batch size (more gradient accumulation per step → same WD penalty → relatively stronger regularization).

## Idea Evaluation

**Evidence**: Candidate 1 (torch.compile) has the most direct evidence — a published benchmark showing 16% speedup on the exact task (ResNet CIFAR-10), and our own experiment history confirming that more epochs at LR=0.001 translate to accuracy gains. Candidate 2 (CutMix) has strong literature support but specifically at 200 epochs; the 83-epoch regime is untested. Candidate 3 (batch 256) has the linear scaling rule as theoretical backing but no specific CIFAR-10 WRN benchmark at our epoch count.

**Mechanism clarity**: Candidate 1 has the clearest mechanism — same training dynamics, just more of them. Candidates 2 and 3 both change training dynamics (augmentation and batch size respectively), introducing confounding factors.

**Expected impact**: Candidate 3 potentially adds the most epochs (~108-115), but the generalization gap from larger batches may offset the gains. Candidate 1 adds moderate epochs (~91-96) with zero training-dynamics risk. Candidate 2 adds accuracy per-epoch but not more epochs.

**Risk**: Candidate 1 is the safest — if torch.compile doesn't speed up H20, we get no-improvement with no destabilization. Candidate 2 risks over-regularization at 83 epochs. Candidate 3 risks FP16 instability at LR=0.2.

**Decision**: Candidate 1 (torch.compile) is the strongest choice. It has the best evidence-to-risk ratio, the clearest mechanism, and follows the validated "more epochs at LR=0.001" recipe from EXP-005/EXP-007. If it succeeds, Candidates 2 and 3 become natural follow-ups.

## Chosen Idea

**Selected**: Candidate 1 — **torch.compile with warmup pass**

**Why this idea**: Targets the epoch count gap (83 vs WRN paper's 200) via a zero-risk throughput improvement. The mechanism is the same one that drove EXP-005's +1.11pp success: more training steps at LR=0.001. The published 16% speedup on the exact task (ResNet CIFAR-10) provides the strongest evidence of any candidate.

**Hypothesis**: Adding `torch.compile(model)` with a warmup pass will increase epoch count from ~83 to ~91-96 by reducing per-step time from 9ms to ~7.5-8ms. These additional epochs in the LR=0.001 phase will raise best_test_acc from 94.82% to **95.0-95.2%**, closing the gap to the WRN paper's 95.3% anchor. Improvement threshold: >= 94.92%.
