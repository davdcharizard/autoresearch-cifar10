# Brainstorm EXP-001
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **PyTorch `torch.compile` documentation** (https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
  PyTorch documents `torch.compile` as a compiler API with Inductor as the default backend. This is relevant because the current model is a simple static-shape CNN where compiler optimization may reduce per-step overhead without changing the benchmark target.

- **PyTorch Automatic Mixed Precision examples** (https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
  PyTorch's AMP examples describe using autocast and gradient scaling for mixed-precision training. This is relevant for H20 Tensor Core hardware because mixed precision may increase convolution throughput, though precision changes can affect accuracy.

- **PyTorch Channels Last Memory Format tutorial** (https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
  PyTorch documents channels-last memory format as a convolution-oriented optimization path, especially with reduced precision on NVIDIA Tensor Core GPUs. This suggests a throughput-focused experiment that preserves the baseline architecture and data recipe.

- **Saved SGDR/cosine note** (`knowledge/papers/sgdr-cosine-schedule.md`)
  EXP-000 showed cosine over the 64k `MAX_STEPS` horizon was too slow for the observed 35k-step wall-clock run, so schedule changes should be calibrated to actual steps if revisited.

## Experimental History Review

- Current baseline remains `best_test_acc=91.52%`; EXP-000 reached 90.45% and was recorded as `no-improvement`.
- Failed approach: combining exact 16x16 cutout, label smoothing, Nesterov, and cosine over 64k steps undertrained ResNet-20 within 300 seconds.
- EXP-000 completed only 35,279 steps. Because the baseline schedule drops LR at 32,000 and 48,000 steps, any throughput increase that reaches more post-drop updates could plausibly improve peak accuracy without changing data augmentation or model capacity.
- The next experiment should not retry strong regularization plus slow cosine. Isolating schedule or throughput effects is better grounded than another multi-component regularization bundle.

## Candidate Ideas

### 1. Baseline-Preserving Throughput Acceleration
**Summary**: Keep the baseline model, augmentation, optimizer, LR milestones, and loss unchanged, but try speed-focused implementation settings in `train.py`: enable cuDNN benchmark mode, use channels-last tensors/model memory format, wrap the forward/loss in CUDA autocast with BF16, and optionally compile the model. The goal is to complete more optimizer steps within the fixed 300 training seconds while preserving the baseline recipe.

**Reasoning**: The fixed wall-clock budget means faster steps can become more training progress. EXP-000 completed only ~35k steps, barely past the first LR milestone and far short of the second. A throughput-only experiment has a clear mechanism: if it increases steps enough to spend more time at lower LR, `best_test_acc` may improve without the over-regularization failure from EXP-000.

**Sources**: PyTorch `torch.compile` docs; PyTorch AMP examples; PyTorch channels-last tutorial; EXP-000 report `reports/exp-report-000.md`.

**Estimated Effort**: medium

**Risk Assessment**: BF16 autocast may slightly change optimization numerics, and `torch.compile` can add first-iteration overhead or fail on unsupported patterns. Channels-last may provide little benefit for small CIFAR tensors. Worst case is a clean no-improvement or a code error that can be fixed by disabling compile while keeping channels-last/AMP.

### 2. Schedule-Only Step Milestone Retuning
**Summary**: Keep the baseline architecture and augmentation unchanged but move LR milestones earlier, such as `[24000, 32000]`, so the run spends more of its observed 35k-step budget in lower LR regimes.

**Reasoning**: EXP-000 suggests late high LR can be a problem under the 300 second budget. The baseline only reaches the first milestone near the end, so earlier decay might improve final refinement and peak test accuracy without adding regularization or model overhead.

**Sources**: EXP-000 report `reports/exp-report-000.md`; saved SGDR/cosine note `knowledge/papers/sgdr-cosine-schedule.md`; local scheduler in `train.py`.

**Estimated Effort**: low

**Risk Assessment**: Earlier LR decay may undertrain if the current baseline depends on high LR exploration for most of the run. It has a smaller expected upside than increasing actual training progress, because it only redistributes the same number of steps.

### 3. Compact WRN-16-2 Architecture
**Summary**: Replace ResNet-20 with a compact WRN-style CIFAR architecture, likely depth 16 and width factor 2, while initially keeping the baseline crop/flip augmentation and SGD family. Adjust LR schedule only enough to account for the lower expected step count.

**Reasoning**: WRN evidence shows wider residual networks can outperform thin ResNets on CIFAR tasks. The H20 has ample VRAM, and the task explicitly allows model architecture changes. This has a higher ceiling than recipe tuning if the baseline is capacity limited.

**Sources**: `knowledge/papers/wide-residual-networks.md`; Wide Residual Networks paper https://arxiv.org/abs/1605.07146; task note that VRAM increases are acceptable for meaningful accuracy gains.

**Estimated Effort**: medium

**Risk Assessment**: WRN may reduce steps per 300 seconds and require schedule tuning. A new architecture adds implementation risk, and if schedule is not retuned the model may not converge enough within the budget.

## Idea Evaluation

Candidate 1 has the best risk-adjusted mechanism for the next loop because it preserves the known-good baseline recipe and targets a resource bottleneck revealed by EXP-000: insufficient useful late-budget optimization. Unlike EXP-000, it does not add regularization or change the intended statistical training signal. Its success criterion is also easy to diagnose: if step count rises and accuracy improves, throughput was limiting; if step count rises but accuracy does not, the baseline may need model or schedule changes instead.

Candidate 2 is the lowest-effort experiment and directly addresses late LR behavior, but it cannot create more training signal. It may be a good follow-up if throughput acceleration fails or destabilizes.

Candidate 3 has the highest potential ceiling, but it is a larger architecture experiment. It should follow a cleaner baseline-preserving throughput test unless the throughput path crashes or clearly cannot move enough steps.

## Chosen Idea
**Selected**: Baseline-Preserving Throughput Acceleration

**Why this idea**:
It tests a clear, low-confound hypothesis: the baseline may improve if the same recipe reaches more optimizer steps and lower-LR refinement within the fixed training budget. It also avoids the exact failed pattern from EXP-000 by leaving augmentation, loss regularization, and the LR schedule conceptually unchanged.

**Hypothesis**:
Using channels-last layout, cuDNN benchmarking, BF16 autocast, and `torch.compile` where compatible will increase the number of training steps completed in 300 seconds; if the baseline is partly step-limited, `best_test_acc` will exceed 91.52% without changing model architecture or evaluation.
