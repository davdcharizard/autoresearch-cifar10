# Brainstorm EXP-000
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **How to Train Your ResNet / cifar10-fast (David Page)** (https://github.com/davidcpage/cifar10-fast)
  ResNet-9 trained to 94% in well under the paper's 64k-step regime using OneCycle LR, batch 512, mixed precision. Demonstrates that for short time budgets, a budget-matched one-cycle schedule plus large-batch mixed-precision training dominates the classic step-decay recipe.
- **94% on CIFAR-10 in 3.29 Seconds on a Single GPU (Keller Jordan, arXiv 2404.00498)** (https://arxiv.org/abs/2404.00498 ; https://github.com/KellerJordan/cifar10-airbench)
  GPU-resident dataset (no DataLoader), channels_last + half precision, derandomized horizontal flipping, whitening-based first-layer init. Shows the data pipeline and precision/memory-format are the dominant throughput levers for small models on big GPUs.
- **Super-Convergence (Smith & Topin, arXiv 1708.07120)** (https://arxiv.org/abs/1708.07120)
  One-cycle schedules with a large peak LR reach high accuracy in far fewer iterations than step decay — directly applicable since the baseline's step-decay schedule never completes within the time budget.

## Experimental History Review

First experiment under this goal — no prior history. Baseline (BASE row in experiment index): best_test_acc 91.97% at commit 14247e0; 97 epochs / 37,515 steps in the 300s budget, peak VRAM 330MB, 270k params.

Key codebase observations driving ideation (train.py at 14247e0):
- **Schedule/budget mismatch**: `MultiStepLR` milestones at 32k/48k steps, but only ~37.5k steps fit in 300s. The second LR drop never fires — training ends mid-schedule at LR 0.01, never annealing further. The recipe was designed for 64k steps that never happen.
- **Throughput on the table**: FP32, batch 128, default (NCHW) memory format, host DataLoader with per-step `cuda.synchronize()`. ~16k img/s on an H20 — a tiny 270k-param model on a 98GB datacenter GPU is host/latency-bound, not compute-bound.
- VRAM is at 330MB of ~98GB — enormous headroom (VRAM is a soft constraint).
- Eval already runs exactly once per epoch (constraint-compliant); eval time is excluded from the training budget but counts toward the 10-min wall clock (370s total at 97 epochs — headroom exists but more epochs ⇒ more eval overhead).

## Candidate Ideas

### 1. Budget-matched modern training recipe (same ResNet-20)
**Summary**: Keep the architecture; modernize the training recipe to fit the actual 300s budget. (a) Replace MultiStepLR with a one-cycle/cosine schedule whose horizon equals the steps that actually fit in the budget (estimate steps from measured early-step throughput, or schedule on fraction-of-budget-elapsed rather than step count so the anneal always completes). (b) bf16 autocast for forward/backward + TF32 matmuls + `channels_last` memory format + `cudnn.benchmark=True` for throughput. (c) Batch size 512 with linearly scaled peak LR and Nesterov momentum. (d) No weight decay on BatchNorm/bias parameters. (e) Label smoothing 0.1. Raise `MAX_STEPS` so it never binds before the time budget.

**Reasoning**: The single clearest defect is the schedule that never completes — super-convergence literature and the cifar10-fast recipe both show budget-matched one-cycle schedules beat truncated step decay decisively in fixed-iteration regimes. The throughput items (bf16, channels_last, larger batch) are standard, near-risk-free wins on H20-class hardware that compound the schedule fix by fitting ~1.5–3× more epochs into the same 300s. Each component has strong public evidence in exactly this setting (small CNN, CIFAR-10, short budget).

**Sources**: cifar10-fast (Page), arXiv 1708.07120 (super-convergence), arXiv 2404.00498 §2 (channels_last/half-precision throughput), baseline run metrics in experiment index.

**Estimated Effort**: low–medium (single-file edits to hyperparameters, schedule, and training loop; no structural rewrite)

**Risk Assessment**: Large-batch + high peak LR can diverge — mitigated by warmup phase of the one-cycle schedule and the well-tested batch-512 recipe from cifar10-fast. bf16 numerics on BN are safe (BN stays fp32 under autocast). If the schedule is keyed to predicted total steps and throughput drifts, the anneal could end early/late — keying it to elapsed-budget-fraction removes that failure mode. Worst case: accuracy regression → no-improvement verdict; nothing here risks crash beyond ordinary bugs.

### 2. GPU-resident data pipeline (airbench-style)
**Summary**: Eliminate the host DataLoader for training: load the whole CIFAR-10 train set once into GPU memory as normalized tensors, and implement batch-wise augmentation (pad+random-crop via indexing, random horizontal flip) directly on GPU. Per-step host work drops to index generation; the per-step `synchronize()` stays (it's how the budget is metered) but the work between syncs shrinks dramatically.

**Reasoning**: airbench demonstrates the data pipeline is the dominant cost for tiny models — its 3.29s result is unreachable with a host DataLoader. The dataset is 50k×3×32×32 ≈ 600MB fp32 (or 150MB uint8) — trivially fits in 98GB. More steps/s ⇒ more epochs ⇒ more anneal-complete training in 300s.

**Sources**: arXiv 2404.00498, https://github.com/KellerJordan/cifar10-airbench.

**Estimated Effort**: medium–high (custom augmentation correctness, shuffling, dtype handling; must reproduce RandomCrop(32, padding=4)+flip semantics exactly)

**Risk Assessment**: Subtle augmentation bugs (wrong padding mode, normalization mismatch with the frozen eval transform) silently hurt accuracy and are hard to spot. Throughput gain without a budget-matched schedule wastes the extra steps (baseline schedule would just truncate differently). Best paired with Idea 1's schedule — as a follow-up loop, so gains are attributable.

### 3. Short-budget architecture swap (ResNet-9-style wide net or widened ResNet-20)
**Summary**: Replace ResNet-20 (deep, narrow: 16/32/64 channels) with a shallow, wide network proven optimal for short budgets — e.g., Page's ResNet-9 (64→128→256→512 with maxpool and two residual blocks) or a 4×-widened ResNet-20. VRAM is a soft constraint and 98GB is available; wider nets extract more accuracy per wall-clock second on modern GPUs because they parallelize better than deep narrow stacks.

**Reasoning**: Every CIFAR-10 speedrun result (94% in seconds) uses shallow-wide architectures, not ResNet-20. Depth costs serial time; width is nearly free on an H20. This is the largest expected accuracy ceiling of the three ideas (94%+ territory).

**Sources**: cifar10-fast ResNet-9, arXiv 2404.00498 network description.

**Estimated Effort**: medium (rewrite model class; retune LR/schedule for the new architecture simultaneously)

**Risk Assessment**: Changes architecture AND requires recipe retuning in one experiment — confounds attribution and raises variance of the first attempt. An untuned wide net under the legacy schedule can easily underperform 91.97%. Better attempted after the recipe foundation (Idea 1) is in place so a tuned schedule carries over.

## Idea Evaluation

**Evidence strength**: All three draw on the same strong public results, but Idea 1 has the most direct evidence for its *exact* failure-mode fix: the baseline demonstrably truncates its LR schedule (37.5k of 64k steps), and super-convergence/cifar10-fast show budget-matched one-cycle schedules are the correct fix. Ideas 2 and 3 have strong evidence too, but their gains are contingent on a budget-matched schedule already existing — extra throughput or capacity is wasted if the anneal still truncates.

**Mechanism clarity**: Idea 1 — two crisp mechanisms: (a) completing the LR anneal within the budget recovers the accuracy the truncated schedule leaves behind; (b) bf16/channels_last/batch-512 raise img/s, fitting more epochs into 300s. Idea 2 — clear mechanism (fewer host-side stalls) but improves the *input* to training, not training itself. Idea 3 — clear ceiling argument but entangled with recipe retuning.

**Expected impact**: Idea 3 has the highest ceiling (94%+), Idea 1 a solid expected +0.5–1.5pp, Idea 2 alone perhaps +0.3–0.8pp (throughput→epochs). But impact must be weighted by probability of landing on the first attempt: Idea 1 is far more likely to clear the +0.1pp bar in one shot.

**Risk profile**: Idea 1 fails gracefully (worst case mild regression). Idea 2 risks silent correctness bugs. Idea 3 risks an untuned-recipe regression and confounded attribution.

**Feasibility / sequencing**: Idea 1 is the natural foundation: it establishes the schedule and precision groundwork that Ideas 2 and 3 need anyway, and keeps the experiment ladder clean (recipe → pipeline → architecture), so each later loop isolates its own contribution.

## Chosen Idea
**Selected**: Idea 1 — Budget-matched modern training recipe (same ResNet-20)

**Why this idea**:
It directly fixes the one demonstrated defect in the baseline (an LR schedule that never completes within the 300s budget), is backed by the strongest and most setting-specific evidence (super-convergence, cifar10-fast), has the safest failure mode, and lays the recipe foundation that the higher-ceiling pipeline/architecture ideas require. Highest probability of clearing the ≥+0.1pp bar on the first loop while building toward larger gains.

**Hypothesis**:
Replacing the truncated MultiStepLR with a budget-matched one-cycle schedule (keyed to elapsed-budget-fraction so the anneal always completes), combined with bf16 autocast + TF32 + channels_last + batch 512 (linearly scaled peak LR, Nesterov, no WD on BN/bias, label smoothing 0.1), will raise best_test_acc from the 91.97% baseline to ≥92.5% — because (a) the LR anneal completing recovers accuracy lost to schedule truncation, and (b) higher throughput fits ≥1.5× more epochs into the same 300s training budget.
