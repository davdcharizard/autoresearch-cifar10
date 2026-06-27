# Brainstorm EXP-003
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new search — knowledge base and prior citations cover the space:
- **Random Erasing (arXiv 1708.04896) / Cutout (arXiv 1708.04552)** (cited in exp-report-001/002 Unexplored Avenues): +0.4–1.0pp on WRN-class CIFAR-10 models; Random Erasing's best CIFAR config is p=0.5, area scale (0.02, 0.4), random fill values. Available as `transforms.RandomErasing` in torchvision (already a dependency; applies on tensors after Normalize — negligible CPU cost).
- **WRN (arXiv 1605.07146)** (knowledge/README.md): width gains at this depth continue past 4× *given enough training* — EXP-002 showed the budget doesn't provide it at current throughput.

## Experimental History Review

- **Trajectory**: 91.97 → 93.16 (EXP-000 recipe) → 95.23 (EXP-001 4× width) → 94.41 (EXP-002 8× width, no-improvement). Baseline: 95.23 @ bd0976e.
- **Failed Approaches** (goal-learnings): capacity increase without throughput increase — 8× starved at 40 epochs, final=best is the undertraining signature. Width 5–6× or throughput work might still help; 8× at current pipeline is dead.
- **Patterns**: time-keyed one-cycle composable (validated 2×); bf16/TF32/channels_last/batch-512 exhausted as a lever; 4× width = 114 epochs at 18.7k img/s, total 395.8s (wall-clock comfortable).
- **New first-principles observation (this loop)**: at 4× width the run is likely GPU-compute-bound, not host-bound — 18.7k img/s ≈ what H20 bf16 throughput predicts for a ~2 GFLOPs/image train step. So the GPU-resident-pipeline idea from exp-report-002 would mostly help at LOW widths (1–2×), not at 4×+; it drops in priority for moving the metric at current width.
- **Untried gaps**: regularization (RandomErasing/Cutout — capacity now exists to exploit it), intermediate width 5–6×, torch.compile throughput, ResNet-9 topology, larger batch.

## Candidate Ideas

### 1. RandomErasing on the 4× net (Cutout-class regularization)
**Summary**: Append `transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value="random")` to the train transform after Normalize. Architecture, recipe, throughput all unchanged (tensor-op, ~no per-image cost).

**Reasoning**: EXP-001's signature — final_test_loss 0.2447 with training loss far lower — shows a real generalization gap at 4.29M params and 114 epochs: capacity now exists for stronger regularization to convert into test accuracy. Random Erasing's published gains on WRN-class CIFAR-10 models are +0.4–1.0pp at full schedules; at 114 one-cycle epochs expect the lower end but still above the +0.1 bar. Orthogonal to the capacity/throughput axis that EXP-002 just probed, so the loop diversifies its information.

**Sources**: arXiv 1708.04896 (Random Erasing, CIFAR config), arXiv 1708.04552 (Cutout); exp-report-001 § Unexplored Avenues; torchvision RandomErasing (in pyproject deps).

**Estimated Effort**: trivial (one transform line)

**Risk Assessment**: Occlusion regularization slows fitting — at "only" 114 epochs the gain may shrink toward noise; worst case mild regression → clean no-improvement. No stability or wall-clock risk whatsoever.

### 2. Intermediate width 6× (stage widths 96/192/384, ~9.6M params)
**Summary**: WIDTH_MULT 4 → 6, recipe unchanged. Targets the width-epoch optimum bracketed by EXP-001 (4×, +2.07) and EXP-002 (8×, −0.82).

**Reasoning**: The unimodal accuracy(width) curve must peak in (4, 8); 6× would get ~60–75 epochs (vs 114 at 4×, 40 at 8×). But interpolation suggests the peak may be close to 4–5×: 8× lost 0.82pp, so the curve falls steeply right of the optimum, and the marginal gain at 6× could be within noise.

**Sources**: experiment-indices rows 001/002; arXiv 1605.07146.

**Estimated Effort**: trivial

**Risk Assessment**: Decent chance of landing within ±0.2pp of 95.23 — a coin-flip against the +0.1 bar; burns a loop on a point estimate of a possibly-flat curve region.

### 3. torch.compile for throughput at 4× (more epochs)
**Summary**: `model = torch.compile(model)` — kernel fusion typically gives 1.2–1.8× on small CNNs; compile time (~1–2 min) lands in startup/epoch-1, excluded from the training budget but added to total wall clock (~395 → ~500s, still under cap).

**Reasoning**: More epochs at fixed width attacks EXP-002's root cause from the other side. But 4× at 114 epochs is already well-trained (one-cycle converged, final≈best), so extra epochs convert weakly; the bigger payoff would be re-enabling 6–8× width later.

**Sources**: PyTorch torch.compile docs; goal-learnings § Patterns (throughput lever status).

**Estimated Effort**: low, but failure modes are fiddly (graph breaks, recompiles per epoch boundary, autocast interactions)

**Risk Assessment**: Compile instability or first-epoch timing distortions could waste the run; accuracy mechanism is indirect (epochs already sufficient at 4×). Better as an enabler experiment before a renewed width push.

## Idea Evaluation

**Evidence strength**: Idea 1 has direct published gains in exactly this model class (WRN-on-CIFAR) plus an observed in-project overfitting signature to exploit. Idea 2's evidence is an interpolation argument with a real chance the peak sits near 4× already. Idea 3's evidence is about speed, not accuracy — its accuracy mechanism is weak at the current operating point.

**Mechanism clarity**: Idea 1 — clear: capacity exceeds effective data complexity at 114 epochs; occlusion augmentation raises effective data complexity, converting the train/test gap into test accuracy. Idea 2 — clear but quantitatively uncertain. Idea 3 — indirect (throughput → epochs → accuracy, with the last link weak at 4×).

**Expected impact**: Idea 1: +0.3–0.7pp expected. Idea 2: −0.1 to +0.4pp. Idea 3: +0.0–0.3pp now (more later as enabler).

**Risk profile**: Idea 1 safest (no stability/wall-clock risk). Idea 2 safe but information-poor if it lands in noise. Idea 3 has real crash/instability tail.

**Feasibility**: all trivial-to-low effort; Idea 1 is one line.

## Chosen Idea
**Selected**: Idea 1 — RandomErasing on the 4× net

**Why this idea**:
Best evidence-to-risk ratio of the three: published +0.4–1.0pp in this exact model class, an observed in-project generalization gap to exploit, one line of code, zero stability or wall-clock risk, and orthogonal to the capacity axis where the loop just took a failure — keeping information gain high regardless of outcome.

**Hypothesis**:
Adding RandomErasing (p=0.5, scale (0.02, 0.4), random fill) to the train transform of the 4×-wide net will raise best_test_acc from 95.23% to ≥95.5%, because the model's train/test gap at 4.29M params and ~114 epochs indicates unconverted capacity that occlusion-based augmentation converts into generalization; throughput and epoch count will be essentially unchanged.
