# Brainstorm EXP-006
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new search this loop — sources already in the knowledge base cover the space:
- **PyTorch torch.compile docs** (standard knowledge; mode="default"/inductor): kernel fusion on conv/BN/ReLU stacks typically yields 1.1–1.5x on small CNNs; recompilation is triggered per (train/eval mode, input shape) — `drop_last=True` keeps the train shape constant at (512,3,32,32), and the frozen Eval uses a fixed batch 256, so exactly two compilations are expected.
- **cifar10-airbench (knowledge/README.md, arXiv 2404.00498)**: speedrun-class training relies on eliminating per-step framework overhead — compile/fusion is one of its standard levers at this model scale.
- **project-insights.md § High**: H20 channel-alignment rule (keep multiples of 32/64) — constrains all width candidates; 4x (64/128/256) is optimally aligned.

## Experimental History Review

- **Trajectory**: 91.97 → 93.16 → 95.23 (4x) → 94.41 (8x, fail) → 96.06 (RE) → 96.23 (TA) → 95.12 (5x, fail). Baseline: 96.23 @ 1174e0d.
- **Failed Approaches (count 2, Medium)**: capacity without throughput — 8x@40ep (−0.82), 5x@52ep (−1.11). Below ~70 epochs starvation dominates width gains even with regularizers. Width is GATED on a throughput unlock.
- **Patterns**: time-keyed one-cycle survives any throughput change (validated 3x) — torch.compile composes with it for free; augmentation returns exhausted at 4x (RE +0.83 → TA +0.17); 4x widths are tensor-core aligned.
- **Protocol Findings**: eval overhead ~1s/epoch counts toward the 600s wall cap; more epochs from a throughput gain → more evals → wall-clock pressure (EXP-000 finished at 596.7s — the cautionary datapoint).
- **Untried gaps**: torch.compile, larger batch (1024), Mixup/CutMix (flagged sub-noise), wider-but-shallower topology, GPU-resident data (deprioritized — GPU-compute-bound at 4x).
- **Critical implementation fact (from train.py)**: the training loop bills each step's wall time into the 300s budget — a naive `torch.compile` would charge ~60–90s of one-time compilation to step 1, wasting ~25% of the budget. Compile warmup MUST happen before the timed loop (forward+backward on a dummy batch, no optimizer.step, grads zeroed) so the cost lands in startup, which is excluded from training_seconds by construction.

## Candidate Ideas

### 1. torch.compile with pre-loop warmup (throughput unlock at 4x)
**Summary**: Wrap the model with `torch.compile` (default/inductor mode) after `.to(device)`. Before the timed loop, run a warmup: 2–3 forward+backward passes on a synthetic batch of the exact training shape/dtype/mode (channels_last, bf16 autocast, model.train()), then `optimizer.zero_grad(set_to_none=True)` — weights untouched since `optimizer.step()` is never called. Everything else unchanged.

**Reasoning**: Width — the biggest remaining accuracy lever — is gated on img/s (two failures, count-2 learning). Compile is the only untried throughput lever that doesn't touch the (frozen) data pipeline and composes with the time-keyed schedule automatically. At 4x the standalone gain is modest (114 → ~135–160 epochs; under heavy augmentation extra epochs still convert weakly but the +0.1 bar is low), and even a bar-miss yields the decisive img/s datapoint that determines whether 6x/8x@aligned-widths becomes viable. Warmup placement keeps compile cost out of the 300s budget; startup_seconds is already separately reported, making the accounting verifiable from the log.

**Sources**: PyTorch compile docs; goal-learnings § Failed Approaches (gating), § Patterns (schedule composability); train.py timing structure; exp-report-005 § Next Steps (medium-high as enabler).

**Estimated Effort**: low (one wrap + ~6-line warmup block)

**Risk Assessment**: (a) inductor crash/graph break → clean failure, possibly fixable in-run via mode fallback; (b) eval-path recompile on first evaluate() adds one-time ~20–40s wall clock; (c) more epochs → more evals → wall clock ~up to 550s, still under 600 but monitor; (d) speedup could be ~1.0x on this already-lean model → no-improvement with full information value.

### 2. Batch 1024 + PEAK_LR 0.8 (utilization + schedule at constant FLOPs)
**Summary**: Double BATCH_SIZE and PEAK_LR (linear scaling). Halves per-step Python/launch overhead; img/s may rise ~5–15% at the GPU-bound regime.

**Reasoning**: Cheap throughput nibble, but the model is GPU-compute-bound at 4x (~19k img/s, dt ~27ms — launch overhead is a small fraction), so the expected epoch gain is small; doubling peak LR to 0.8 adds real optimization-stability risk (loss spikes at warmup peak), and large-batch generalization at fixed time budget is uncertain. Two coupled variables — weaker experiment design than Idea 1.

**Sources**: goal-learnings § Patterns (bf16/batch-512 lever exhausted); linear-scaling rule (Goyal et al. 2017, standard knowledge).

**Estimated Effort**: trivial

**Risk Assessment**: Instability tail at LR 0.8; confounded two-variable change; expected gain likely < bar.

### 3. Mixup or CutMix (third regularizer class)
**Summary**: Batch-level label-mixing augmentation in the training step (loss becomes a convex combination).

**Reasoning**: Exp-report-004 measured steeply diminishing augmentation returns (0.83 → 0.17pp); a third regularizer at 4x capacity is predicted sub-noise. Also interacts with label smoothing and requires loss-path changes — more surface for subtle bugs than its expected value justifies. Kept as a future option after a successful capacity step.

**Sources**: exp-report-004 § Unexplored Avenues; arXiv 1710.09412 (Mixup, standard knowledge).

**Estimated Effort**: low-medium

**Risk Assessment**: Expected gain below the +0.1 bar at current capacity; failure uninformative (already predicted).

## Idea Evaluation

**Evidence strength**: Idea 1 rests on a well-documented systems technique plus the count-2 in-project learning that throughput is THE gate; its information value is guaranteed even on a metric miss. Idea 2 contradicts the in-project finding that the launch-overhead lever is mostly exhausted. Idea 3 contradicts the measured diminishing-returns trend.

**Mechanism clarity**: Idea 1 — fusion cuts per-step GPU time → more epochs in 300s → better convergence under augmentation, plus enables width next. Idea 2 — marginal overhead reduction, confounded. Idea 3 — regularization where regularization is exhausted.

**Expected impact**: Idea 1: +0.1–0.3pp now, plus unlocks the width direction (where +0.5pp-class gains may remain). Idea 2: ±0.2pp. Idea 3: <+0.1pp.

**Risk profile**: Idea 1's failure modes are clean and diagnosable from the log (startup_seconds, dt, num_epochs). Idea 2 has an instability tail. Idea 3 wastes a loop on a predicted-negative.

**Feasibility**: all low effort; Idea 1 needs the warmup subtlety handled (identified above, solution in hand).

## Chosen Idea
**Selected**: Idea 1 — torch.compile with pre-loop warmup

**Why this idea**:
It attacks the demonstrated gate (throughput) with the only untried lever that respects the frozen data pipeline, composes automatically with the validated time-keyed schedule, and pays out twice: a possible direct gain now, and the img/s datapoint that decides whether aligned-width 6x/8x is viable next. The budget-accounting trap (compile cost billed to step 1) is identified and neutralized by design.

**Hypothesis**:
Compiling the 4x model with a pre-loop warmup will raise steps/s by ≥15% (num_epochs 114 → ≥130) at unchanged accuracy-per-epoch dynamics, lifting best_test_acc from 96.23% to ≥96.33% (the +0.1 bar) via a longer effective schedule under the augmented recipe; startup_seconds will absorb the compile cost (expect 60–120s) and total wall clock stays ≤ 580s. If throughput does NOT improve ≥10%, the run is expected to be a no-improvement and conclusively deprioritizes the compile/width path in favor of recipe-level ideas.
