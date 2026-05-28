# Report EXP-009: Batch Size 256 with Linear LR Scaling
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-009.md
- **Plan**: plans/plan-009.md
- **Log**: logs/exp-log-009.md

## Goal

Maximize CIFAR-10 test accuracy (best_test_acc, %). Higher is better. Baseline entering this experiment: 94.82% (EXP-007, commit 1c37a9f). Verification threshold: best_test_acc > 94.92% (baseline + 0.1pp).

## Idea & Hypothesis

Double batch size from 128 to 256 with LR scaled from 0.1 to 0.2 (linear scaling rule, Goyal et al. 2017) and a 5-epoch gradual warmup. Throughput is the binding constraint — every successful accuracy improvement in this project's history has been driven by either more capacity (width) or more throughput (AMP). At 484 MB VRAM out of ~96 GB H20 capacity, the GPU is massively underutilized. Hypothesis: 2x batch size → ~30-50% throughput increase → ~108-120 epochs in 300s → accuracy above 94.92%.

## Approach

Three changes to `train.py`:
1. `BATCH_SIZE = 128 → 256`, `LR = 0.1 → 0.2` (linear scaling rule)
2. Added warmup mechanism: `WARMUP_EPOCHS = 5`, `_epoch_count` mutable cell, warmup multiplier `(e+1)/5` applied to the existing wall-clock-fractional step-decay schedule during the first 5 epochs, ramping LR from 0.04 to 0.2
3. `_epoch_count[0] = epoch - 1` after `epoch += 1` in the training loop to drive warmup state

No deviations from plan. The existing `_lr_progress` mutable cell pattern made the warmup addition clean.

## Execution

Single run, local execution on H20 GPU. Training started cleanly at ~16ms/step, ~16,300-16,500 img/s throughput. Warmup executed correctly over 5 epochs. Completed 98 epochs in 300.0s (vs ~83 epochs at batch 128 — 18% more epochs). LR cascade: 0.2 → 0.02 at ~150s → 0.002 at ~225s. Peak VRAM 864.6 MB. No errors or retries.

## Results

- **Primary metric**: 95.39% (baseline: 94.82%, delta: +0.57pp, +0.60%)
- **Observations**: Throughput increase was ~18% more epochs (98 vs 83), at the lower end of the predicted 30-50% range — per-step time increased from 9ms to 16ms, partially offsetting the batch size doubling. Accuracy trajectory showed the familiar pattern: warmup → plateau ~84% → first LR drop → jump to ~93% → plateau → second LR drop → jump to ~95.4%. The warmup phase was visible in the first 5 epochs with LR ramping from 0.04 to 0.2. Best accuracy 95.39% at epoch 96, final 95.29% at epoch 98.
- **Analysis**: The hypothesis was directionally correct — more throughput led to more accuracy. However, the throughput gain was smaller than predicted (18% vs 30-50%) because per-step time nearly doubled (9ms → 16ms), suggesting the GPU was not as underutilized at batch 128 as the VRAM metric implied. The accuracy gain (+0.57pp) exceeded the predicted +0.3-0.5pp, suggesting diminishing returns are not yet severe at this epoch count. The linear scaling rule + warmup worked as expected with no instability.
- **Key Learning**: Batch size scaling on H20 yields sublinear throughput gains (~1.18x for 2x batch) due to compute overhead beyond VRAM, but the throughput-to-accuracy conversion rate remains strong — each additional epoch still contributes meaningfully to accuracy.

## Verification

- **Conditions**: All 3 passed — (1) best_test_acc 95.39% > 94.92% threshold, (2) all 10 summary fields present, (3) eval count 98 = num_epochs 98
- **Review Notes**: Results confirmed trustworthy. Metrics consistent across log observations and final summary. No signs of stale output or evaluation artifacts.
- **Verdict**: improvement
- **Verdict Basis**: All verification conditions passed, primary metric improved by +0.57pp (well above the 0.1pp threshold)

## Unexplored Avenues

- **Batch size 512 with LR 0.4**: The next step on the batch-scaling ladder. VRAM headroom is still massive (864 MB / ~96 GB). However, per-step time scaling suggests diminishing returns — throughput gain may be only ~10-15% more epochs. LR=0.4 with FP16 may need longer warmup or gradient clipping for stability.
- **Cosine annealing LR (with correct T_max)**: Now that epoch count is ~98, a cosine schedule with T_max=98 could extract more from the continuous decay vs the 3-stage step schedule. The step schedule wastes the first 50% of budget at a fixed high LR; cosine starts decaying immediately.
- **OneCycleLR / triangular policy**: Referenced in the fast CIFAR-10 literature (David Page, Smith & Topin). The warm-up + peak + cooldown shape may be better suited to the fixed time budget than step decay. Complementary to larger batch sizes.

## Next Steps

1. **Batch size 512 with LR 0.4** (medium confidence) — Push the batch scaling further. Expected sublinear throughput gain but still potentially +0.2-0.3pp. Risk: FP16 instability at LR=0.4.
2. **CutMix batch-level augmentation** (medium confidence) — With 98 epochs, there is more training budget to exploit CutMix's regularization benefit than at 83 epochs. The over-regularization risk from EXP-004 is somewhat reduced.
3. **Cosine or OneCycle LR schedule** (medium confidence) — Replace the step-decay schedule. Cosine with correct T_max avoids the plateau periods. OneCycle has strong literature support for fast CIFAR-10 training at larger batch sizes.

## Exit Action Results

- **PR creation failed**: `gh pr create` returned "Resource not accessible by personal access token (createPullRequest)". Branch `autoresearch/exp-009` pushed to remote — user may create PR manually from `autoresearch/exp-009` → `main`.
