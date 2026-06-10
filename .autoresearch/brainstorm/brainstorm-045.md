# Brainstorm EXP-045
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Existing knowledge: PyTorch EMA averaging** (`.autoresearch/knowledge/references/pytorch-ema-averaging.md`)
  PyTorch's `AveragedModel` supports EMA-style model averaging without changing the optimizer or harness. Prior project use shows the implementation must avoid high update overhead and BatchNorm integer-buffer averaging pitfalls.
- **Existing knowledge: SGDR cosine scheduling** (`.autoresearch/knowledge/papers/sgdr-cosine-schedule.md`)
  A no-restart cosine decay is still an available schedule alternative to abrupt drops, but local step-drop evidence remains strong.
- **Existing knowledge: Wide Residual Networks** (`.autoresearch/knowledge/papers/wide-residual-networks.md`)
  Compact wider residual variants can improve CIFAR accuracy in principle, but this repo has already found width beyond 28/56/112 fragile under the fixed 300s budget.

No new external sources were consulted for this brainstorm; the existing knowledge base already contains the relevant EMA, schedule, and architecture references for the current decision.

## Experimental History Review

- Current baseline is `best_test_acc=93.97%` from EXP-038 / commit `755be2c`; the active goal requires `best_test_acc >= 94.07%`.
- The current anchor is `STAGE_WIDTHS=(28,56,112)`, reflected `RandomCrop`, `label_smoothing=0.05`, `WEIGHT_DECAY=2e-4`, `LR=0.1`, `MOMENTUM=0.9`, and `LR_MILESTONES=[21000,64000]`.
- Local scalar optimizer space is now low value: `WEIGHT_DECAY=2e-4` is bracketed by worse 1.5e-4 and 3e-4 probes, and `LR=0.1` is bracketed by worse 0.08 and 0.12 probes.
- Isolated augmentation is mixed. Reflection padding and label smoothing were successful, but cutout, mixup run-control, and mild RandAugment did not improve the current anchor.
- Late accuracy drift remains common: EXP-038 peaked at 93.97% but ended at 93.54%, EXP-043 peaked at 93.49% but ended at 93.19%, and EXP-044 peaked at 93.83% but ended at 92.80%.
- Prior averaging failures are specific, not a blanket rejection of late EMA. EXP-004 updated EMA every step for the full run and lost about 6.8k steps; EXP-021 used long equal post-drop averaging and collapsed, with a first attempt also exposing integer BatchNorm buffer averaging as unsafe.
- The cleanest open gap is a bounded late-stability method that avoids EXP-004's full-run per-step overhead and EXP-021's long equal-average collapse.

## Candidate Ideas

### 1. Sparse Late EMA Evaluation After First LR Drop
**Summary**: Maintain an EMA copy only after the first LR drop, update it sparsely every 100 optimizer steps, and evaluate the EMA model once per epoch after activation.

**Reasoning**: The current anchor repeatedly shows a peak-to-final gap, implying late SGD noise or checkpoint instability. EXP-004 and EXP-021 do not rule out EMA itself: EXP-004 paid full-run per-step overhead, while EXP-021 used equal averaging over an increasingly long post-drop trajectory. A sparse late EMA should preserve most step throughput, use a short effective averaging window, and target the exact post-drop phase that produces the current best accuracies. Using `use_buffers=False` avoids the BatchNorm integer-buffer crash discovered in EXP-021.

**Sources**: `.autoresearch/knowledge/references/pytorch-ema-averaging.md`; reports/exp-report-004.md; reports/exp-report-021.md; late drift in reports/exp-report-038.md, reports/exp-report-043.md, and reports/exp-report-044.md.

**Estimated Effort**: medium

**Risk Assessment**: EMA can still underperform if copied BatchNorm buffers mismatch averaged weights, and evaluating EMA instead of the raw model could miss raw peaks after activation. The failure mode should be a valid no-improvement rather than invalid if validation cadence remains once per epoch and only `train.py` changes.

### 2. No-Restart Cosine Schedule on the Final Anchor
**Summary**: Replace `MultiStepLR([21000,64000])` with a no-restart cosine schedule over the reachable step horizon while preserving the final regularized anchor.

**Reasoning**: Cosine annealing can smooth the abrupt first-drop transition and might improve anytime accuracy. The final `2e-4` anchor has not received a clean cosine-only experiment. However, many schedule-only variants around prior anchors underperformed, and the current 21k first drop consistently creates the main accuracy jump. This makes cosine clean and informative but less locally supported than late EMA.

**Sources**: `.autoresearch/knowledge/papers/sgdr-cosine-schedule.md`; failed schedule-only entries in `.autoresearch/goal-learnings/maximize-cifar10-best-test-accuracy.md`; EXP-016 and EXP-038 reports.

**Estimated Effort**: medium

**Risk Assessment**: A full schedule swap may spend too little time in the proven low-LR refinement regime or may decay too slowly for the fixed budget. It is unlikely to violate constraints, but expected impact is uncertain.

### 3. Compact WRN-Style ResNet-16 Capacity Rebalance
**Summary**: Test a shallower, wider residual network variant that keeps the CIFAR residual family but changes depth/width balance under the fixed budget.

**Reasoning**: WRN literature supports shallower/wider residual networks on CIFAR, and this repo's current wider ResNet-20 anchor is the best architecture so far. A ResNet-16-style variant could trade depth for width and improve throughput or optimization. But local history strongly warns against additional width changes above the 28/56/112 anchor, and capacity experiments are more disruptive than a late-stability probe.

**Sources**: `.autoresearch/knowledge/papers/wide-residual-networks.md`; High Importance goal-learning entry on failed widening beyond 28/56/112; EXP-014, EXP-017, EXP-019, and EXP-020 reports.

**Estimated Effort**: high

**Risk Assessment**: Architecture changes can alter parameter count, runtime, and schedule calibration all at once. It may be worth revisiting later, but it has a higher invalid/crash/regression risk than an optimizer-adjacent late-stability experiment.

## Idea Evaluation

Sparse late EMA has the best combination of mechanism clarity and open experimental gap. It directly targets the observed late drift without changing the dataset, schedule, architecture, optimizer, or evaluation harness. It also differs materially from both failed averaging experiments: updates begin only after the first LR drop, happen sparsely rather than every step, use EMA rather than equal averaging, and avoid unsafe integer buffer averaging.

Cosine scheduling is a clean future test, but the local schedule evidence is weak. The 21k step drop is one of the best-supported pieces of the current recipe, and several schedule-only changes have failed. A cosine run may still be useful for map coverage, but it is not the strongest next move after scalar and augmentation probes.

The WRN-style capacity rebalance has theoretical upside but conflicts with the strongest local failure pattern: widening beyond the current anchor has repeatedly failed under this budget. It should wait until lower-risk recipe and late-stability experiments are exhausted.

## Chosen Idea
**Selected**: Sparse Late EMA Evaluation After First LR Drop

**Why this idea**:
It attacks the repeated peak-to-final drift through a bounded late-stability mechanism while preserving the validated anchor. The design explicitly avoids the known EMA failure modes: full-run per-step overhead, long equal averaging, and BatchNorm integer-buffer averaging.

**Hypothesis**:
Updating a late EMA copy every 100 optimizer steps after step 21000 and evaluating that EMA model once per epoch will smooth post-drop weights enough to reach `best_test_acc >= 94.07%` without violating the fixed harness, single-GPU, `train.py`-only, or validation-cadence constraints.
