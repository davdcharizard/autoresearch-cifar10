# Brainstorm EXP-026
**Created**: 2026-05-29
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- No new search. Width scaling is the most validated pattern in this project's history.

## Experimental History Review

- **27 experiments**, baseline 96.39%, ten consecutive failures
- **Width scaling trajectory**: k=1→k=2 (+1.93%), k=2→k=3 (+0.77%), k=3→k=4 (+0.45%). Diminishing returns but still positive each step.
- **k=6 failed** (EXP-005, 94.52%): 9.7M params, only 32 epochs — insufficient convergence
- **k=5 NEVER TRIED**: ~6.7M params, should get ~42-45 epochs. Key difference from k=6: (1) 50% fewer excess params, (2) ~10 more epochs for convergence, (3) the full modern recipe (EMA, CutMix, TTA) was added AFTER k=6 was tested
- **Critical**: T_max must match actual epochs. Need to estimate k=5 epoch count and set T_max accordingly.
- **Estimation**: k=4 gets ~54 epochs (4.3M). k=5 has 1.56x FLOPs → ~35 epochs? Or ~42 with AMP efficiency? Must calibrate T_max carefully.

## Candidate Ideas

### 1. Width k=5 with Calibrated T_max
**Summary**: Change WIDTH_MULT from 4 to 5, giving channels [80, 160, 320] and ~6.7M parameters. Set COSINE_T_MAX to match expected cosine epochs. Run a 2-epoch calibration first to estimate per-epoch time, then set T_max before the full run. The increased model capacity may yield higher accuracy if enough epochs are available.

**Reasoning**: Width scaling has been the most reliable improvement lever throughout the project. Each k increment has improved accuracy: k=2 (+1.93%), k=3 (+0.77%), k=4 (+0.45%). k=5 was skipped — we jumped from k=4 to k=6 and k=6 failed due to insufficient epochs. k=5 is the sweet spot between more capacity and sufficient training time. The modern recipe (EMA, CutMix, label smoothing, TTA) was NOT present when k=6 was tested — these techniques help larger models converge faster. T_max will be set dynamically: measure epoch 2 timing, estimate total epochs, set T_max = estimated_epochs - warmup.

**Sources**: EXP-001 through EXP-005 (width scaling trajectory), goal-learnings patterns

**Estimated Effort**: medium — WIDTH_MULT change + T_max calibration logic

**Risk Assessment**: Medium. If k=5 only gets ~35 epochs, it may underconverge like k=6. But the modern recipe should help. T_max calibration is critical — the protocol finding says mismatch causes severe regression.

## Chosen Idea
**Selected**: Width k=5 with Calibrated T_max

**Why this idea**: Width scaling is the only proven trajectory in this project. k=5 is untested. The modern recipe might make it viable where k=6 wasn't.

**Hypothesis**: k=5 (6.7M params) with properly calibrated T_max will improve best_test_acc above 96.49% by leveraging additional model capacity.
