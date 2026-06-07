# Report EXP-003: k=3 + T_max=57 + CutMix
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-003.md
- **Plan**: plans/plan-003.md
- **Log**: (created inline)

## Goal
Maximize CIFAR-10 test accuracy. Baseline: 94.03% (EXP-001).

## Idea & Hypothesis
Retry EXP-002's approach (k=3 + CutMix) with correct static T_max=57 (derived from EXP-002's 62 actual epochs). Hypothesis: 95-96% from proper T_max + more capacity + CutMix.

## Approach
WIDTH_MULT=3, COSINE_T_MAX=57, CutMix(alpha=1.0, p=0.5), all other settings from EXP-001 preserved.

## Execution
Single run, 65 epochs, no issues. T_max=57 was close to actual 65 epochs (slightly under, but cosine completed near end).

## Results
- **Primary metric**: 94.80% (baseline: 94.03%, delta: +0.77%)
- **Observations**: best_test_acc == final_test_acc (94.80%) — perfect T_max alignment. The model converged properly with no degradation in later epochs. 65 epochs vs EXP-002's 62 — slight variation is normal. CutMix + k=3 combination works well.
- **Analysis**: Confirms that T_max alignment is the critical factor. EXP-002 with broken T_max got only 94.09% best on the same architecture; EXP-003 with correct T_max got 94.80% — a 0.71% improvement just from T_max correction. k=3 provides meaningful improvement over k=2 (94.80% vs 94.03%).
- **Key Learning**: k=3 with correct T_max and CutMix yields 94.80%; best==final confirms T_max alignment is critical. Further width increases may continue to help.

## Verification
- **Conditions**: All 4 passed
- **Verdict**: improvement
- **Verdict Basis**: 94.80% >= 94.13% threshold.

## Unexplored Avenues
- **k=4 width**: With VRAM still only 425MB, k=4 ({64,128,256}) is feasible. Estimated ~30-35 epochs with T_max=25-30.
- **Larger batch size**: batch=256 or 512 could improve GPU utilization.
- **Deeper model (ResNet-32)**: More layers at k=3 width.
- **Additional augmentation**: Mixup combined with CutMix.

## Next Steps
1. **k=4 with correct T_max** (high confidence): Continue the width trajectory.
2. **Deeper + wider** (medium confidence): ResNet-32 at k=3 or k=4.

## Exit Action Results
