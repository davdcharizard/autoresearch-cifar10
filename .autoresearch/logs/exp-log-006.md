# EXP-006: k=4 + TrivialAugment + CutMix

## Execution
- **Created**: 2026-05-28
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-006
- **Outcome**: failed

## Run Log
### Run 1
- **Status**: completed
- best_test_acc: 95.15%, final: 94.76%, 64 epochs, 4.3M params

## Verification Results
95.15% < 95.35% — FAIL

## Errors & Dead Ends
### 2026-05-28 — TrivialAugment + CutMix too aggressive
- Error: 95.15% < 95.25% baseline
- Root cause: Combined augmentation too heavy for ~60 epoch budget
- Do NOT retry: avoid stacking heavy augmentation techniques without increasing training time

## Human Notes
