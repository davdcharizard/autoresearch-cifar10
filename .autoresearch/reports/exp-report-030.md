# Report EXP-030: RandomCrop padding 6
- **Created**: 2026-05-29

## Results
- **Primary metric**: 94.29% (baseline: 96.39%, delta: -2.10%)
- **Key Learning**: Padding 6 is far too aggressive; larger zero-padded regions make the task much harder and fewer epochs (47 vs 54).

## Verification
- **Verdict**: no-improvement

## Exit Action Results
