# Report EXP-032: RandomCrop reflect padding
- **Created**: 2026-06-04

## Results
- **Primary metric**: 94.77% (baseline: 96.39%, delta: -1.62%)
- **Key Learning**: Reflect padding hurts significantly; the model and hyperparameters are tuned for zero-padded crop distribution. Only 47 epochs (reflect padding may be slower through torch.compile).

## Verification
- **Verdict**: no-improvement

## Exit Action Results
