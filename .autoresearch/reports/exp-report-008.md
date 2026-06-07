# Report EXP-008: k=4 + Stochastic Depth + EMA
- **Created**: 2026-05-28

## Results
- **Primary metric**: 93.22% (baseline: 95.73%, delta: -2.51%)
- **Key Learning**: Stochastic depth is unsuitable for shallow 9-block ResNet-20. The technique was designed for 110+ layer networks where block redundancy absorbs drops. At 3 blocks per group, dropping any block removes too much capacity.
- **Verdict**: no-improvement

## Exit Action Results
