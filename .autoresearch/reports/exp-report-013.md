# Report EXP-013
## Results
- best: 94.79% (baseline: 95.73%, delta: -0.94%)
- 91 epochs, 7.68M params. Custom VGG-style ConvNet.
- T_max alignment good (best/final gap 0.06%)
- **Verdict**: no-improvement
- **Key Learning**: Without airbench's whitening layer and 1x1 expansion convs, a generic VGG-style net is less efficient than well-tuned ResNet-k4. The architecture details matter more than the family.
