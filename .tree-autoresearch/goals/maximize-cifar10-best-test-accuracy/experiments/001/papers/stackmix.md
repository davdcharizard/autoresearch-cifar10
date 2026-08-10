# StackMix: A Complementary Mix Algorithm
- **Authors**: John Chen, Samarth Sinha, Anastasios Kyrillidis
- **Venue**: UAI 2022 (PMLR 180)
- **URL**: https://proceedings.mlr.press/v180/chen22b.html

## Key Contributions
- Concatenates two inputs and averages their labels, creating a complementary mixed-example augmentation.
- Reports that combining StackMix with CutMix improves CIFAR-10 accuracy by 0.5 percentage points over CutMix in its evaluated setup.

## Relevance
The result supports combining orthogonal mixed-sample augmentations, but spatial concatenation changes model input geometry and may reduce throughput. It is better treated as a later exploratory branch than the first high-confidence experiment.

## Key Techniques
- Input concatenation with averaged targets.
- Combination with CutMix or Mixup.
