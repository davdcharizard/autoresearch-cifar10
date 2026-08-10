# Control Batch Size and Learning Rate to Generalize Well
- **Authors**: Fengxiang He, Tongliang Liu, Dacheng Tao
- **Venue**: NeurIPS 2019
- **URL**: https://papers.nips.cc/paper_files/paper/2019/hash/dc6a70712a252123c40d2adba6a11d84-Abstract.html

## Key Contributions
- Relates an SGD PAC-Bayes generalization bound positively to the batch-size/learning-rate ratio.
- Tests 1,600 ResNet-110 and VGG-19 models on CIFAR-10/100 and reports statistically significant empirical support.
- Recommends preventing batch-size/LR ratio from becoming too large when scaling batches.

## Relevance
Larger batches may increase fixed-time image exposure on the H20, but LR should scale coherently and local update count, momentum horizon, BN noise, and switch fit still require direct verification. The paper does not prove that extra examples compensate for fewer decisions in this 300-second CutMix/RandAugment regime.

## Key Techniques
- Couple batch size and LR instead of tuning either independently.
- Treat batch/LR ratio as a generalization-sensitive operating quantity.
- Verify hardware throughput and short-horizon optimization separately from the first-order scaling rule.
