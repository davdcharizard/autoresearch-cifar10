# Control Batch Size and Learning Rate to Generalize Well: Theoretical and Empirical Evidence
- **Authors**: Fengxiang He, Tongliang Liu, Dacheng Tao
- **Venue**: NeurIPS 2019
- **URL**: https://papers.nips.cc/paper_files/paper/2019/hash/dc6a70712a252123c40d2adba6a11d84-Abstract.html

## Key Contributions
- Relates an SGD PAC-Bayes generalization bound positively to the batch-size/learning-rate ratio.
- Tests 1,600 ResNet-110 and VGG-19 models on CIFAR-10/100 and finds statistically significant empirical support.
- Recommends preventing the batch-size/learning-rate ratio from becoming too large when scaling batches.

## Relevance
The H20 has large unused memory and batch scaling could increase image exposure per fixed second, but batch 256 at unchanged LR would double the ratio associated with worse generalization. A coherent local test should scale LR with batch size and separately verify update geometry, switch fit, and actual image throughput.

## Key Techniques
- Couple batch-size changes to learning-rate changes rather than tuning either independently.
- Treat batch/LR ratio as a generalization-sensitive operating quantity.
- Verify on the local short-horizon CutMix/RandAugment regime because the paper uses much deeper models and epoch budgets.
