# Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour
- **Authors**: Priya Goyal et al.
- **Venue**: arXiv 2017 / Facebook AI Research
- **URL**: https://arxiv.org/abs/1706.02677

## Key Contributions

- Uses linear learning-rate scaling to preserve first-order update magnitude as batch size grows.
- Shows large batches introduce optimization difficulty rather than an unavoidable generalization gap when the operating regime is controlled.
- Uses gradual warmup for extreme distributed scaling and reports zero-initialized final residual BN as a separate optimization refinement.

## Relevance

EXP-013 considers only a 2x single-GPU batch increase, so linear scaling is a coherent pre-registered batch/LR method while extreme-batch warmup is an additional mechanism rather than an automatic requirement. Local H20 timing and strong-phase diagnostics remain necessary because the paper does not establish accuracy for a 300-second CIFAR-10 horizon with CutMix/RandAugment.
