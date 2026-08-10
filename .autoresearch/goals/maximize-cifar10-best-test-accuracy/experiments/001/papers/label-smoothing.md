# When Does Label Smoothing Help?
- **Authors**: Rafael Muller, Simon Kornblith, Geoffrey E. Hinton
- **Venue**: NeurIPS 2019
- **URL**: https://papers.nips.cc/paper_files/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html

## Key Contributions
- Shows that soft targets often improve generalization and calibration.
- Connects label smoothing to tighter within-class representation clusters.
- Identifies a tradeoff with knowledge distillation, which is irrelevant to this goal.

## Relevance
Cross-entropy label smoothing is a one-line, near-zero-cost change that may improve the baseline's generalization. Its likely effect is smaller than a richer augmentation or architecture change, and combining it with Mixup can over-regularize a small model.

## Key Techniques
- Replace hard one-hot targets with a mixture of the target and uniform label distributions.
- Use a modest smoothing strength, commonly around 0.1.
- Evaluate interactions with other soft-label regularizers rather than stacking blindly.
