# Deeply-Supervised Nets
- **Authors**: Chen-Yu Lee, Saining Xie, Patrick Gallagher, Zhengyou Zhang, Zhuowen Tu
- **Venue**: AISTATS 2015
- **URL**: https://proceedings.mlr.press/v38/lee15a.html

## Key Contributions
- Adds companion classification objectives at intermediate hidden layers during end-to-end training.
- Targets more discriminative early features and stronger gradient delivery without using auxiliary heads at inference.
- Reports improved CIFAR-10 and CIFAR-100 classification under data augmentation.

## Relevance
The compact WRN is not obviously gradient-starved, but an inexpensive pooled classifier after the middle stage can directly shape intermediate features without another backbone forward. A clean test must use the same CutMix-adjusted labels for main and auxiliary losses, charge the head computation, discard it from evaluation, and avoid post-hoc coefficient selection.

## Key Techniques
- Attach a lightweight classifier to a selected intermediate representation.
- Optimize a weighted companion cross-entropy jointly with the final classifier.
- Remove or ignore the auxiliary classifier at evaluation.
