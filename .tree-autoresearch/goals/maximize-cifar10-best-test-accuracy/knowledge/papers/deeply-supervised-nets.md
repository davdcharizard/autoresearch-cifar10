# Deeply-Supervised Nets
- **Authors**: Chen-Yu Lee, Saining Xie, Patrick Gallagher, Zhengyou Zhang, Zhuowen Tu
- **Venue**: AISTATS 2015
- **URL**: https://proceedings.mlr.press/v38/lee15a.html

## Key Contributions
- Adds companion classification objectives at intermediate layers during end-to-end training.
- Encourages discriminative hidden representations and supplies direct gradient signal.
- Uses auxiliary heads only for training and reports CIFAR gains.

## Relevance
A small pooled middle-stage head can add representation supervision to the compact WRN without another backbone forward. It must share the parent's augmented target semantics, remain charged, and stay excluded from evaluation.

## Key Techniques
- Intermediate pooled classifier with weighted companion cross-entropy.
- Joint training with the main output and inference-time removal.
