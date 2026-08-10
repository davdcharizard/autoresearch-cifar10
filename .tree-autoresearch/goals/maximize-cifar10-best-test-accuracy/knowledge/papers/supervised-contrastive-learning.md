# Supervised Contrastive Learning
- **Authors**: Prannay Khosla et al.
- **Venue**: NeurIPS 2020
- **URL**: https://papers.nips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html

## Key Contributions
- Treats all same-class batch examples as positives and other classes as negatives.
- Uses normalized embeddings, temperature 0.1, two augmented views, and a disposable projection head.
- Reports classification and corruption-robustness gains, with hard positive/negative emphasis emerging from the loss.

## Relevance
Same-batch class geometry is a plausible CIFAR representation lever, but the canonical evidence uses two views, long contrastive training, a projection head, and a separate classifier stage. Single-view joint-CE variants are substantial adaptations and must not inherit the headline effect size.

## Key Techniques
- Many-positive supervised contrastive log-softmax.
- Normalized projection embeddings and temperature scaling.
- Inference-time removal of the contrastive projection head.
