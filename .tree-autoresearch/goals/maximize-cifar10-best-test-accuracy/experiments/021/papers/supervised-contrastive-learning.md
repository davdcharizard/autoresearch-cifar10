# Supervised Contrastive Learning
- **Authors**: Prannay Khosla et al.
- **Venue**: NeurIPS 2020
- **URL**: https://papers.nips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html

## Key Contributions
- Extends contrastive objectives to use all same-class samples in a batch as positives and all other classes as negatives.
- Uses normalized embeddings, a disposable projection head, and temperature 0.1; reports consistent classification and corruption-robustness gains over cross-entropy.
- Its preferred formulation averages log-probabilities across positives and implicitly emphasizes hard positives and negatives.

## Relevance
A single-view, same-batch auxiliary SupCon term could directly shape EXP004's pooled representation without another backbone forward. This is an adaptation, not a reproduction: the primary paper uses two augmented views, a disposable projection head, long training, and then a separate linear-classifier stage. Under this fixed 300-second joint-CE protocol, compute and transfer risk are material and effect size must be discounted.

## Key Techniques
- Normalize embeddings and compute the batch similarity matrix with self-comparisons masked.
- Treat same-class samples as multiple positives and use temperature-scaled log-softmax.
- Discard the projection head at inference.
