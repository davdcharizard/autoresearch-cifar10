# Lookahead Optimizer: k Steps Forward, 1 Step Back
- **Authors**: Michael Zhang, James Lucas, Jimmy Ba, Geoffrey E. Hinton
- **Venue**: NeurIPS 2019
- **URL**: https://papers.nips.cc/paper_files/paper/2019/hash/90fd4f88f588ae64038134f1eeaa023f-Abstract.html

## Key Contributions
- Wraps an inner optimizer with slow weights updated every `k` fast steps, then copies the slow weights back to the online model.
- Motivates the slow path as a recent exponential average of fast endpoints that reduces variance and improves stability at negligible claimed compute.
- Reports improvements for SGD and Adam across CIFAR-10/100, ImageNet, translation, and language modeling.

## Relevance
Lookahead would alter the whole accepted trajectory rather than post-hoc average the harmful annealed tail from EXP-018. However, it still repeatedly pulls a progressing optimizer backward, requires a fixed synchronization period and interpolation coefficient, and adds full-model copies within counted time. Its relationship to the failed local averaging result is a major risk.

## Key Techniques
- Keep detached slow weights initialized from the online model.
- After fixed `k` inner steps, update `slow += alpha * (fast-slow)` and install slow into fast.
- Charge every interpolation/copy to counted time and keep evaluation on installed online weights.
