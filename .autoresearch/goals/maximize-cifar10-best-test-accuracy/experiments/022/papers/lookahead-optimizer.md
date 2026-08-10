# Lookahead Optimizer: k steps forward, 1 step back
- **Authors**: Michael Zhang, James Lucas, Jimmy Ba, Geoffrey E. Hinton
- **Venue**: NeurIPS 2019
- **URL**: https://papers.nips.cc/paper_files/paper/2019/hash/90fd4f88f588ae64038134f1eeaa023f-Abstract.html

## Key Contributions
- Maintains fast optimizer weights and slow weights, synchronizing every `k` inner steps with interpolation factor `alpha`.
- Reduces optimizer variance and improves stability with negligible reported compute overhead.
- Demonstrates gains with SGD and Adam on CIFAR-10/100 and other tasks.

## Relevance
The accepted recipe uses ordinary momentum SGD and a short fixed-time horizon. Lookahead is a small, dependency-free change that can smooth the noisy high-LR trajectory without the backward-pass multiplier of SAM or the late-window bias observed for uniform SWA in EXP-018. Its slow-weight interpolation is also distinct from merely evaluating an EMA shadow model: it feeds the smoothed point back into optimization.

## Key Techniques
- Keep a detached slow copy of trainable parameters.
- Every `k` optimizer steps, update `slow += alpha * (fast - slow)` and copy slow values back to fast weights.
- Preserve the existing inner SGD, learning-rate schedule, augmentation, and evaluation protocol.
