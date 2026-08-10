# PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
- **Authors**: Zhaoqi Leng, Mingxing Tan, Chenxi Liu, Ekin Dogus Cubuk, Jay Shi, Shuyang Cheng, Dragomir Anguelov
- **Venue**: ICLR 2022
- **URL**: https://openreview.net/forum?id=gSdSJoenupI

## Key Contributions
- Views cross-entropy and focal losses as polynomial expansions whose coefficients can be adjusted for a task.
- Poly-1 adds one term, `epsilon * (1 - p_t)`, to cross-entropy and reports gains across image classification and detection tasks.
- Provides an explicit soft-target-compatible form: `p_t` is the target-weighted softmax probability, so CutMix area labels can be handled without constructing dense labels.

## Relevance

Poly-1 is a one-forward, allocation-light loss change orthogonal to EXP-012's spatial erasure and compatible with both hard and CutMix targets. The key risk is coefficient transfer: the paper states the optimum is task dependent, so EXP-013 must bound gradient inflation rather than copy a large published coefficient blindly.

## Key Techniques
- Hard target: `CE + epsilon * (1 - softmax(logits)[target])`.
- Soft target: use the target-weighted probability in both the cross-entropy and polynomial terms.
- The gradient is a confidence-dependent rescaling of the ordinary softmax-cross-entropy gradient, which supports coefficient calibration from a maximum inflation budget.
