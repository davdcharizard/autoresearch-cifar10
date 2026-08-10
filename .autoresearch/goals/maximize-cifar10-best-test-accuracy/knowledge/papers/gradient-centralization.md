# Gradient Centralization: A New Optimization Technique for Deep Neural Networks
- **Authors**: Hongwei Yong, Jianqiang Huang, Xiansheng Hua, Lei Zhang
- **Venue**: ECCV 2020
- **URL**: https://arxiv.org/abs/2004.01461

## Key Contributions
- Projects eligible weight gradients onto a zero-mean subspace by subtracting their mean over non-output dimensions.
- Interprets the operation as constrained projected gradient descent and reports improved optimization/generalization across vision tasks.
- Embeds into existing optimizers without a second gradient evaluation or model-forward change.

## Relevance

Conv-only centralization is a low-overhead, fixed-graph regularizer for the accepted width-2 ResNet-20. It must preserve ordinary SGD momentum and all-parameter coupled decay semantics: centralize the data gradient after backward but before `optimizer.step`, while recognizing PyTorch adds decay inside SGD afterward. Direct CIFAR-10 evidence is not from this exact short CutMix recipe, and any extra reduction cost must pass fixed-time timing.

## Key Techniques
- For Conv2d weight gradient `[out,in,h,w]`, subtract the mean across `(in,h,w)` for each output filter.
- Do not centralize 1D BN/bias gradients or the 2D classifier unless separately justified.
- Verify projected means, momentum-buffer recurrence, update scale, and finite multi-step behavior.
