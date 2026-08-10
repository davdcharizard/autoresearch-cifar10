# Temperature in Cosine-based Softmax Loss
- **Authors**: Takumi Kobayashi
- **Venue**: ICCV 2025
- **URL**: https://openaccess.thecvf.com/content/ICCV2025/papers/Kobayashi_Temperature_in_Cosine-based_Softmax_Loss_ICCV_2025_paper.pdf

## Key Contributions
- Studies fixed and adaptive scale/temperature in cosine-normalized softmax classifiers.
- Demonstrates strong task/model sensitivity: poor scales can fail optimization, while a competitive fixed scale can exceed affine softmax.
- On ResNet-34/CIFAR-10, affine softmax reached 95.56%, fixed scale 40 reached 95.85%, and the proposed learned scale reached 95.49%.

## Relevance

Cosine classification is a low-MAC class-boundary intervention, but fixed scale 40 should be treated as a sweep-selected transfer point rather than an unbiased +0.29 expectation. Feature dimension changes cosine spread, so the same scale is effectively sharper for this project's 256-dimensional representation than the paper's 512-dimensional ResNet-34 representation. Any test must preregister one scale, audit early optimization and raw row-norm dynamics, and forbid post-metric retuning.

## Key Techniques
- L2-normalize pooled features and classifier rows before the final matrix multiply.
- Multiply cosine similarities by one fixed scale controlling softmax sharpness.
- Separate directional classifier geometry from radial weight dynamics and observe norm behavior explicitly.
