# Time Matters in Regularizing Deep Networks
- **Authors**: Aditya Sharad Golatkar, Alessandro Achille, Stefano Soatto
- **Venue**: NeurIPS 2019
- **URL**: https://papers.nips.cc/paper_files/paper/2019/hash/87784eca6b0dea1dff92478fb786b401-Abstract.html

## Key Contributions
- Shows that weight decay, augmentation, and Mixup have their strongest generalization effect during an early critical period.
- Finds that removing regularization late can retain or improve generalization, while adding it only late does not repair poor early dynamics.

## Relevance
The benchmark is wall-clock limited, so regularization should shape early learning without imposing unnecessary late cost. This supports front-loaded augmentation or regularization schedules paired with deliberate late optimization.

## Key Techniques
- Phase-dependent augmentation and weight decay.
- Early strong regularization followed by late relaxation.
