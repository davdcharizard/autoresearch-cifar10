# Time Matters in Regularizing Deep Networks
- **Authors**: Aditya Sharad Golatkar, Alessandro Achille, Stefano Soatto
- **Venue**: NeurIPS 2019
- **URL**: https://papers.nips.cc/paper_files/paper/2019/hash/87784eca6b0dea1dff92478fb786b401-Abstract.html

## Key Contributions
- Shows that augmentation, weight decay, and Mixup have their strongest generalization effect during an early critical period.
- Finds that late removal can preserve or improve generalization, while late-only regularization cannot repair poor early dynamics.

## Relevance
Wall-clock-limited CIFAR training should front-load useful regularization and reserve a clean, low-LR late phase for fitting and refinement.

## Key Techniques
- Phase-dependent augmentation and weight decay.
- Early strong regularization followed by late relaxation.
