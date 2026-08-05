# Time Matters in Regularizing Deep Networks
- **Authors**: Aditya Sharad Golatkar, Alessandro Achille, Stefano Soatto
- **Venue**: NeurIPS 2019
- **URL**: https://papers.nips.cc/paper_files/paper/2019/hash/87784eca6b0dea1dff92478fb786b401-Abstract.html

## Key Contributions
- Shows that weight decay, augmentation, and mixup exert most of their generalization effect during an early critical period.
- Finds that removing regularization after the early transient can preserve or improve generalization.

## Relevance
The baseline has only 300 training seconds. Strong augmentation or mixup early followed by clean fine-tuning may capture generalization benefits without slowing final convergence or depressing late hard-label accuracy.

## Key Techniques
- Apply regularization strongly during early training.
- Reduce or disable regularization during the late convergence phase.
