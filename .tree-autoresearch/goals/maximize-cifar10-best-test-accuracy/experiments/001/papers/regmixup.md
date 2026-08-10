# Using Mixup as a Regularizer Can Surprisingly Improve Accuracy & Out-of-Distribution Robustness
- **Authors**: Francesco Pinto, Harry Yang, Ser Nam Lim, Philip Torr, Puneet Dokania
- **Venue**: NeurIPS 2022
- **URL**: https://papers.nips.cc/paper_files/paper/2022/hash/5ddcfaad1cb72ce6f1a365e8f1ecf791-Abstract-Conference.html

## Key Contributions
- Uses Mixup as an additional regularizer alongside ordinary cross-entropy rather than replacing the clean-example objective.
- Reports improved accuracy and uncertainty behavior across ImageNet and CIFAR-10/100.

## Relevance
The baseline has no mixed-sample regularization. A clean-plus-mixed loss can add a strong generalization prior while retaining direct supervision, though its extra forward work must be weighed against the fixed time budget.

## Key Techniques
- Joint clean-example cross-entropy and Mixup regularization.
- Mixup strength and loss weighting as tunable controls.
