# RandAugment: Practical Automated Data Augmentation with a Reduced Search Space
- **Authors**: Ekin Dogus Cubuk, Barret Zoph, Jon Shlens, Quoc V. Le
- **Venue**: NeurIPS 2020
- **URL**: https://papers.nips.cc/paper_files/paper/2020/hash/d85b63ef0ccb114d0a3bb7b7d808028f-Abstract.html

## Key Contributions
- Collapses automated augmentation policy tuning to a small shared magnitude and operation-count search space.
- Reports state-of-the-art results on CIFAR-10 without a separate expensive policy search.

## Relevance
Torchvision already provides RandAugment, so it can add augmentation diversity without a new dependency. CPU transform overhead and excessive strength are the main risks under a fixed training-time budget.

## Key Techniques
- Apply a small number of randomly chosen transforms at a shared magnitude.
- Tune only operation count and magnitude rather than a full policy.
