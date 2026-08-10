# RandAugment: Practical Automated Data Augmentation with a Reduced Search Space
- **Authors**: Ekin Dogus Cubuk, Barret Zoph, Jon Shlens, Quoc V. Le
- **Venue**: NeurIPS 2020
- **URL**: https://papers.nips.cc/paper/2020/hash/d85b63ef0ccb114d0a3bb7b7d808028f-Abstract.html

## Key Contributions
- Reduces automated augmentation to the number of operations and a shared magnitude.
- Removes the need for a costly policy-search proxy task.
- Reports state-of-the-art performance on CIFAR-10 and other image benchmarks.

## Relevance
`torchvision.transforms.RandAugment` is already available, so this can strengthen augmentation without a dependency or evaluation change. The main risk is host-side transform overhead and excessive distortion for a small ResNet20 under a tight wall-clock limit.

## Key Techniques
- Apply a small fixed number of randomly selected transforms per image.
- Tune one shared magnitude for the target model and dataset size.
- Combine with standard random crop and horizontal flip.

