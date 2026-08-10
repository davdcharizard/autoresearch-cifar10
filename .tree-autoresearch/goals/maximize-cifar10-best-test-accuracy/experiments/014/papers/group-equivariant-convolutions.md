# Group Equivariant Convolutional Networks
- **Authors**: Taco Cohen, Max Welling
- **Venue**: ICML 2016
- **URL**: https://proceedings.mlr.press/v48/cohenc16.html

## Key Contributions
- Generalizes convolution to discrete transformation groups so feature maps share weights across rotations and reflections.
- Increases effective expressive capacity through symmetry-aware weight sharing without proportional parameter growth.
- Demonstrates strong CIFAR-10 results and reports low overhead for suitable discrete groups.

## Relevance
CIFAR-10 has horizontal-flip augmentation but not an exact rotation-label symmetry, so a reflection-equivariant stem is more defensible than a rotation group. Implementing group convolutions inside the one-file constraint would still be a large architecture change with kernel-layout and initialization risks; it is a moonshot, not a default extension.

## Key Techniques
- Lift ordinary feature maps into a discrete group orientation axis.
- Tie convolutional filters under chosen group transformations.
- Restrict the symmetry group to transformations that preserve labels and the existing data contract.
