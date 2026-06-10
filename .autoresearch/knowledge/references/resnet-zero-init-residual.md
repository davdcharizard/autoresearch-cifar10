# ResNet Zero-Init Residual Branches

## Sources
- Torchvision ResNet source: https://docs.pytorch.org/vision/master/_modules/torchvision/models/resnet.html
- He et al., "Identity Mappings in Deep Residual Networks": https://arxiv.org/abs/1603.05027
- De and Smith, "Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks": https://arxiv.org/abs/2002.10444

## Key Insight
Zero-initializing the final BatchNorm scale in each residual branch makes each block start closer to an identity mapping. Torchvision implements this as `zero_init_residual` by setting `BasicBlock.bn2.weight` or `Bottleneck.bn3.weight` to zero after normal module initialization.

## Relevance to This Repo
The local CIFAR ResNet uses a post-activation `BasicBlock` with `bn2` immediately before the residual addition, so the analogous isolated change is `init.constant_(m.bn2.weight, 0)` for every `BasicBlock`. This changes initialization only: no parameter count, optimizer, schedule, augmentation, evaluation, or throughput path should change.

## Risks
Under a strict 300s training budget, identity-biased residual branches may slow early feature learning even if they improve deeper or longer-trained ResNets.
