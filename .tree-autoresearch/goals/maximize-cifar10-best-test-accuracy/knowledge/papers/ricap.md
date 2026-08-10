# RICAP: Random Image Cropping and Patching Data Augmentation for Deep CNNs
- **Authors**: Ryo Takahashi, Takashi Matsubara, Kuniaki Uehara
- **Venue**: ACML 2018 (PMLR 95)
- **URL**: https://proceedings.mlr.press/v95/takahashi18a.html

## Key Contributions
- Constructs images from four random crops with area-weighted labels, combining feature removal and soft targets.
- Reports gains across WideResNet, Pyramidal ResNet, and Shake-Shake on CIFAR.
- Its controlled WideResNet table reports 16x16 reference Cutout reducing CIFAR-10 error from 3.89% to 3.08%.

## Relevance
Spatial feature removal has matched CIFAR/WideResNet evidence, but RICAP overlaps CutMix's multi-image mixing. Complementary center-sampled Cutout on non-CutMix batches is a cleaner additive test that preserves validated mixed-label exposure.

## Key Techniques
- Reference Cutout samples an image center uniformly and clips the nominal square at edges.
- RICAP uses area-proportional labels and warns that excessive softening can hurt.
- Spatial masks encourage use of secondary object features.
