# RICAP: Random Image Cropping and Patching Data Augmentation for Deep CNNs
- **Authors**: Ryo Takahashi, Takashi Matsubara, Kuniaki Uehara
- **Venue**: ACML 2018 (PMLR 95)
- **URL**: https://proceedings.mlr.press/v95/takahashi18a.html

## Key Contributions
- Constructs one training image from four random image crops and mixes labels by patch area, combining spatial feature removal with soft targets.
- Reports CIFAR-10 improvements across WideResNet, Pyramidal ResNet, and Shake-Shake families.
- Its controlled WideResNet table also reports cutout improving error from 3.89% to 3.08%, while simple input dropout worsened it to 4.69%.

## Relevance
EXP-012 needs a single-view, low-overhead regularizer that preserves the proven CutMix/SAM/EMA package. The paper supplies direct CIFAR WideResNet evidence that spatial occlusion can add meaningful generalization, but RICAP itself overlaps CutMix's multi-image label mixing. A more differentiated candidate is dedicated-RNG cutout only on early batches where CutMix was not selected, preserving CutMix dose and adding no second model view.

## Key Techniques
- CIFAR cutout uses a 16x16 masked square.
- RICAP uses four cropped images with area-proportional labels; excessive label smoothing harmed some operating points.
- Spatial masking forces the network to use secondary object features rather than the most salient patch.
