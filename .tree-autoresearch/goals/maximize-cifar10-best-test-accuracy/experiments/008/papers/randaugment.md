# RandAugment: Practical Automated Data Augmentation with a Reduced Search Space
- **Authors**: Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, Quoc V. Le
- **Venue**: CVPR Workshops 2020
- **URL**: https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html

## Key Contributions

- Reduces automated augmentation to a small search over the number of operations and a shared distortion magnitude.
- Matches or improves earlier automated augmentation on CIFAR-10/100, SVHN, ImageNet, and COCO without a separate policy-search phase.
- On CIFAR-10, the paper reports competitive results across WRN-28-2, WRN-28-10, Shake-Shake, and PyramidNet; its default pipelines also include crop, flip, and Cutout.

## Relevance

The direct CIFAR WRN evidence gives RandAugment a larger effect prior than many optimizer refinements. However, the published CIFAR recipe selected operation count and magnitude on a held-out split, and EXP-004 already uses strong front-loaded CutMix. A one-run frozen protocol cannot fairly search those scalars, while stacking transformations may over-regularize or reduce CPU data throughput.

## Key Techniques

- Sample a fixed number of transformations uniformly from a standard operation set.
- Tie all operation strengths to one integer magnitude.
- Retain ordinary crop/flip and combine with other regularization only under a separately validated recipe.
