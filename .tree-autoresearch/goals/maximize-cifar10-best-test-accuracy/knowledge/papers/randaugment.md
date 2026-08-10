# RandAugment: Practical Automated Data Augmentation with a Reduced Search Space
- **Authors**: Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, Quoc V. Le
- **Venue**: CVPR Workshops 2020
- **URL**: https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html

## Key Contributions

- Reduces automated augmentation to operation count and one shared distortion magnitude.
- Reports competitive results across CIFAR-10 WRN-28-2, WRN-28-10, Shake-Shake, and PyramidNet without a separate learned policy.
- The matched operating point depends on model capacity; CIFAR WRN settings should be transferred explicitly rather than inferred from generic library defaults.

## Relevance

RandAugment offers direct CIFAR WRN evidence and no extra model pass, but its policy can overlap CutMix and disrupt a validated clean optimization tail. Future experiments should preregister a capacity-matched published configuration, isolate transform RNG from the parent pipeline, measure loader wall time, and use phase gating when the existing recipe already front-loads regularization.

## Key Techniques

- Sample a fixed number of operations from a standard transform set.
- Tie operation strengths to one magnitude index.
- Apply operations in PIL/image space before tensor normalization.
