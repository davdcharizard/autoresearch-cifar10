# Squeeze-and-Excitation Networks
- **Authors**: Jie Hu, Li Shen, Gang Sun
- **Venue**: CVPR 2018
- **URL**: https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper

## Key Contributions

- Uses global pooling and a small bottleneck MLP to model channel interdependencies and recalibrate feature responses.
- Inserts channel recalibration into residual blocks with modest added parameters and compute.
- Establishes channel attention as an effective general architectural mechanism, though the strongest published evidence is on deeper ImageNet models.

## Relevance

SE offers a fuller cross-channel model than ECA but adds more parameters and kernels. Standard sigmoid gates start near one half; a short fixed-time CIFAR run needs an identity-centered, zero-final-layer adaptation to preserve the parent function and initialization stream.

## Key Techniques

- Global average pooling.
- Reduction MLP with nonlinear hidden layer.
- Sigmoid channel excitation applied to a residual transformation.
