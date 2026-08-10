# Deep Pyramidal Residual Networks
- **Authors**: Dongyoon Han, Jiwhan Kim, Junmo Kim
- **Venue**: CVPR 2017
- **URL**: https://openaccess.thecvf.com/content_cvpr_2017/html/Han_Deep_Pyramidal_Residual_CVPR_2017_paper.html

## Key Contributions
- Replaces abrupt channel doubling at downsampling boundaries with gradual channel growth across residual units.
- Reports stronger CIFAR-10/CIFAR-100 generalization than conventional residual networks.
- Motivates channel allocation as a representation-design variable rather than a fixed stage convention.

## Relevance
The current WRN jumps 64 to 128 to 256 channels. The paper supports testing a more deliberate distribution of capacity, but its every-block projection path would add launches. The transferable principle is to question abrupt width allocation; a budget-compatible experiment should keep the existing convolution count and benchmark latency before training.

## Key Techniques
- Gradual feature-map expansion.
- Residual shortcuts that accommodate changing channel dimensions.
- CIFAR-specific architecture comparison.

_Full PDF fetch returned HTTP 403; distilled from the official CVPR abstract and indexed PDF snippet._
