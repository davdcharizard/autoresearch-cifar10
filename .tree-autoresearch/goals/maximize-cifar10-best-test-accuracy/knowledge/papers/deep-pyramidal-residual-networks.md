# Deep Pyramidal Residual Networks
- **Authors**: Dongyoon Han, Jiwhan Kim, Junmo Kim
- **Venue**: CVPR 2017
- **URL**: https://openaccess.thecvf.com/content_cvpr_2017/html/Han_Deep_Pyramidal_Residual_CVPR_2017_paper.html

## Key Contributions
- Replaces abrupt channel doubling at downsampling boundaries with gradual channel growth across residual units.
- Reports stronger CIFAR-10/CIFAR-100 generalization than conventional residual networks.
- Establishes channel allocation as an architecture variable, while not directly validating back-loaded stage depth.

## Relevance
Future architecture experiments can vary capacity distribution rather than only total depth or width. Implementations under a fixed wall-clock budget must separately account for projection-path kernel overhead; the paper's every-block gradual widening is not compute-equivalent to simple stage-depth reallocation.

## Key Techniques
- Gradual feature-map expansion.
- Residual shortcuts accommodating changing dimensions.
- CIFAR-specific architecture comparisons.

_Full PDF fetch returned HTTP 403; distilled from the official CVPR abstract and indexed PDF snippet._
