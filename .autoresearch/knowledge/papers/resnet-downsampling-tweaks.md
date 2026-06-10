# ResNet Downsampling Tweaks

**Sources**:
- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016, https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
- He et al., "Bag of Tricks for Image Classification with Convolutional Neural Networks", CVPR 2019, https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf

## Key Takeaway
ResNet shortcut/downsampling details can affect accuracy. The original CIFAR option-A shortcut is parameter-free, using identity/zero-padding across dimension changes, while later ResNet downsampling tweaks use average pooling in shortcut/downsample paths to reduce information loss from stride-based sampling.

## Relevance to This Project
The current CIFAR model uses a parameter-free option-A-style shortcut implemented as strided spatial slicing plus zero-channel padding. A low-overhead experiment can replace slicing with average pooling before zero-padding, testing transition quality without adding trainable projection shortcuts or changing the optimizer recipe.

## Implementation Notes
- Keep the shortcut parameter-free to remain distinct from the failed learned projection shortcut experiment.
- Apply the change only when `need_pad` is true and spatial stride is greater than 1.
- Preserve zero-channel padding after spatial downsampling so output shapes match the residual branch.
- Verify step-21000 LR drop, since even small shortcut changes can alter throughput under the fixed time budget.
