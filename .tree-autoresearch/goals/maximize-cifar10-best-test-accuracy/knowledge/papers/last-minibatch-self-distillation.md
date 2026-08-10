# Self-Distillation from the Last Mini-Batch

- Source: Yiqing Shen et al., CVPR 2022, https://openaccess.thecvf.com/content/CVPR2022/html/Shen_Self-Distillation_From_the_Last_Mini-Batch_for_Consistency_Regularization_CVPR_2022_paper.html
- Setting: supervised image classification on CIFAR-10, CIFAR-100, and TinyImageNet.

## Core Method

DLB rearranges sampling so half of each batch repeats half of the previous batch. Detached logits from the immediately previous model state become soft targets for the repeated half on its next occurrence. The total objective is ordinary cross-entropy plus `alpha * tau^2 * KL(teacher || student)`, with published defaults `tau=3` and `alpha=1`.

It requires one ordinary model forward/backward per step, no teacher model, and only one half-batch of cached logits. The benefit is temporal and augmentation-view consistency from the freshest available historical prediction.

## Evidence

The paper reports 0.37-1.01-point CIFAR-10 error reductions across VGG, ResNet, WRN-20-8, and DenseNet backbones. DLB is explicitly composed with CutMix: CIFAR-10 CutMix+DLB improves over CutMix alone by 0.09-1.48 points across the reported architectures, including a 0.60-point error reduction on WRN-20-8. Results are three-run averages on weaker and longer-trained baselines than the current goal recipe.

## Reusable Caveats

- Correctness depends on exact sample/logit alignment; cache sample identities and assert equality at runtime.
- Spatially mixed examples do not retain a simple one-example teacher identity. Either reproduce the paper's exact composition or gate distillation to clean-clean transitions.
- Half-overlap preserves examples per optimizer step but halves the rate of new unique examples and can lengthen the natural epoch, reducing evaluation opportunities when evaluation is epoch-based.
- When combined with a two-pass optimizer such as SAM, use the same detached incoming teacher in both objectives and publish only the intended primary-forward outgoing logits.
- Gains on a strongly regularized parent should be discounted from the paper's baseline improvements.
