# A Unified Analysis of Mixed Sample Data Augmentation
- **Authors**: Chanwoo Park, Sangdoo Yun, Sanghyuk Chun
- **Venue**: NeurIPS 2022
- **URL**: https://papers.nips.cc/paper_files/paper/2022/hash/e6f32e64b9c27d153b46c94f0fe22b56-Abstract-Conference.html

## Key Contributions
- Analyzes Mixup and CutMix as pixel-level loss and first-layer regularizers.
- Explains that CutMix and Mixup regularize input gradients differently and proposes efficient hybrid variants.

## Relevance
Mixed-sample augmentation can provide meaningful regularization at negligible arithmetic overhead. The task-dependent distinction argues for a simple selectable Mixup/CutMix hybrid rather than assuming one method is universally best.

## Key Techniques
- Mixup, CutMix, and hybrid mixed-sample augmentation.
- Soft-target cross-entropy.
