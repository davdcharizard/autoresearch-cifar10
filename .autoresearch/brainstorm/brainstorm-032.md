# Brainstorm EXP-032
**Created**: 2026-06-04
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- Reflect padding is used in several high-performance CIFAR-10 pipelines as an alternative to zero padding
- Research notes that zero-padding borders serve as spatial position anchors that don't generalize — reflect padding removes this artifact
- The airbench CIFAR-10 speedrun uses reflect padding in its augmentation pipeline

## Experimental History Review

- **33 experiments** (BASE through EXP-031), baseline 96.39%, 16 consecutive failures
- NEVER TRIED: RandomCrop padding mode. All experiments used `padding_mode='constant'` (zero padding, the default)
- The training augmentation pipeline has been static across ALL experiments: `RandomCrop(32, padding=4)` + `RandomHorizontalFlip()` + `ToTensor()` + `Normalize()`
- Padding mode is the one augmentation parameter never explored
- EXP-030 showed padding SIZE matters (6 is too large → -2.1%), but padding MODE is orthogonal — same padding size, different fill strategy

## Candidate Ideas

### 1. RandomCrop padding_mode='reflect'
**Summary**: Change `transforms.RandomCrop(32, padding=4)` to `transforms.RandomCrop(32, padding=4, padding_mode='reflect')`. Instead of zero-padded borders (black regions) on shifted crops, the padding uses reflected image content. This is the only augmentation pipeline parameter never explored in 31 experiments.

**Reasoning**: With zero padding (current), crops near the image border contain black zero-filled regions. The model may learn to use these zeros as positional cues — a form of shortcut learning that doesn't generalize to test images (which are center-cropped, no padding). With reflect padding, shifted crops contain natural image content at borders, eliminating this artifact. The model is forced to classify based purely on image features, not border artifacts. This has zero computational cost — reflect padding is the same speed as zero padding.

**Sources**: Research on position information from zero-padding, airbench implementation notes

**Estimated Effort**: low — change one parameter in one line

**Risk Assessment**: Very low. The model still sees the same crop positions, just with natural content instead of zeros at borders. Worst case: no improvement (the model doesn't use border artifacts anyway). Cannot cause regression since the augmented images are strictly more natural.

### 2. Asymmetric Channel Widths [48, 128, 384]
**Summary**: Replace uniform width scaling [64, 128, 256] with asymmetric [48, 128, 384]. Less capacity in layer1 (simpler spatial features at 32×32) and more in layer3 (complex abstract features at 8×8, feeding the classifier). Total params similar to current 4.3M.

**Reasoning**: The first layer processes 32×32 spatial features that are relatively simple (edges, textures). The final layer processes 8×8 abstract features that are most critical for classification. Allocating more capacity to layer3 gives the classifier more features to work with. This is inspired by EfficientNet's compound scaling principle.

**Sources**: EfficientNet compound scaling, asymmetric ResNet variants in literature

**Estimated Effort**: medium — change width multiplier to custom list, verify param count and timing

**Risk Assessment**: Medium. Custom widths change the architecture significantly. If layer1 is too narrow (48 channels), early feature extraction may suffer. Untested combination.

## Idea Evaluation

Reflect padding is the clear winner: zero cost, zero risk, genuinely untried, and addresses a real artifact in the training pipeline. Asymmetric widths are more speculative and higher risk.

## Chosen Idea
**Selected**: RandomCrop padding_mode='reflect'

**Why this idea**: The only augmentation pipeline parameter never explored in 31 experiments. Zero cost, eliminates artificial border artifacts, provides more natural training images.

**Hypothesis**: Reflect padding in RandomCrop will eliminate zero-border positional shortcuts, improving generalization and pushing best_test_acc above 96.49%.
