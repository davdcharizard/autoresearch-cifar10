# Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour
- **Authors**: Priya Goyal et al.
- **Venue**: arXiv technical report, 2017
- **URL**: https://arxiv.org/pdf/1706.02677

## Key Contributions
- Establishes a large-batch ResNet training recipe with linear LR scaling and warmup.
- Reports that zero-initializing the last BatchNorm scale in each residual branch improved ResNet-50 ImageNet top-1 error from 23.84 to 23.60.
- Motivates the initialization by making residual blocks begin as identity mappings.

## Relevance
The accepted postactivation CIFAR blocks place `bn2` immediately before shortcut addition, so zero gamma can create exact forward identities in same-width blocks. Transfer is not literal: padded Option-A transition channels have zero shortcuts and a post-add ReLU, so zeroing transition residuals can permanently deadlock the new channels. Apply only where an actual equal-shape identity shortcut exists, and preflight first-update scale.

## Key Techniques
- Zero the final residual BatchNorm scale without changing topology or parameter count.
- Preserve active residual branches at zero-padded dimension transitions.
- Verify gamma receives a finite first gradient and residual convolutions recruit after the first update.
