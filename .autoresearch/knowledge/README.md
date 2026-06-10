# Knowledge Base Index

This directory contains persistent external knowledge that informs brainstorming and planning across all research loops. Unlike experiment logs (which are loop-specific), entries here represent standing background context pulled in throughout the research process: relevant paper distillations, reference implementation notes, and domain knowledge.

## How to use
- `research-brainstorm` loads this README as a lightweight index, then loads specific entries on demand
- Entries are added by researchers (manually), by `/lit-search` (automated paper distillations), or proposed by the agent after consulting external sources
- Add entries for any paper, reference implementation, or domain insight that is likely to inform future experiments

## Venues
- `venues.md` (if it exists in this directory) defines which academic proceedings to search via `/lit-search`
- Copy the template from `${CLAUDE_PLUGIN_ROOT}/skills/lit-search/templates/venues-template.md` and edit to match your research domain
- Without a venues file, `/lit-search` defaults to top CS/ML conferences (NeurIPS, ICML, ICLR)

## Papers

| File | Topic | Key Takeaway |
|------|-------|-------------|
| `papers/cutout-cifar-regularization.md` | CIFAR regularization | Cutout-style masking is a cheap first recipe upgrade over crop/flip. |
| `papers/cutmix-regularization.md` | CIFAR regularization | CutMix regionally mixes image patches and labels, retaining pixels unlike Cutout while differing from global mixup. |
| `papers/mixup-beyond-erm.md` | CIFAR regularization | Mixup can regularize CIFAR training via input/label interpolation, but fixed-budget overhead must be measured. |
| `papers/randaugment-augmentation.md` | CIFAR augmentation | RandAugment can test policy augmentation with built-in torchvision transforms and no dependency changes. |
| `papers/resnet-downsampling-tweaks.md` | ResNet architecture | Shortcut/downsampling details can affect accuracy without adding broad model capacity. |
| `papers/sgdr-cosine-schedule.md` | SGD scheduling | Cosine annealing is a low-risk replacement for abrupt step drops under a fixed step/time horizon. |
| `papers/squeeze-and-excitation-networks.md` | CNN channel attention | SE blocks can improve existing CNNs by lightweight channel-wise feature recalibration. |
| `papers/stochastic-depth-resnets.md` | Residual regularization | Training-only residual-branch dropping can regularize ResNets without changing evaluation-time parameters. |
| `papers/wide-residual-networks.md` | CIFAR architectures | Compact wider residual networks are a promising later capacity experiment if recipe changes plateau. |

## References

| File | Topic | Key Takeaway |
|------|-------|-------------|
| `references/pytorch-throughput-tools.md` | PyTorch CNN throughput | `torch.compile`, AMP, and channels-last can test speedups without changing the benchmark recipe. |
| `references/pytorch-ema-averaging.md` | PyTorch EMA averaging | `AveragedModel` can maintain EMA weights without changing the optimizer path or benchmark harness. |
| `references/resnet-zero-init-residual.md` | ResNet initialization | Zeroing the last BN scale in residual branches is a narrow identity-preserving initialization lever. |
| `references/torchvision-randomcrop-padding.md` | CIFAR crop augmentation | `RandomCrop` padding modes allow no-overhead boundary-fill experiments; reflection padding is now validated. |

## Domain Notes

| File | Topic | Key Takeaway |
|------|-------|-------------|
| _(empty — add domain-specific notes here)_ | | |
