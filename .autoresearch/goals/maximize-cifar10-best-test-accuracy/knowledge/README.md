# Knowledge Base Index

This directory contains persistent external knowledge that informs brainstorming and planning across all research loops. Unlike experiment logs (which are loop-specific), entries here represent standing background context pulled in throughout the research process: relevant paper distillations, reference implementation notes, and domain knowledge.

## How to use
- `research-brainstorm` loads this README as a lightweight index, then loads specific entries on demand
- Entries are added by researchers (manually), by `/lit-search` (automated paper distillations), or proposed by the agent after consulting external sources
- Add entries for any paper, reference implementation, or domain insight that is likely to inform future experiments

## Venues
- `venues.md` in this directory defines which academic proceedings `/lit-search` searches for this goal — seeded with the default top CS/ML conferences (NeurIPS, ICML, ICLR)
- Edit it to match this goal's research domain; venues are per-goal, so different goals can search different domains

## Papers

| File | Topic | Key Takeaway |
|------|-------|-------------|
| `papers/mixup.md` | Soft-target data mixing | Conservative input/target interpolation can improve CIFAR-10 generalization. |
| `papers/mixed-pooling.md` | Learned spatial aggregation | A learned max/average mixture can improve CNN readout with modest overhead. |
| `papers/randaugment.md` | Automated augmentation | A small augmentation search space gives strong CIFAR results but adds host work. |
| `papers/sgdr.md` | Cosine LR annealing | Express smooth annealing in the actual training horizon for strong anytime convergence. |
| `papers/weight-averaging.md` | Trajectory averaging | Averaging complements annealing, but BatchNorm state must be handled explicitly. |
| `papers/label-smoothing.md` | Soft-target regularization | Modest label smoothing can improve generalization at near-zero runtime cost. |
| `papers/large-minibatch-sgd.md` | Batch/LR scaling | Linear LR scaling is a coherent large-batch rule, but local optimization and throughput must be verified. |
| `papers/wide-residual-networks.md` | Width scaling | Wider shallow residual networks can improve CIFAR capacity efficiently, but fixed-time update loss must be measured. |
| `papers/cutmix.md` | Labeled regional mixing | CutMix preserves class-bearing pixels and area targets, avoiding Cutout's information deletion. |
| `papers/zero-gamma-resnet.md` | Residual identity initialization | Zero final BN scales only where shortcuts are true identities; padded Option-A transitions must stay active. |
| `papers/resnet-d-downsampling.md` | Information-preserving transition shortcuts | Pool before learned projection to use every shortcut sample; CIFAR effect size and fixed-time cost require local validation. |
| `papers/eca-net.md` | Lightweight residual channel attention | Tiny channel-axis attention can improve ResNets, but shallow CutMix models need identity-scale and timing safeguards. |
| `papers/positive-negative-momentum.md` | Optimizer-shaped gradient noise | PNM has close CIFAR evidence, but paper-default scale and decay do not match PyTorch momentum without explicit correction. |
| `papers/lookahead-optimizer.md` | Fast/slow optimizer interpolation | Lookahead can reduce trajectory variance cheaply, but local use must check committed progress and momentum/location mismatch. |
| `papers/gradient-centralization.md` | Projected convolution gradients | Conv-filter gradient centering is a cheap regularizer, but decay ordering, update scale, and fixed-time cost need local gates. |
| `papers/stochastic-depth.md` | Random residual-block bypass | Can regularize and shorten training graphs, but published gains target far deeper ResNets than the local nine-block model. |
| `papers/activation-initialization-edge-of-chaos.md` | Smooth activations and signal propagation | Swish/SiLU has a signal-propagation mechanism beyond ReLU, but initialization compatibility and local throughput must be verified. |
| `papers/control-batch-size-and-learning-rate.md` | Batch/LR scaling and generalization | Preserve batch-size/LR ratio when scaling SGD, then verify local throughput, update loss, and phase fit. |
| `papers/demystifying-loss-functions.md` | Classification loss and normalized heads | Cosine/logit-normalized heads require explicit temperature and optimizer-geometry checks; output-scale parity alone is insufficient. |

## References

| File | Topic | Key Takeaway |
|------|-------|-------------|
| `references/pytorch-channels-last.md` | CUDA convolution memory layout | Channels-last may accelerate supported vision kernels, but FP32 CIFAR gains require direct end-to-end measurement. |

## Domain Notes

| File | Topic | Key Takeaway |
|------|-------|-------------|
| _(empty — add domain-specific notes here)_ | | |
