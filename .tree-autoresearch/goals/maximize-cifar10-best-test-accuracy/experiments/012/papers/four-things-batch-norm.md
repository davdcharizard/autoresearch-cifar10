# Four Things Everyone Should Know to Improve Batch Normalization
- **Authors**: Cecilia Summers, Michael J. Dinneen
- **Venue**: ICLR 2020
- **URL**: https://arxiv.org/abs/1906.03548

## Key Contributions
- Identifies four context-dependent BatchNorm improvements with no additional training computation, including Ghost BatchNorm and regularizing scale/shift parameters.
- Finds that weight decay on BatchNorm gamma/beta can help residual architectures because skip paths keep their regularization effect visible to the output.
- Reports a small but significant 0.3-point CIFAR-100 gain across ten runs from regularizing gamma/beta; effects vary by architecture and task.

## Relevance
The current optimizer already applies weight decay to every parameter, including all BatchNorm gamma/beta and classifier bias. The evidence argues against the common reflex to exclude all normalization parameters from decay on ResNets, so an optimizer-group experiment that removes BN decay lacks support here. A more defensible normalization candidate is Ghost BatchNorm, but its extra per-group statistics alter the successful SAM/EMA state dynamics and need careful feasibility and BatchNorm-buffer semantics.

## Key Techniques
- Ghost BatchNorm partitions a large batch into smaller statistic groups to inject normalization noise.
- Gamma/beta weight decay is architecture-sensitive and can be beneficial in residual networks.
- Inference-statistic corrections target train/eval discrepancy but would modify evaluation semantics and are out of scope for this frozen goal.
