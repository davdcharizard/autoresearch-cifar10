# Channels-Last Memory Format for Convolutional Networks

## Sources

- PyTorch, "Channels Last Memory Format in PyTorch": https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html
- PyTorch, `convert_conv2d_weight_memory_format`: https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.convert_conv2d_weight_memory_format.html
- NVIDIA, "Convolutional Layers User's Guide": https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
- PyTorch, "Performance Tuning Guide": https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html

## Distillation

- PyTorch's channels-last format preserves a four-dimensional tensor's logical NCHW shape while changing its physical strides to NHWC. A model and its inputs can be converted explicitly with `memory_format=torch.channels_last`.
- Convolution and BatchNorm support channels-last propagation, but an unsupported or layout-insensitive operator can introduce a permutation or return a tensor in another format. A candidate therefore needs per-stage layout checks rather than assuming end-to-end propagation.
- NVIDIA recommends NHWC because Tensor Core convolution kernels operate most efficiently in that layout; NCHW can require transpose work. Modern cuDNN can automatically pad an RGB input's three channels, while this model's later width-2 channels (32, 64, and 128) already satisfy common alignment requirements.
- PyTorch's performance guidance presents channels-last primarily together with reduced precision or AMP. None of the sources establishes a speedup for this repository's FP32 ResNet-20 workload, so local paired timing is decisive.
- PyTorch also exposes a selective utility that converts only Conv2d weights. Full-model conversion is simpler but may change parameter layouts beyond convolutions; the implementation choice must be explicit and checked against optimizer and BatchNorm behavior.

## Relevance to EXP-021

The accepted system profile attributes 97.57% of measured GPU-stage time to model forward plus backward, so a convolution-layout speedup attacks the actual fixed-time bottleneck. The intervention does not add an accuracy mechanism directly: it can help only if it increases useful optimizer exposure within 300 counted seconds without changing the model, data, loss, schedule, or evaluation opportunity count. Because the strongest documented gains concern AMP and this goal rejected the tested BF16 operating point, EXP-021 must use FP32, require numerical/layout correctness, and pass fresh paired timing with a meaningful margin before production.
