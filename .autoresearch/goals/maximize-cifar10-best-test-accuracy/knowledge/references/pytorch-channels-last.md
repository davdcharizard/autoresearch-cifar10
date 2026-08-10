# PyTorch Channels-Last Memory Format

- **Source**: PyTorch official tutorial
- **URL**: https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html

## Key Contributions

- Channels-last changes physical strides while preserving logical NCHW tensor dimensions.
- PyTorch documents channels-last support for CUDA convolution and BatchNorm; converting both model weights and inputs allows the format to propagate through supported operators.
- Unsupported operators can force format conversions, so performance is shape/operator/dtype dependent and must be measured end to end.

## Relevance

The accepted CIFAR workload spends 75.46% of counted step time in convolution/BN backward, making layout one of the few systems levers that can plausibly increase exposure without changing batch noise, model capacity, or data. Official examples emphasize the largest GPU gains with reduced precision, however; this project is fixed FP32/default-TF32 and uses tiny 32x32 activations, so no speedup can be assumed.

## Local Requirements

- Construct/initialize the model exactly as accepted before converting it to `torch.channels_last`, preserving parameter values and RNG state.
- Transfer each 4-D training input with explicit `memory_format=torch.channels_last` inside the counted step.
- Verify propagation and absence of repeated hidden conversions with strides/profiler evidence, then use alternating fresh-process full-step timing.
- Treat changed cuDNN kernel numerics as an experimental effect; use exact-corpus safety and one fixed-seed production run rather than claiming bitwise equivalence.
