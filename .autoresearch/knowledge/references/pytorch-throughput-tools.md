# PyTorch Throughput Tools for CNN Training

- Sources:
  - https://docs.pytorch.org/docs/stable/generated/torch.compile.html
  - https://docs.pytorch.org/docs/stable/notes/amp_examples.html
  - https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html

## Key Takeaway

`torch.compile`, AMP autocast, and channels-last memory format are official PyTorch mechanisms for improving model execution throughput. For this CIFAR-10 repo, they can be tested without changing model architecture, data semantics, loss target, or the evaluation harness.

## Use In This Project

Use them as a baseline-preserving throughput experiment: enable cuDNN benchmarking, convert the model and inputs to channels-last, use CUDA autocast with BF16 around forward/loss, and try `torch.compile` with a fallback path if compilation fails during checks.
