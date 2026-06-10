# PyTorch EMA Weight Averaging

- Source: https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html
- Relevance: Low-overhead model weight averaging for training recipes.

## Key Takeaway

PyTorch's `torch.optim.swa_utils.AveragedModel` supports both SWA and EMA through configurable averaging functions, including `get_ema_multi_avg_fn`. It can maintain a second averaged copy of the model parameters and buffers during training without changing the optimizer, data pipeline, or evaluation harness.

## Use In This Project

Test an EMA copy of the successful FP32 throughput ResNet-20 as a low-risk way to improve the evaluated weights. Use `use_buffers=True` so BatchNorm running statistics are averaged with parameters, update the EMA after optimizer steps, and evaluate the EMA model once per epoch to preserve the validation-cadence constraint.
