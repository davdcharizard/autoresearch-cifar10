# Proposal: Time-Aware Pre-Activation WRN-16-4 with Restrained Stochastic Depth

## Summary

Replace the 269,722-parameter post-activation ResNet-20 with a pre-activation
Wide ResNet of depth 16 and width multiplier 4 (approximately 2.75M
parameters), then make the wider network practical on the H20 with batch 256,
BF16 autocast, and channels-last tensors. Regularize only the residual branches
with a small, depth-dependent stochastic-depth rate (maximum 0.08), active
early and annealed to zero during the final quarter of the 300-second training
budget.

This is an architecture-first experiment. Keep random crop, horizontal flip,
the evaluator-compatible input normalization, and clean cross-entropy. In
particular, do not add Mixup/CutMix in the same run: the first question is
whether substantially more residual capacity can be optimized adequately in
the fixed wall-clock window.

## Why This Targets the Current Limiter

The baseline reaches only 91.51% while using 269,722 parameters and 330.1 MiB
peak VRAM. The H20 therefore has enormous unused memory headroom, and the model
is so narrow (16/32/64 channels) that its many small convolutions are unlikely
to use the accelerator efficiently. The likely limiter is representational
capacity and hardware utilization rather than memory. A shallow-wide network
is preferable to a very deep network here: it adds channel capacity and
Tensor-Core-friendly dimensions while retaining only six residual blocks, so
kernel-launch and sequential-depth costs remain restrained.

The proposal also removes a schedule mismatch. The current milestones are at
steps 32,000 and 48,000 and `MAX_STEPS` is 64,000. A wider model will complete a
different number of steps in 300 seconds, so retaining those milestones would
leave its learning rate schedule dependent on throughput. A cosine schedule
indexed by measured training time ensures that the wider model still receives
the full high-to-low learning-rate trajectory.

## Mechanism

### Pre-activation wide residual representation

Use the CIFAR Wide ResNet depth convention `depth = 6n + 4`, with `n = 2` and
stage widths `[64, 128, 256]` after a 16-channel stem. Each residual unit is:

```text
x -> BN -> ReLU -> 3x3 conv -> BN -> ReLU -> 3x3 conv -> stochastic depth -> + shortcut
```

The first unit in stages 2 and 3 uses stride 2 in its first convolution. When
shape changes, apply a learned 1x1 projection to the pre-activated input. End
the network with BN, ReLU, global average pooling, and a 256-to-10 linear
classifier. There is no ReLU after residual addition. This gives six short
residual paths and roughly 2,748,890 trainable parameters, about 10.2x the
baseline size while remaining tiny relative to 98 GB of device memory.

Pre-activation gives every residual branch a normalized optimization path and
lets the identity addition remain unobstructed. Widening attacks the baseline's
capacity ceiling without the serial cost of a 50- or 100-layer model.

### Restrained stochastic residual survival

For block index `i` in `0..5`, assign a maximum drop probability
`p_i = 0.08 * (i + 1) / 6`. During training, multiply the block's residual
branch by a per-example mask of shape `[N, 1, 1, 1]`, sampled with survival
probability `1 - p_i`, and divide surviving branches by `1 - p_i`. The shortcut
is never dropped. At evaluation, return the residual branch unchanged.

This is deliberately a conservative, expectation-preserving stochastic-depth
variant, not a claim to reproduce the stronger ShakeDrop noise distribution.
At each training step, scale all `p_i` by:

```text
1.0                                  if time_progress <= 0.75
(1.0 - time_progress) / 0.25         otherwise
```

Thus the deepest block is dropped on at most 8% of examples, the mean rate is
only 4.67%, and all stochastic residual dropping disappears as optimization
settles. This supplies ensemble-like path diversity without the instability
risk of aggressive ShakeDrop in a six-block, short-horizon model.

### Throughput-aware optimization

Use the following exact initial settings:

| Setting | Value |
|---|---:|
| Architecture | PreAct WRN-16-4 |
| Batch size | 256 |
| Precision | CUDA BF16 autocast, no gradient scaler |
| Tensor layout | model and inputs in `channels_last` |
| Optimizer | SGD, momentum 0.9, Nesterov enabled |
| Peak learning rate | 0.20 |
| Weight decay | `5e-4` |
| Warmup | linear over the first 5 epochs |
| Decay | cosine to zero by measured training-time progress 1.0 |
| Loss | ordinary cross-entropy |
| Maximum stochastic-depth probability | 0.08 |
| Data augmentation | existing random crop + horizontal flip |

The peak learning rate preserves the baseline's 0.1 learning rate per 128
examples under linear batch scaling. Five-epoch warmup limits initial shocks
from the larger width and Nesterov momentum. Compute the per-step learning rate
before the forward pass as:

```python
progress = min(total_training_time / TIME_BUDGET_S, 1.0)
warmup = min((step + 1) / (5 * len(train_loader)), 1.0)
lr = 0.20 * warmup * 0.5 * (1.0 + math.cos(math.pi * progress))
```

Set this value on every optimizer parameter group. Remove the fixed
`MultiStepLR` and make the wall-clock budget the primary stopping condition;
retain a very high emergency `MAX_STEPS` only if desired. Use
`optimizer.zero_grad(set_to_none=True)`, move both the model and each image
batch to channels-last format, and wrap forward plus loss in
`torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. Keep parameters,
optimizer state, BatchNorm statistics, and evaluation in FP32. Enable
`torch.backends.cudnn.benchmark = True`; do not introduce `torch.compile` in
this first run because compilation/recompilation adds a separate failure mode.

## Exact `train.py` Implementation

1. Add `import math`; replace `BasicBlock` and `ResNet` with
   `DropPath`, `PreActWideBlock`, and `PreActWideResNet` classes. Build exactly
   two blocks per stage at widths 64, 128, and 256.
2. Implement `DropPath.forward` using `torch.rand` on `x.device`, a mask with
   shape `(x.shape[0], 1, 1, 1)`, and inverse-survival scaling. Store each
   block's base probability so the training loop can apply the late annealing
   without reconstructing modules.
3. Initialize convolution weights with Kaiming normal initialization, BatchNorm
   weights to one and biases to zero, and the classifier bias to zero. Do not
   zero-initialize the pre-convolution BN scale, because ReLU at an all-zero
   residual input can delay learning in this implementation.
4. Instantiate the model on physical GPU 0 only through the required launch
   environment, then call
   `model.to(device, memory_format=torch.channels_last)`. Set batch size 256 and
   keep `drop_last=True`.
5. Replace fixed step milestones with the warmup-plus-time-cosine formula
   above. Set the current drop-path scale from the same `progress` value before
   each forward pass.
6. In the batch loop, transfer images with `non_blocking=True` and then call
   `inputs = inputs.contiguous(memory_format=torch.channels_last)`. Run forward
   and cross-entropy inside BF16 autocast; run backward and SGD normally.
7. Preserve the current timing boundary, synchronization, once-per-epoch
   evaluation, best-accuracy tracking, and complete final summary. Do not edit
   `prepare.py` or special-case evaluation inputs.

## Evidence

- The measured baseline in
  `.tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv`
  is 91.51%. Its 269,722 parameters and 330.1 MiB peak allocation show that
  substantially more capacity is feasible under the task's soft memory
  constraint.
- `.tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/001/papers/shakedrop.md`
  reports successful stochastic residual-branch scaling across Wide ResNet and
  related residual families, while warning that stronger published settings
  often rely on deeper models and longer schedules. That supports using the
  mechanism but caps its strength at 0.08 here.
- `.tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/001/papers/time-matters-regularization.md`
  finds that regularization has its greatest effect in the early critical
  period and that removing it late can retain or improve generalization. That
  directly motivates holding stochastic depth early and annealing it away in
  the last quarter.
- `.tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/001/papers/mixed-sample-analysis.md`
  and `regmixup.md` support mixed-sample regularization, but RegMixup adds an
  extra clean/mixed objective and therefore extra forward work. Deferring these
  methods keeps this experiment focused and preserves the widest possible
  architecture within the time budget. They remain natural follow-ups after a
  strong architecture base exists.

## Expected Impact

The primary prediction is `best_test_acc` of 93.0-95.0%, a gain of roughly
1.5-3.5 percentage points over 91.51% and comfortably above the required
91.61% success threshold. Most of the gain should come from the 10x capacity
increase and better residual optimization; stochastic depth is expected to
contribute a smaller 0.1-0.4 point improvement or to protect against the wider
model's overfitting.

The estimate assumes that BF16 channels-last execution permits at least 20,000
optimizer steps at batch 256 (more than 5.1M augmented sample presentations,
or about 102 CIFAR-10 epochs). If throughput is materially below that level,
optimization exposure rather than capacity becomes the limiter and the
accuracy range should be revised downward.

## Throughput and Failure Risks

- **Undertraining risk (medium):** WRN-16-4 has about an order of magnitude more
  parameters and several times the convolutional work of ResNet-20. The H20 may
  execute the wider Tensor-Core-friendly kernels efficiently, but this must be
  measured. A median synchronized step time above 15 ms projects fewer than
  20,000 steps and is the explicit signal to reduce width.
- **Regularization risk (low-medium):** six blocks may not need path dropping,
  and strong residual noise can slow early fitting. The maximum 0.08 rate and
  late annealing bound this risk. If training loss stays visibly high, the
  no-drop-path ablation should precede other changes.
- **BF16 numerical risk (low):** BF16 has ample exponent range and does not need
  loss scaling, but the final logits should be accepted by FP32 evaluation.
  If a CUDA operator fails under autocast, disable autocast without changing
  the architecture rather than adding a dependency.
- **Batch-size generalization risk (low-medium):** batch 256 halves the number
  of optimizer updates per sample relative to batch 128. It is a compromise
  between GPU utilization and update frequency; do not jump to 512 in the
  initial run.
- **Confounding risk (controlled):** width, pre-activation, schedule, precision,
  and restrained path regularization change together. They are a coherent
  time-budgeted architecture package, but follow-up ablations are needed to
  assign causality.

## Ranked Ablations and Contingencies

1. **No stochastic depth:** set all drop probabilities to zero while preserving
   WRN-16-4 and the time-based schedule. Run this first if the full proposal
   underfits or misses 91.61%; it distinguishes architecture failure from
   over-regularization.
2. **WRN-16-3 throughput fallback:** use stage widths `[48, 96, 192]` if the
   first 50 post-warmup steps have median synchronized time above 15 ms. Keep
   all other settings fixed. This retains the shallow-wide representation with
   about 1.55M parameters and should restore optimizer exposure.
3. **Drop-rate sensitivity:** compare maximum probabilities 0.04 and 0.08 only
   after WRN-16-4 itself clearly improves the baseline. Choose 0.04 when late
   training loss remains high; choose 0.08 when training accuracy saturates but
   test accuracy plateaus.
4. **Width-up follow-up:** consider WRN-16-6 only if the initial model exceeds
   91.61% and still completes more than roughly 35,000 steps. This is a later
   capacity experiment, not an automatic modification of experiment 001.
5. **Mixed-sample follow-up:** after establishing the architecture, test a
   low-overhead CutMix or Mixup variant separately. Do not use RegMixup's extra
   clean forward until its wall-clock cost is measured.

## Verification

1. Before execution, verify GPU identity with `nvidia-smi -i 0` and launch only
   with `CUDA_VISIBLE_DEVICES=0`.
2. Run under the required outer limit:
   `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`.
3. Confirm the first 50 stable training steps imply median step time at or below
   15 ms. This is a diagnostic, not a reason to interrupt a healthy first run;
   use it to select the next ablation.
4. Confirm training ends because measured training time reaches approximately
   300 seconds, never because the old 64,000-step schedule silently truncates
   it, and that validation occurs no more than once per completed epoch.
5. Extract `best_test_acc` and `peak_vram_mb`, then parse the complete final
   summary. Success requires `best_test_acc >= 91.61%`, no crash, total runtime
   below 10 minutes, and a complete metric summary.
6. Verify the implementation diff changes only `train.py`; keep seed 42 fixed
   and do not rerun with alternate seeds to select a favorable result.

