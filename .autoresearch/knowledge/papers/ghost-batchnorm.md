# Ghost BatchNorm (Hoffer, Hubara & Soudry, "Train longer, generalize better", NeurIPS 2017)

## Core idea
Compute BatchNorm statistics over small disjoint **ghost** sub-batches of the full minibatch instead of the whole batch. The smaller per-ghost sample injects statistical noise into the normalization, which acts as an **implicit regularizer** (analogous to the noise that helps small-batch SGD generalize). Originally introduced to recover the generalization gap incurred by **large-batch** training.

## Mechanism
- Split batch of N into `s` ghost groups of size `N/s` (ghost size). Normalize each group with **its own** mean/var.
- Running statistics (used at **eval**, unchanged from standard BN) accumulate toward the population estimate; eval is byte-identical to standard BN (single normalization with population running stats).
- Benefit is strongest at large batch (>256). At small batch (e.g. 128) the added noise is milder and may not help — the large-batch gap it targets is already narrow.

## Implementation notes for THIS project (k=4 ResNet-20, bf16, channels_last, torch.compile reduce-overhead)
- **Static shapes are mandatory for CUDA-graph safety.** `BATCH_SIZE=128` + `drop_last=True` ⇒ every training batch is exactly 128 ⇒ a fixed ghost split (e.g. s=4, ghost 32) keeps shapes static ⇒ reduce-overhead CUDA graph is preserved (the EXP-042 dt-confound is from *data-dependent* control flow, NOT static reshapes).
- **channels_last view trap**: the classic "fold groups into channels" trick `x.view(N/s, C*s, H, W)` is INVALID on a channels_last tensor (incompatible strides → forces an NCHW copy → dt cost). Instead split only the OUTER batch dim: `x.view(s, N/s, C, H, W)` is always a valid view (it just factors dim-0's stride) and preserves the channels_last per-sample layout. Normalize per group with manual mean/var over `(group, H, W)`.
- **Running-stat update MUST be in-place** (`running_mean.mul_(...).add_(...)`), NOT reassignment to a new tensor — reassignment reallocates the buffer each step and breaks CUDA-graph capture (the EXP-031 `set_to_none` lesson). Updating from the FULL-batch stats (vs ghost-averaged) gives a cleaner population estimate for eval.
- **`self.training` branch is safe** here because the compiled handle only ever runs in train mode (eval uses the eager `model`), so the flag is a compile-time constant — no runtime toggle, no recompile.
- **Always verify dt** stays ~8ms after the swap (`tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c`); a rise means a graph break / layout copy → result is throughput-confounded.

## Relevance
The NORMALIZATION axis is the one accuracy axis untouched across EXP-000..046 on this goal. GhostBN is a strong, mechanistically-distinct regularizer (the class that produced this project's only plateau-breaker, TrivialAugment EXP-012) and is throughput-neutral by construction (dodges the epoch wall). Used in EXP-047.
