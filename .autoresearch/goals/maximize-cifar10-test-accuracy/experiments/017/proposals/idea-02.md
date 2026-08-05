# Proposal idea-02: Anti-aliased (BlurPool) downsampling — shift-invariance inductive bias

## Core change (train.py only)
Replace the stride-2 `nn.MaxPool2d(2)` downsamplers in layer1/layer2/layer3 with **MaxBlurPool**: `MaxPool2d(kernel=2, stride=1)` (dense max, no subsample) followed by a fixed (non-learnable) depthwise blur conv with stride 2 using a binomial kernel (e.g. 3×3 [1,2,1]⊗[1,2,1]/16). Anti-aliases before subsampling, restoring approximate shift-equivariance lost by naive strided pooling (Zhang, "Making Convolutional Networks Shift-Invariant Again", ICML 2019). Optionally also the final `MaxPool2d(4)`. Env `BLURPOOL` (0=baseline MaxPool, 1=on). Blur kernel registered as a frozen buffer (requires_grad=False, excluded from optimizer like the whitening conv).

## Why it targets the limiter
The limiter is the ~96.3–96.5 generalization ceiling, and the dominant strategic read (project-insights High, EXP-014) is that the productive move is a DIFFERENT mechanism, not capacity/optimizer/aug (all saturated). BlurPool is a textbook **inductive-bias** change — it improves test-time generalization via shift-invariance, NOT capacity (adds ~0 learnable params) — so it sidesteps the capacity-saturation verdict (EXP-014) and the regularization-axis saturation (EXP-008/011/012/015). Documented +0.5–1.0pp on CIFAR/ImageNet ResNets at matched epochs.

## Throughput discipline (the #1 failure-mode gate)
Per project-insights High, any per-step cost trades against epochs at the 300s budget. BlurPool adds a depthwise blur conv at each pool site; depthwise 3×3 is cheap, but the largest (32→16 on 128ch via dense maxpool then blur) must be measured. MITIGATION ladder: (a) measure num_epochs in a throughput pre-smoke; (b) if >~8% slowdown, use a lighter 2-tap [1,1] / `Rect-2` blur or apply BlurPool only at the later, small-spatial pools (8→4, 16→8) where shift-aliasing accumulates and cost is tiny; (c) compose with the EXP-014 torch.compile recipe (banked +12% throughput) to fully absorb the cost. Reject a cell with num_epochs <142 as under-anneal-confounded.

## Design — SAME-SESSION multi-cell
- c0: `BLURPOOL=0` (standard MaxPool control ~150ep).
- cA: `BLURPOOL=1` all three stride-2 pools, binomial-3 kernel.
- cB: `BLURPOOL=1` later pools only (or lighter kernel) — the throughput-safe variant.
`CUDA_VISIBLE_DEVICES=1 timeout 600`, nvidia-smi logged, num_epochs cross-checked.

## Verification
- Best BlurPool cell ≥ **96.48** AND > same-session c0 by >0.1pp.
- num_epochs ≥142 (throughput gate); ep25 sane; fully annealed.
- Integrity: train.py-only; prepare.py byte-unchanged; blur conv frozen & optimizer-excluded; seed 42.
- ON A WIN: bake BlurPool as default.

## Hypothesis
Anti-aliased downsampling adds shift-invariance the strided MaxPool lacks, improving test generalization by ≥0.1pp over the same-session control and clearing 96.48 at matched ~150 epochs — IF the blur's per-step cost stays small enough to keep epochs ≥142. If it ties at healthy epochs, shift-aliasing is not the generalization bottleneck for this whitened net; if it loses via epochs, retry the later-pools-only/compile-funded variant.

## Effort: medium (new module + frozen-kernel wiring + throughput smoke). Risk: (a) per-step cost cuts epochs → under-anneal (mitigated by lighter kernel/later-only/compile); (b) gain may not transfer at 150ep near ceiling like other levers; (c) interaction with the existing whitening front-end / MaxPool(4) head.
## Sources: Zhang ICML 2019 (arXiv:1904.11486, anti-aliased CNNs); knowledge/references/fast-cifar10-recipes.md; project-insights High (generalization ceiling, under-anneal gate, FLOP≠wall EXP-005); EXP-014 compile recipe (knowledge/references/torch-compile-throughput.md).
