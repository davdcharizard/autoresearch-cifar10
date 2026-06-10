# Gradient Centralization (Yong, Huang, Hua & Zhang — ECCV 2020)

**arXiv**: 2004.01461

## What it is
A one-line modification to the optimizer step: before `optimizer.step()`, centralize each weight tensor's gradient by subtracting its mean computed over the INPUT/fan-in dimensions, per output unit.
- Conv weight `(C_out, C_in, kH, kW)`: subtract the mean over (C_in, kH, kW) per output channel → `g -= g.mean(dim=(1,2,3), keepdim=True)`.
- Linear weight `(out, in)`: subtract the per-row (per-output) mean → `g -= g.mean(dim=1, keepdim=True)`.
- Leave 1-D params (BN γ/β, biases) untouched.
- Generic: for any `g.ndim > 1`, `g.add_(-g.mean(dim=tuple(range(1, g.ndim)), keepdim=True))`.

Drop-in between `loss.backward()` and `optimizer.step()`. Core torch ops only — no new dependency.

## Mechanism (why it should help)
GC projects the gradient onto a hyperplane with zero mean over fan-in. The authors show two coupled effects:
1. **Weight-space regularization**: it constrains the weight space (the centralized gradient keeps the sum of each output unit's weights on a constraint surface), which smooths/regularizes the loss landscape.
2. **Gradient standardization**: it bounds/standardizes the gradient, improving the Lipschitz properties → accelerates and stabilizes training.

Crucially it claims to BOTH accelerate convergence AND improve generalization — distinct from a pure regularizer (which costs convergence) or a pure averaging/polish trick (which moves loss not top-1).

## Reported results
Small consistent top-1 gains across CIFAR/ImageNet ResNets (~+0.2–0.6%), faster + more stable training, enlarged stable-LR range. Largest gains on deeper/larger nets.

## Relevance to this project
- Opens the **optimizer/gradient-dynamics class** — the one axis never explored here (only WD ever swept, EXP-005). Distinct from every closed axis: not capacity (no epoch wall), not data aug, not weight-averaging/polish, not a scalar knob.
- **EXP-030 result**: full-net GC → **96.21** (tie with baseline 96.22) with final_test_loss **0.1934 < 0.195** (better) — BUT the naive per-param Python for-loop (23 tensors, ~46 tiny kernel launches/step) cost ~1ms/step → dt 8→9ms → epochs 91→88. Read as a near-miss (tied + better loss at a 3-epoch handicap).
- **EXP-031 result (RESOLVED)**: fixed the throughput — hoist the weight-param list once + `torch.compile` the out-of-place centralization (default mode; reassign `p.grad`) → **dt 8ms / 91 ep = baseline (fix confirmed)** → **96.14 (−0.08pp, within ±0.2pp noise) with loss 0.1894 < 0.195**. The fair, throughput-neutral test shows GC improves test LOSS but NOT top-1; the EXP-030 96.21 was the noise-favorable tail of a top-1 null. **GC is a convergence-POLISH lever here (loss↓, top-1 flat), like EMA/SWA/Bag-of-Tricks — NOT a top-1 gain on this shallow well-tuned net.**
- **Implementation note (validated)**: the per-parameter Python loop was the throughput bottleneck, not GC's math; `torch.compile`+hoist fully restored 8ms/91ep (reduce-overhead is invalid because `set_to_none=True` reallocates grads each step). Useful pattern for any per-step per-param op.
- **Standing value**: GC remains a free, validated way to LOWER test loss / improve calibration — relevant if a future goal targets loss or ECE rather than top-1. Same-class untried members (AGC, gradient-noise, LARS/LAMB) are now LOWER-confidence for top-1 (GC, the best-documented member, gave a top-1 null).
