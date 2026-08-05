# Proposal idea-03: Squeeze-Excitation channel attention

## Core change (train.py only)
Add a Squeeze-Excitation (SE) block (Hu et al., CVPR 2018, arXiv:1709.01507) to the residual branches: after `c2(c1(x))` in `Residual`/`GatedResidual`, recalibrate channels via global-avg-pool → FC(C→C/r) → ReLU → FC(C/r→C) → sigmoid → channel-wise multiply. Reduction r=8 or 16. Env `SE_REDUCTION` (0=off=baseline, else r). Identity-friendly init: initialize the second FC so the initial gate ≈1 (near-identity at init, preserving the proven recipe — analogous to the ReZero discipline in EXP-004).

## Why it targets the limiter
The limiter is the ~96.3–96.5 generalization ceiling, where capacity (EXP-007/014), optimizer (EXP-009/010), input-aug (EXP-008/011/015), and reg-scalars (EXP-012) are all saturated. SE is a DIFFERENT mechanism — content-adaptive per-channel feature recalibration (a lightweight attention) — that adds representational flexibility with almost no parameters/FLOPs, not raw capacity. It is among the highest value-per-param CIFAR ResNet additions in the literature (+0.5–1.5pp at matched epochs). Composes with the existing whitening + ReZero + EMA stack.

## Throughput discipline
SE is cheap (a global pool + two small FCs per block; r=8 keeps the FCs tiny) and runs at the small-spatial residual stages (16×16/128, 8×8/256, 4×4/512), so per-step cost is modest. Still gated per project-insights High: measure num_epochs in a pre-smoke; reject <142 as under-anneal. If costly, apply SE only to layer2/layer3 blocks.

## Design — SAME-SESSION multi-cell
- c0: `SE_REDUCTION=0` (control ~150ep).
- cA: `SE_REDUCTION=16` (light) on all residual blocks.
- cB: `SE_REDUCTION=8` (stronger) or layer2+3-only — second operating point.
Near-identity SE init so cA/cB start bit-close to the proven net (single-variable test). `CUDA_VISIBLE_DEVICES=1 timeout 600`, nvidia-smi logged.

## Verification
- Best SE cell ≥ **96.48** AND > same-session c0 by >0.1pp.
- num_epochs ≥142; ep25 sane; fully annealed; SE gate not collapsed (sanity: gate std >0).
- Integrity: train.py-only; prepare.py byte-unchanged; seed 42; SE adds params but within "architecture change" scope (no fixed-param constraint in goal file).
- ON A WIN: bake SE_REDUCTION as default.

## Hypothesis
SE channel recalibration on the residual branches adds content-adaptive representational power orthogonal to the saturated capacity/aug/optimizer axes and lifts best_test_acc ≥0.1pp over the same-session control, clearing 96.48 at matched ~150 epochs. If it ties at healthy epochs, channel-attention does not move this whitened net's ceiling (it may already be channel-efficient via whitening); if it loses via epochs, retry layer2+3-only.

## Effort: low-medium (small module + near-identity init + throughput smoke). Risk: (a) per-step cost cuts epochs (mitigated: SE is cheap / later-only); (b) sigmoid gate interaction with ReZero α and EMA; (c) gain may not transfer at 150ep near ceiling; (d) adds params (acceptable per goal scope, but is a capacity-adjacent confound — mitigated by SE being recalibration not width).
## Sources: Hu et al. CVPR 2018 (arXiv:1709.01507, SENet); knowledge/references/rezero-identity-init.md (near-identity init discipline, EXP-004); project-insights High (generalization ceiling, under-anneal gate).
