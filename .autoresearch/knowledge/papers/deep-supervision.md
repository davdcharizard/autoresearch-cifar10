# Deep Supervision / Auxiliary Classifiers

- **Sources**: Deeply-Supervised Nets (Lee, Xie, Gallagher, Zhang, Tu — AISTATS 2015); GoogLeNet
  auxiliary classifiers (Szegedy et al., "Going Deeper with Convolutions", CVPR 2015).
- **Used in**: EXP-042 (first deep-supervision probe on this goal).

## Core idea
Attach lightweight classifier head(s) to intermediate layer(s) of the network and add their
cross-entropy to the training loss: `L = CE_main + Σ λ_i · CE_aux_i`. The aux heads inject direct
gradient signal into mid/early layers, easing signal propagation and (claimed) making intermediate
features more discriminative. Aux heads are **discarded at inference** — only the main head is scored,
so it is NOT an eval-protocol change (unlike TTA). GoogLeNet used weight 0.3 on two aux heads.

## Practical notes
- The benefit is **depth-driven**: it addresses vanishing-gradient / signal-propagation in VERY deep
  nets (20–100+ layers, pre-BN-era or very deep post-BN). On shallow nets that already train cleanly
  with BatchNorm + warmup, there is no propagation problem to fix.
- Decaying the aux weight λ→0 over training keeps the final objective pure-main (avoids distorting the
  evaluated head at convergence). Compute: aux head fwd/bwd is cheap; the extra backward through the
  early layers is real but, when compiled with a STABLE forward output structure, fuses near-free here.
- **torch.compile gotcha (EXP-042)**: returning a train-only aux output via an `if self.training` branch
  (tuple vs tensor) breaks `reduce-overhead` CUDA-graph capture → dt doubles. Use a separate always-tuple
  `forward_train` compiled for training; keep `forward` single-tensor for the eager eval handle.

## Outcome on this project (EXP-042)
NULL→mild REGRESSION on top-1. Aux layer2 classifier, λ=0.3→0, throughput-neutral (8ms / 90 ep, fair):
best_test_acc **95.91 (−0.31pp)**, final_test_loss 0.2026 (> baseline 0.195 — not even a loss win). On
the shallow 9-block ResNet-20 the aux gradient pulls layer2 features toward premature class-
discriminability, mildly fighting the tuned coarse-to-fine hierarchy — a gentler version of the
multi-scale-head regression (EXP-032, −1.5pp). **Intermediate-feature-routing family CLOSED here (both
input-concat and aux-supervision regress). Do NOT re-try aux heads / λ sweeps / earlier aux points.**
