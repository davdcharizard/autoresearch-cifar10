# AdamW — Decoupled Weight Decay Regularization

- **Source**: Loshchilov & Hutter, ICLR 2019 (arXiv:1711.05101).
- **Used in**: EXP-043 (first optimizer-FAMILY change on this goal).

## Core idea
Adam couples L2 weight decay into the adaptive gradient (so decay is scaled by the per-parameter
second-moment estimate, weakening it for large-gradient params). AdamW DECOUPLES weight decay: apply
`θ ← θ − lr·(m̂/(√v̂+ε)) − lr·wd·θ` separately. This restores proper regularization and closes much of
Adam's generalization gap vs SGD on image models. Decoupled wd is ~100–1000× larger than SGD's coupled wd
(typical 0.01–0.1); peak lr is ~100× smaller than SGD (typical 1e-3–3e-3 from scratch on small models).

## Practical notes
- Warmup matters for early-step stability (the second-moment estimate is noisy at step 0).
- Two moment buffers per param (≈2× param state) — negligible VRAM here (small model).
- Throughput: `_foreach_`-fused update is sub-ms on 4.3M params → dt-neutral (EXP-043: steady 8ms).

## Outcome on this project (EXP-043)
REGRESSION. AdamW (lr 2e-3, wd 0.05, betas 0.9/0.999, same cosine+warmup/TA+Cutout/LS recipe) trained
STABLY (no divergence, smooth convergence, flat 95.7–95.9 tail) and throughput-NEUTRALLY (8ms/91ep), but
best_test_acc **95.87 (−0.35pp)**, final_test_loss 0.2023 (> baseline 0.195) — the textbook adaptive-
optimizer generalization gap (a genuine deficit, not under-training). Tuned SGD+Nesterov (peak 0.2,
cosine-to-0, wd 1e-4) remains the better optimizer. **Optimizer axis (family + GC/SAM/PolyLoss grad/objective
mods) fully CLOSED here. Do NOT re-try AdamW LR/WD sweeps or other adaptive optimizers (LAMB/Lion/Adan).**
