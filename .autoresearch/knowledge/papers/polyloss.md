# PolyLoss — A Polynomial Expansion Perspective of Classification Loss Functions

- **Source**: Leng, Tan, Liu, et al., ICLR 2022. arXiv:2204.12511.
- **Used in**: EXP-041 (first objective-polynomial-shape probe on this goal).

## Core idea
Cross-entropy can be Taylor-expanded in `(1 − p_t)` (p_t = softmax prob of the true class):
`CE = Σ_{j≥1} (1/j)·(1 − p_t)^j`. PolyLoss treats the polynomial coefficients as tunable.
**Poly-1** perturbs only the leading term:

```
L_Poly1 = CE(y, p) + ε·(1 − p_t)
```

- `ε > 0` increases the gradient magnitude on hard/low-`p_t` examples → behaves as a mild
  convergence accelerator / hard-example emphasizer (focal-loss is the opposite-sign / higher-order cousin).
- `ε < 0` down-weights hard examples (more like label smoothing's effect).
- Subsumes CE (ε=0) and is related to focal loss.

## Practical notes
- **ε is dataset/task-dependent and must be tuned.** Paper's ImageNet ResNet optimum ≈ +1 to +2; other
  tasks (detection/segmentation/3D) use different values. No universal CIFAR value published.
- Compute-free: one `softmax().gather()` on logits already computed; no new params/deps; convergence-neutral.
- Composes with label smoothing, though LS (softens targets) and Poly-1 ε>0 (sharpens hard-example
  gradients) push in partially opposite directions → may partially cancel.
- Implementation (keep existing label smoothing):
  ```python
  ce = F.cross_entropy(logits, targets, label_smoothing=LS)
  pt = F.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
  loss = ce + eps * (1.0 - pt).mean()
  ```

## Relevance to this project (CIFAR-10 k=4 WRN, 300s budget)
The objective's polynomial shape is a compute-free, convergence-neutral, top-1-affecting lever that dodges
all three established plateau walls (compute / polish-vs-top1 / regularizer-underfit). ε>0 aligns with the
convergence-bound hypothesis. Sibling objective tweaks were null here (label-smoothing-down EXP-023, cosine
head EXP-039), so the honest expectation is modest-or-null, but it is a distinct, cited mechanism.

## Outcome on this project (EXP-041)
NULL on top-1, pure polish. ε=1.0 on a clean throughput-neutral 90-ep run: eval CE loss crashed to **0.158**
(project-lowest, ~29% below baseline 0.195) but best_test_acc **96.11 (−0.11pp, within noise)**. ε>0 pushes
`p_t→1` (confidence/calibration), not the decision boundary → fits the polish-vs-top1 wall. Objective/
loss-shape axis CLOSED here (with LS-down EXP-023 and cosine head EXP-039). Do NOT re-try ε sweeps.
