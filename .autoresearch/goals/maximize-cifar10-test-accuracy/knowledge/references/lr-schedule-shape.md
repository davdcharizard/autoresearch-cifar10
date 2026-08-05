# LR schedule shape: cosine decay vs cyclic/linear one-cycle (CIFAR)

Standing reference for the anneal-SHAPE lever (chosen EXP-020). Our recipe uses a TIME-based **linear** triangular one-cycle (EXP-001): ramp 0→PEAK over PCT_START=0.15, then **linear** decay PEAK→0. The decay shape was never tuned.

## Key external finding (the evidence for trying cosine)
- **MosaicML LR-schedule benchmark** (https://cameronrwolfe.substack.com/p/the-best-learning-rate-schedules): cyclic/one-cycle schedules "do not necessarily lead to improved accuracy when compared to cosine decay … in many instances the cyclic tradeoff curve underestimated the standard [cosine] tradeoff curve by a margin of 0.5% validation accuracy," and "results held for CIFAR-10 as well." → at a budget where the net is anneal-saturated, cosine decay may beat linear/cyclic.
- **fastai one-cycle uses COSINE annealing** for the curve, not linear — our linear-triangular decay is the less-standard variant.
- **SGDR** (Loshchilov & Hutter, arXiv:1608.03983): cosine annealing (optionally with warm restarts for snapshot ensembling) is the standard research-grade CIFAR schedule; restarts lower CIFAR-10 error but need a multi-cycle budget and ≥1 model snapshot (not viable under our 1-eval/epoch, single-model, 300s-budget constraint → use plain cosine decay, no restarts).
- **Bag of Tricks** (He et al. CVPR 2019, arXiv:1812.01187): cosine LR is a recognized accuracy trick.

## Mechanism (why shape matters here)
EXP-001 established most accuracy lands in the **low-LR tail** of a completing one-cycle (project-insights Medium). Cosine HOLDS LR HIGHER early after warmup, crosses linear near mid-decay, then spends MORE time at very low LR — a different exploration/anneal balance that selects a different (plausibly flatter) minimum. Both linear and cosine finish at exactly 0 (full anneal preserved — no under-anneal).

## Implementation on THIS harness (load-bearing)
- Change ONLY the post-warmup branch (train.py:286-290): `q=(progress−PCT_START)/(1−PCT_START)`; cosine `lr=PEAK·0.5·(1+cos(πq))`. Warmup/peak/EMA-gate(0.15)/TTA-gate(0.8) all key on `progress`, NOT LR → untouched.
- Exactly THROUGHPUT-FREE (a scalar formula change; num_epochs identical) → zero under-anneal risk, the #1 failure mode on this goal.
- `SCHEDULE=tri` must reproduce the baseline LR trace bit-for-bit (regression smoke: sampled progress, lr_tri, lr_cos + fraction of steps below LR thresholds).
- AVOID warm restarts (under-anneal risk at a fixed 300s budget); prefer plain cosine decay finishing at 0.

## Status on this goal
**EXP-020 NO-IMPROVEMENT (closed).** Cosine tied the linear-triangular incumbent. Two same-session pairs at a matched 150ep: session-1 cos 96.36 vs linear 96.04 = +0.32pp did NOT replicate — confirmation cos 96.39 vs linear 96.35 = **+0.04pp** (tie); cos never cleared 96.48. The +0.32 was a low-control-draw artifact (session-1 linear drew 96.04 vs 96.35 normally), the recurring host pattern. The honest prior held: the ~0.5% MosaicML figure (ResNet-50/longer-schedule) shrank to noise on this small heavily-augmented net at 150ep. Diagnostic: shorter cosine warmup (PCT_START=0.10) underperformed 0.15. The schedule-SHAPE axis is exhausted — do NOT re-test cosine/cyclic/linear permutations. Only faint residual: cosine + higher PEAK_LR (cosine holds LR higher mid-run) — low confidence, likely still ceiling-bound. See experiments/020/04-analysis.md.

## Sources
- MosaicML/Cameron Wolfe LR schedules (URL above); SGDR arXiv:1608.03983; Bag of Tricks arXiv:1812.01187.
