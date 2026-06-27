# SWA — Stochastic Weight Averaging (Izmailov et al. 2018) — measured on this project

**Source**: "Averaging Weights Leads to Wider Optima and Better Generalization" (UAI 2018, arXiv 1803.05407); `torch.optim.swa_utils` (core PyTorch: `AveragedModel`, `update_bn`). Measured in-project: EXP-032 (cf. EXP-011 EMA, EXP-029 BN-stats law).

## Technique
After annealing to a modest CONSTANT LR, equal-average end-of-cycle SGD iterates; the average sits nearer the flat-basin center than any single iterate. Paper gains +0.2–0.6 on CIFAR-10 ResNets/WRNs at fixed epochs. Implementation requirements: (1) constant tail LR (at LR→0 iterates freeze and the average degenerates); (2) BN running stats MUST be re-estimated for the averaged weights via a forward pass over the AUGMENTED training loader (`update_bn`).

## Measured result on CIFAR-10 / WRN-20-4x / H20, 300s charged budget (EXP-032)
- Config: cosine frozen at 85% of budget (lr ≈ 0.030 ≈ canonical swa_lr scaled), ~21 end-of-epoch snapshots, full-loader `update_bn` (+~2s uncharged wall each) before every tail eval.
- Mechanics all confirmed healthy: first SWA eval (n=1 = BN re-est only) +0.71 over the last raw eval; test_loss fell monotonically to 0.1756 — strictly better than the raw family's ~0.185.
- Accuracy: **96.60 = baseline mean (96.57), −0.11 vs recorded baseline** — zero level gain. The average exactly recovers what the frozen tail forfeits from the cosine anneal.
- Interpretation: a time-keyed annealed schedule is already an implicit iterate average (classical averaging≈annealing equivalence); under a FIXED WALL CLOCK the SWA phase replaces the anneal tail instead of extending training, so the paper's fixed-epoch gains do not transfer. Recurring signature across EMA (EXP-011) and SWA: improved loss/calibration, unchanged argmax accuracy.

## Reusable facts
- `AveragedModel` defaults to `use_buffers=False` — BN buffers are never averaged; `update_bn` (resets + momentum=None cumulative pass) is mandatory before any eval of the averaged model. Skipping it reproduces the EXP-029 stats/weights-mismatch damage; doing it with CLEAN (test-transform) data also damages (EXP-029): use the AUGMENTED loader.
- `update_bn` handles (input, target) tuples natively; ~2s per 97×512-batch pass on H20 (uncharged wall under this benchmark's accounting).
