# System Understanding: Maximize CIFAR-10 Best Test Accuracy
**Last verified**: 2026-08-06 @ d68f73a (best: 95.61%)

## Problem Decomposition

- **Input stream** - 50,000 training images, random crop/flip, 256 independent images per batch, and 195 dropped-last batches per natural epoch; bound by stochastic sample diversity and host loading. Eight workers keep the measured GPU step near 10 ms (`prepare.py`; `experiments/004/04-analysis.md`).
- **Early regularized training** - the first 75% of charged time uses a 0.5 CutMix gate and full drop path. EXP-011 presented 20,857 eligible batches and applied 10,345 mixes (0.4960); representation learning and generalization, not memory, bound this phase (`experiments/011/04-analysis.md`).
- **Late clean optimization** - the final 25% removes CutMix, decays drop path, and applies rho-0.05 SAM every second eligible step. EXP-011 used 2,471 SAM pulses among 4,941 eligible batches and finished 25,798 total steps (`experiments/011/04-analysis.md`).
- **Late trajectory averaging** - every 31st clean-tail state feeds a full-state, 18.75-second-half-life EMA. EXP-011 made 160 balanced samples and 27 exact EMA evaluations; EMA/live relative parameter distance ended at 1.51% (`experiments/011/04-analysis.md`).
- **Model** - PreAct WRN-16-4 has 2,748,890 parameters and runs BF16/channels-last. Online model plus EMA/audit shadows allocated 1,222.4 MiB on a 97,871 MiB H20, so memory capacity is not limiting (`experiments/011/04-analysis.md`).
- **Evaluation** - the fixed 10,000-image test set is evaluated once per epoch outside charged time, live before EMA activation and EMA afterward. EXP-011 made 133 evaluations and took 447.9 total seconds; `best_test_acc` selects the checkpoint maximum (`prepare.py`; `experiments/011/04-analysis.md`).

## Current Bottleneck

The dominant limiter is a stable generalization gain above the new EMA tail plateau, not throughput or memory. Lineage gains diminished from +3.11 points for WRN to +0.61 CutMix, +0.17 SAM, and +0.21 EMA package-level improvement. EXP-011's formal best was 95.61, but epochs 118-133 averaged 95.493 and ended at 95.46; the next formal threshold is 95.71 (`experiments/011/04-analysis.md`). A credible child therefore needs a plausible stable effect near 0.25-0.30 points, not a micro-change that merely shifts the selected maximum.

## Headroom Assessment

- Memory headroom is extreme: 1,222.4 MiB is about 1.25% of the H20's 97,871 MiB, leaving capacity for wider/fused representations or additional sparse state (`experiments/011/04-analysis.md`).
- Compute headroom is schedule-dependent: ordinary steps are about 10 ms and SAM steps about 20 ms; EXP-011 completed 25,798 updates. Sparse EMA was effectively free (0.9996x paired latency), while extra model forwards directly reduce optimizer exposure (`experiments/011/04-analysis.md`; `experiments/006/03-execute.md`).
- Data exposure should be preserved. EXP-005 kept 25,492 steps but halved new identities per epoch and lost 0.12 points, showing that step count alone does not bound useful training work (`experiments/005/04-analysis.md`).
- The fixed protocol has limited resolution: throughput changes phase dose, and EXP-011's 16-checkpoint EMA tail still spanned 0.17 points. This is frozen protocol context, not a reason to change evaluation mid-tree (`experiments/011/04-analysis.md`).

## Open Questions

- Can a low-overhead representation or logit-calibration mechanism lift the stable 95.49 EMA plateau rather than only its maximum?
- Does unused memory support a wider or recalibrated representation with enough effect to offset any throughput loss?
- How much of the remaining error is confidence/calibration versus incorrect class boundaries? EXP-011 improved final loss to 0.1552 versus EXP-004's 0.1654, but only added 0.06 final accuracy.
- Is the 18.75-second EMA horizon near the useful bias/variance operating point, or would another trajectory summary improve the stable tail?
- Would an identical-code repeat quantify wall-clock phase-dose variance, or is that diagnostic run too costly relative to direct mechanism tests?
