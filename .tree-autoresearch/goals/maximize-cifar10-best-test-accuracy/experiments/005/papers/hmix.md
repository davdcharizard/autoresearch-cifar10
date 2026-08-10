# HMix: Hybrid Mixup and CutMix

- Source: Chanwoo Park, Sangdoo Yun, Sanghyuk Chun, NeurIPS 2022, https://papers.nips.cc/paper_files/paper/2022/hash/e6f32e64b9c27d153b46c94f0fe22b56-Abstract-Conference.html
- Relevance: a nearly free replacement for the parent's early-phase CutMix that changes the form of input-gradient regularization rather than only tuning its probability.

## Mechanism

HMix shrinks the CutMix box and linearly interpolates the two images outside the box. A ratio `r` interpolates between Mixup (`r -> 0`) and CutMix (`r -> 1`) while preserving the expected label mixture. The paper uses `r=0.5` on CIFAR-100. For sampled label weight `lambda`, the pasted box has area `(1-lambda)*r`; outside-box pixels use a constant mixture coefficient chosen so the expected image contribution remains `lambda`.

## Evidence

Across CIFAR-100 backbones, HMix improves over CutMix by 0.16 to 0.59 points and is usually competitive with or better than stochastic Mixup/CutMix. On WRN-28-2, reported accuracy is 75.68 versus 74.79 for CutMix and 75.49 for stochastic Mixup/CutMix. The paper reports negligible extra computation. It does not report CIFAR-10, a short wall-clock budget, or a CutMix schedule that turns off for a clean final quarter.

## Experiment implications

- Preserve the parent's 0.5 application probability and 75% cutoff; only replace the selected CutMix mask with HMix at `r=0.5`.
- Recompute the effective label weight from the exact mask/mixing coefficients, including clipped boxes.
- Cost should be close to the current in-place CutMix path.
- Prior EXP-003 showed that small CutMix/drop-path changes are noisy and saturated, so this needs a larger mechanistic effect than another probability tweak.

## Verdict

Keep as a finalist, but rank below mechanisms with direct CIFAR-10 evidence because it remains in a region already explored by EXP-003.
