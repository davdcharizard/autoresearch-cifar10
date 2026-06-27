# Bag of Tricks for Image Classification with CNNs (zero-γ init)

- **Source**: arXiv 1812.01187 (He, Zhang, Zhang, et al. — CVPR 2019); corroborated by Goyal et al. arXiv 1706.02677 (large-minibatch SGD)
- **Consulted**: EXP-018 brainstorm (2026-06-10)

## Key Claims

- Initializing γ=0 in the LAST BN of each residual block makes every block an identity map at init; forward/backward signal initially flows through the shortcuts, "easing optimization at the start of training".
- Stacked with linear-scaled LR + warmup it contributes to ~+1% on ResNet-50/ImageNet; Goyal et al. treat it as a standard ingredient for batch-8k training.
- The trick's primary documented value is STABILITY headroom — tolerating larger peak LRs / batches without divergence.

## In-Project Relevance

- Motivated EXP-018: zero-γ in all nine bn2 layers of the baseline recipe (large batch 512 + warmup 0.15 + peak 0.4 — the literature's exact regime), zero cost on every signature.
- **EXP-018 result: INVERTED — −0.99pp (95.72 vs 96.71), the largest clean deficit since EXP-008.** Early trail far SLOWER (ep5 35.3% vs 63.8%): the net spends warmup/peak as effectively stem-only while γs grow from 0, and under a fixed wall clock that early heat is unrecoverable (fixed-epoch ImageNet schedules amortize the turn-on cost; ours cannot). Second, the stability headroom it buys is unspendable here — peak 0.4 is certified optimal via QUALITY (EXP-010: 0.6 lost while remaining stable). Use this trick only where (a) iterations are fixed/long and (b) the peak LR is stability-limited. For fixed-wall-clock training, init interventions must ADD early learning (e.g. data-derived whitening init), not subtract early expressivity (reports/exp-report-018.md).
