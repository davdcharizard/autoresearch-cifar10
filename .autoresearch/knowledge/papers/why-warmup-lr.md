# Why Warmup the Learning Rate? Underlying Mechanisms and Improvements

- **Source**: arXiv 2406.09405 (NeurIPS 2024)
- **Consulted**: EXP-014 brainstorm (2026-06-10)

## Key Claims

- Warmup's PRIMARY benefit is enabling the network to tolerate a larger peak LR; other benefits are marginal (shown across architectures/datasets incl. CIFAR-10, SGD and Adam).
- The tolerance benefit is weakest with cross-entropy loss and standard (small) initializations — exactly our setup.
- Common practice: warmup = 1–5% of total training steps.

## In-Project Transfer Caveat (IMPORTANT — EXP-014)

The paper's framing ("warmup beyond peak-tolerance is waste") is a FIXED-ITERATION intuition and **inverted** under our time-keyed schedule (progress = elapsed/TIME_BUDGET_S): shortening warmup does not free budget — it starts the cosine anneal earlier, raising lr(p) at EVERY subsequent progress point. WARMUP_FRAC 0.15 → 0.08 behaved as a net heat increase (mid-schedule deficit, −0.22pp, exp-report-014.md), forming a two-point heat dose-response with EXP-010 (peak +50% → −0.57pp).

Use this paper for stability questions (e.g., "can we ramp faster without divergence?" — yes, confirmed: zero instability at the 2x-faster ramp). Do NOT use it to predict accuracy gains from warmup reduction under time-keyed schedules.
