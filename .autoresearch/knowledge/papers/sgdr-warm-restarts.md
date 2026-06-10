# SGDR: Stochastic Gradient Descent with Warm Restarts (Loshchilov & Hutter, ICLR 2017)

**Source**: Loshchilov & Hutter — "SGDR: Stochastic Gradient Descent with Warm Restarts", arXiv:1608.03983

## Key Insights

- **Cosine annealing with warm restarts**: anneal LR from PEAK→~0 over a cycle of length T_i, then RESTART LR to PEAK and repeat; optionally lengthen cycles (T_mult>1). Two benefits reported on CIFAR WRN: (1) faster anytime convergence; (2) restarts can escape a sharp basin so the final cycle's minimum generalizes better.
- **The two benefits are separable**: the *single-final-model* benefit is the re-exploration (escaping a basin). The larger reported gains usually come from **snapshot ensembling** (Huang et al. 2017) — averaging the model at each cycle end — which is a form of weight/output averaging.
- Restarts cause a transient LOSS SPIKE (LR jumps to PEAK), then recovery. Long total schedules (hundreds of epochs) let each cycle fully converge, so the restart's re-exploration pays off.

## Relevance to this project (CIFAR-10 k=4 WRN, 300s/H20)

- Tested as EXP-029 (2-cycle SGDR, restart at 50% budget) → **REGRESSED to 95.55 (−0.67pp), final_test_loss WORSE 0.195→0.208**, at a PERFECTLY throughput-neutral 91 ep / 8ms (cleanest fair test in the project). The restart fired correctly (lr 0→0.2, loss 0.84→1.16 at 50.1%).
- **WHY IT FAILED HERE**: the budget is SHORT (~91 epochs). Splitting it into 2×~45 ep gives two under-resolved anneals — the restart destroys cycle-1's converged minimum and cycle-2 can't re-converge past what a single full-budget cosine-to-0 reaches. SGDR's re-exploration needs many epochs; its stronger (snapshot-ensemble) benefit overlaps the project's CLOSED weight-averaging axis (SWA/EMA, EXP-006/019/020). Monotone toward FEWER cycles → 1 cycle (the baseline single cosine-to-0) is optimal.
- **Do NOT retry** more cycles, T_mult>1, or restart variants at this budget. Combined with the settled LR-PEAK (EXP-016/017) and closed FLOOR (cosine-to-0, EXP-019/020), the ENTIRE LR-schedule axis is closed.
- General lesson for short-budget regimes: a single long cosine-to-0 is the strong default; warm restarts are a long-schedule technique that under-converges when epochs are scarce.
