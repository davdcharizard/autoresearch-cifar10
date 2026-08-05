# EXP-012 Blind Idea Review

Offline local `idea-critic` fallback review, 2026-07-24.

## Prioritized Feedback

1. **Late whole-state EMA has a stronger diagnosis than direct local evidence.** EXP-011's worse loss motivates generalization control, but accepted and EXP-011 best scores were terminal; late iterate variance is not demonstrated and EMA lag is a concrete risk. Treat it as exploratory and retain the fixed short horizon/cadence.
2. **The exact bottleneck has the best empirical bridge, but constrained capacity is not proven to generalize better.** EXP-010/011 support low-resolution capacity, while the bottleneck retains only about 18% of EXP-011's added transform budget. Keep its fixed ratio, placement, accepted initialization, topology checks, and throughput gate; no endpoint-zeroing rescue.
3. **EMA BatchNorm state remains approximate.** Averaged running moments are not exact statistics for averaged weights. Whole-state EMA is coherent, but swap/restore, integral counters, and exception restoration must be hard semantic gates.
4. **Zero endpoints are technically careful but evidentially weakest.** Zeroing `conv2.weight` avoids the permanently dead branch caused by zeroing pre-ReLU `bn2`, but there is no direct evidence for benefit in this shallow WRN. Treat it as an optimization-geometry moonshot.
5. **No finalist has a fatal scope or constraint problem.** All are single `train.py` treatments with fail-closed local preflights and fixed one-run decision rules.

## Scored Verdict

| Candidate | Evidence and reasoning | Potential impact |
|---|---:|---:|
| Exact 8x8 Bottleneck Residual Refinement | **7.5/10** - two distinct low-resolution capacity probes were positive, including a 94.15% near miss, and this topology is materially different and exact. | **7.5/10** - it may retain most accepted exposure while preserving enough nonlinear refinement to clear 94.17%, though reduced rank may erase the signal. |
| Late Whole-State EMA | **7.0/10** - averaging targets generalization at low cost and is rigorously specified, but terminal-best trajectories provide little direct variance evidence. | **6.5/10** - mild gains are plausible, but trajectory lag and approximate BN statistics limit upside. |
| Zero-Initialized Residual Endpoints | **6.0/10** - the pre-activation adaptation is mathematically sound, but lacks direct local or cited evidence for this shallow WRN. | **6.5/10** - unchanged throughput and a different basin offer upside, but the effect may wash out over about 27,000 updates. |

## Pick

**Advance Exact 8x8 Bottleneck Residual Refinement.** It is anchored to the clearest local signal while testing a genuinely different, much cheaper transformation. Its exact fixed topology and gates make failure interpretable. Preserve accepted initialization and do not combine it with endpoint zeroing, EMA, or any adaptive fallback.
