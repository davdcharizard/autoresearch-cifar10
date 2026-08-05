# Plan Critique - EXP-037

Offline fallback critic verdict: **REVISE**, then proceed.

1. Verification step 9 overclosed the family after any valid miss, while the milestone/proposal require >=127 passes for broad under-decay closure. Below 127 closes exact zero decay only and is mechanism-inconclusive.
2. The second-step momentum oracle must begin from a fresh common model snapshot with identical preseeded nonempty buffers. Continuing after first-step divergence invalidates cross-arm common-gradient assumptions.

All group counts, first-step Nesterov semantics, timing formula, baseline/threshold, and evaluator contract otherwise pass.
