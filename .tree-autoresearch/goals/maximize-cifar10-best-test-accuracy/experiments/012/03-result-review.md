# Claude Adversarial Result Review: EXP-012

- **Reviewer**: Claude Code 2.1.220
- **Evidence**: complete raw `run.log`, frozen parent/thresholds, preflight values, dose and stable-tail rules
- **Initial verdict**: BLOCK on causal wording; run classification itself confirmed as exit-0 no-improvement

## Independently Recomputed Results

- Summary: best/final 95.52%, final loss 0.1557, 300.0 charged seconds, 450.1 total seconds, 131 epochs, 25,376 steps, 2,748,890 parameters.
- Evaluation routing: 131 total = 105 live + 26 EMA; swaps/restoration checks 26/26.
- Complement: `10,310 = 20,461 - 10,151`; early/all-step shares 0.503885/0.406289; 2,639,360 images.
- Area: `517,469,257 / 2,639,360 = 196.058612`, inside `[195.8,196.2]`; masked fraction 0.191463.
- SAM: first step 20,462 immediately after the early boundary; 2,458/4,915 = 0.50010.
- EMA: 158 updates, 79/79 parity, 157 finite nonzero consecutive distances, and zero restoration/coverage/nonfinite/RNG failures.
- Final 16: `95.37, 95.33, 95.36, 95.35, 95.35, 95.38, 95.38, 95.40, 95.39, 95.46, 95.43, 95.48, 95.48, 95.50, 95.51, 95.52`; sum 1,526.69, mean 95.418125, range 95.33-95.52, progress span 0.139813.
- Formal deltas: 95.52-95.71 = -0.19 versus threshold; 95.52-95.61 = -0.09 versus parent best.

## Blocking Concern And Resolution

Claude confirmed that all integrity audits are clean and the correct tree classification is exit-0 no-improvement. It blocked the initial causal phrase “stable-effect hypothesis falsified” because the mechanism-dose condition is conjunctive and `25,376 <25,500`: the observed final-16 values all fall in the preregistered falsified bin, but the under-dosed run cannot cleanly falsify the mechanism at full dose. The durable execution record now states exactly that distinction and adds the unreconciled 218.8-step projection miss. No causal efficacy claim and no retry are permitted.

## Required Classification

EXP-012 is an exit-0 no-improvement. Its best is below both parent and formal threshold, its final-16 observed tail occupies the falsified bin, and its step dose is invalid despite otherwise complete Cutout/SAM/EMA integrity. It neither improves the tree nor provides a valid full-dose causal falsification.

## Follow-up Verdict

- **Verdict**: PASS
- Claude confirmed the revised execution record now makes the required observed-bin versus full-dose-causality distinction, discloses the projection miss, prohibits retry and causal efficacy claims, and preserves every independently recomputed value.
