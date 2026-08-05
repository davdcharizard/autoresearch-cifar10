# Adversarial Review - EXP-027 Candidate Ideas

## Prioritized Feedback

1. **The composition's overconfidence-cure story is unsupported.** Both EXP-011 and EXP-026 worsened final test loss. Pre-register whether combined loss falls below EXP-011's 0.2782 as a mechanism check, without confusing it with the primary acceptance rule.
2. **Its margin requires a high-variance interaction.** The two deltas sum to only 94.20; be explicit that the bet needs near-additive top-1 benefit and sub-additive harm, not that orthogonality guarantees success.
3. **Protect the extra-block exposure regime.** RandAugment should overlap behind the slower GPU path, but require stable augmented-loader timing and at least 130 projected and realized passes.
4. **Reconstruction must be exact.** Reuse EXP-026's validated worker RNG/cutoff protocol verbatim and prove the model matches EXP-011 state/topology rather than re-deriving approximate variants.
5. **Batch 128 remains confounded and weakly evidenced.** It co-varies update granularity, BatchNorm statistics, and mixup refresh while halving the LR curve.
6. **Alpha 0.1 is honest but low-value map completion.** Its endpoint-heavy distribution and the uniformly negative mixup-perturbation ledger predict under-regularization.

## Scored Verdict

| Candidate | Evidence / reasoning | Potential impact |
|---|---|---|
| Extra block plus early RandAugment | **3.5/5** - two real positive signals on distinct axes, but both worsen loss and interaction evidence is absent. | **4/5** - only candidate with a credible path above 94.17, though it needs near-additive benefit. |
| Batch 128 with scaled LR | **2.5/5** - defensible operating point but several co-varying mechanisms and no direct local signal. | **2.5/5** - plausible but bounded, with under-update risk. |
| Alpha-0.1 mixup | **2/5** - clean bracket completion whose own mechanism predicts failure. | **1.5/5** - low expected ceiling after all prior mixup perturbations regressed. |

## Pick

**Extra 8x8 Block Plus Fixed Early RandAugment.** It is a genuinely new interaction with the only non-trivial ceiling. Keep exact component oracles, >=130 passes, and report whether loss improves over EXP-011 to distinguish a useful interaction from coincident top-1 noise.
