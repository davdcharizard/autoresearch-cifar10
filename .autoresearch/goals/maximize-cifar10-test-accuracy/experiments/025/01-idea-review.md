# Adversarial Review - EXP-025 Candidate Ideas

## Prioritized Feedback

1. **The slate is dominated by noise-scale closures.** Early-only mild RandAugment remains the broader orthogonal lever for a future loop.
2. **Neighbor mixing does not match the global-interaction diagnosis.** Width-three convolution on unordered channels is arbitrary sparse mixing and is dominated by full SE for acceptance.
3. **Diagnostic-free SE has a narrow, exposure-only thesis.** History says added passes do not reliably convert top-1. State honestly that >=137 passes must convert the exact +0.09 trajectory, and treat any sub-threshold result as closure.
4. **Alpha 0.1 carries an adverse under-regularization prior.** It is map completion, not a limiter-directed bet.
5. **Full SE needs semantic-neutrality proof.** Exact seed, gate state, accepted common tensors, RNG, placement, initial logits, and optimizer grouping must match EXP-017's mechanism without diagnostics.

## Scored Verdict

- **Diagnostic-free full two-gate SE**: evidence 4/5, impact 2/5. Preserves the only positive mechanism but has only noise-scale headroom.
- **Neighbor-mixing gates**: evidence 2/5, impact 2/5. Cheap but mismatched to global interaction and likely below full SE.
- **Alpha-0.1 mixup**: evidence 2/5, impact 1/5. Clean implementation with no favorable local direction.

## Pick

**Diagnostic-Free Full Two-Gate SE Closure.** It preserves the demonstrated +0.09 function class and yields a clean test of whether observation overhead held back the fixed-time trajectory. A miss closes the exposure explanation.
