# EXP-019 Plan Review

1. **The expected margin is narrow.** Static scaling is weaker than a fully conditional first gate, so recovered exposure must outweigh lost conditionality. Keep this as the central research risk; preflight cannot de-risk accuracy.
2. **Low gate-output variance does not imply input independence.** EXP-017's feature logits were stronger than bias logits for gate 0. Terminal static-scale statistics make a miss interpretable: a mean remaining near one indicates failed attenuation learning; substantial attenuation with poor accuracy implicates conditional interaction.
3. **Exact-neutral initialization must learn attenuation from one.** This is required for accepted-logit isolation and cannot be replaced by post-hoc initialization at 0.65. The terminal mean/std/min/max are observational and must not trigger tuning or reruns.
4. **Reusing seed 17017 creates an avoidable seed-selection concern.** Replace the reproduction/discard sequence with one final gate initialized from the project's fixed seed 42 inside a restored CPU RNG fork. This matches the no-reroll rule and makes the static first-block scale the only addition relative to EXP-018.
5. **Throughput is not the binding accuracy risk.** Retain the standard measured guard because fixed-time feasibility is mandatory, but do not treat it as evidence for metric success.

