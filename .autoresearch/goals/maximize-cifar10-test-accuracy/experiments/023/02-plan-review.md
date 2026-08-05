# Adversarial Plan Review - EXP-023

1. **The 125-pass gate sits at the multiplicative-overhead estimate.** Tighten it so the composition does not score at the exact exposure-starved boundary suggested by standalone timings.
2. **The lower floor needs an explicit rationale.** EXP-017 projected 135.31 passes and EXP-010 accepted a 120-pass capacity floor; a composed architecture needs a distinct bound rather than inheriting either number silently.
3. **The 160-channel gates are not the validated 128-channel gates.** Treat transfer of conditional selection to a new feature distribution as the experiment's hypothesis, not as known +0.09 evidence.
4. **Keeping both gates needs history reconciliation.** EXP-017 suggested final-only, but EXP-018 and EXP-019 subsequently showed final-only/static approximations destroy the signal; the plan should cite that later evidence.
5. **Selection bias remains high.** Honest preregistration prevents post-hoc rescue but does not make noisy near-miss deltas additive; report failure as closing only this exact composition.

Scope, seed discipline, parameter arithmetic, and hard constraints are otherwise sound.
