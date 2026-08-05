# Adversarial Review - EXP-023 Candidate Ideas

## Prioritized Feedback

1. **FP32 fused SGD does not attack the diagnosed limiter.** EXP-009 already gained 12.1% exposure while losing accuracy; a likely 1-2% optimizer speedup has no credible path through the generalization boundary.
2. **The diagonal-gate exposure premise is contaminated by EXP-017 diagnostics.** Full SE's MLP is tiny; pooling, launches, and observation overhead may dominate. A diagonal gate is scientifically clean but removes cross-channel interaction and has a thin accuracy ceiling.
3. **Width plus SE must not be justified by adding noisy deltas.** Both mechanisms act at 8x8 and may be redundant. Its legitimate hypothesis is super-additivity: conditional routing makes the additional channels useful.
4. **Composition exposure needs a stricter gate.** Remove all runtime gate diagnostics, use matched production timing, and require at least 125 projected passes before scoring.
5. **Interpretation must be preregistered.** Anything below 94.17 is sub-additive failure, not evidence for post-hoc width, ratio, seed, or schedule rescue.

## Scored Verdict

- **Compose selective width plus two-gate SE**: evidence 6/10, impact 7/10. Highest ceiling through a plausible feature-supply/conditional-routing interaction, with significant redundancy and exposure risks.
- **Two diagonal conditional gates**: evidence 5/10, impact 4/10. Isolates an untested mechanism but weakens the only near-positive attention design.
- **FP32 fused SGD**: evidence 3/10, impact 2/10. Likely near-no-op against a non-exposure limiter.

## Pick

**Compose selective width plus full two-gate SE**, conditional on diagnostic-free implementation and a fail-closed >=125-pass timing gate. It is the only candidate with a credible mechanism and ceiling beyond the acceptance margin.
