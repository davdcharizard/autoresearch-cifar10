# Adversarial Review - EXP-024 Candidate Ideas

## Prioritized Feedback

1. **Diagnostic-free full SE has a weak success mechanism.** EXP-018/013 show extra exposure does not reliably improve top-1, and EXP-017's seed-17017 near miss is an optimistic result-selected anchor. It is legitimate but low-ceiling.
2. **Diagonal gates risk removing essential cross-channel mixing.** This is the strongest failure mode, but the design uniquely retains both placements and per-example conditioning, the two ingredients later ablations established as necessary.
3. **Diagonal gating is the clean remaining mechanism test.** It asks only whether self-channel pooled response suffices, is exposure-safe, and can close that question informatively on failure.
4. **Alpha 0.1 is a low-value map-completion probe.** Endpoint-heavy coefficients move toward less regularization despite repeated evidence that perturbing the accepted mixup regime loses accuracy.

## Scored Verdict

- **Two diagonal conditional gates**: evidence 5/10, impact 6/10. Strong simplification risk, but it preserves known necessary ingredients and has the best feasible ceiling.
- **Diagnostic-free full two-gate SE**: evidence 6/10, impact 4/10. Strong direct anchor but outcome-selected and capped near the threshold.
- **Weaker alpha-0.1 mixup**: evidence 3/10, impact 2/10. No favorable local direction and no connection to the diagnosed limiter.

## Pick

**Two Diagonal Conditional Stage-3 Gates.** Frame it as a test of whether per-channel self-gating retains the attention signal or global cross-channel mixing is essential. Require zero-init gradient semantics and at least 138 projected passes before scoring.
