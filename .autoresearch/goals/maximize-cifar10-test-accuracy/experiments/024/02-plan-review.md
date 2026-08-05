# Adversarial Plan Review - EXP-024

1. **Diagonal gating is a lower-expressivity, high-probability negative test.** Its value is closing whether self-channel conditioning suffices; exposure recovery is its only route beyond full SE's 94.16.
2. **A bare threshold pass is not multi-seed causal proof.** The fixed goal criterion governs acceptance, but analysis must avoid claiming robust sufficiency or necessity from a noise-scale single run.
3. **The 138 projected-pass floor may realize fewer passes.** EXP-017 projected 135.31 and realized 133.64. Retain or tighten the floor consciously and report realized exposure.
4. **Gradient opening must use aggregate vector norms.** Signed pooled residual can cancel for individual channels, so per-element nonzero assertions would spuriously abort.
5. **Preflight files must remain ignored and experiment-scoped.** Place them only under `.autoresearch/.../experiments/024/`, never the project root.
