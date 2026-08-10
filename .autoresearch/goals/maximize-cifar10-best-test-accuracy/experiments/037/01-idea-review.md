# Adversarial Review — EXP-037

## Prioritized Feedback

1. BN-stat reset is analytically a no-op by the first legal evaluation (`0.9**390` inherited mass); retire it.
2. Stem centering's dominant risk is immediate BN redundancy. Add a cheap post-BN divergence null-detector before timing/production.
3. Cosine scale8 collides with recurring global readout/optimizer geometry failures and remains initially one-class despite bounded logits.
4. Cosine feature normalization may discard sample/mixing ambiguity that makes accepted CutMix effective; separate hard/CutMix diagnostics are mandatory if pursued.
5. Stem null-space decay is real but bounded and correctly attributable; raw/effective/null norms should remain explicit.

## Scored Verdict

- **BN reset — evidence/reasoning 3/10; impact 1/10.** Rigorous analysis proves practical erasure before scoring.
- **Fixed-scale cosine head — evidence/reasoning 5/10; impact 6/10.** Higher ceiling, but indirect evidence, unvalidated scale, CutMix risk, and recurrent collapse exposure.
- **Mean-centered stem — evidence/reasoning 7/10; impact 5/10.** Clean one-site projection with local/literature support and bounded geometry; BN redundancy caps likely effect.

## Pick

**Mean-Centered Stem Convolution.** It offers the strongest evidence-to-risk ratio and cleanest attribution, avoids EXP029's all-layer overhead and EXP034's norm shrink, and can prospectively detect a BN-washed-out null before spending a scored run.
