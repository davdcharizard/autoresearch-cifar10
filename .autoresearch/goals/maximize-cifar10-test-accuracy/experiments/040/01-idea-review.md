# Idea Review EXP-040

Offline fallback adversarial review of randomized finalists:

| Candidate | Evidence | Potential impact | Verdict |
|---|---:|---:|---|
| Frobenius-preserving equal-row-norm classifier | 3/5 | 3/5 | Select |
| Decay-calibrated centered-simplex regularization | 2/5 | 2/5 | Reject |
| One-time Nesterov reset | 1/5 | 1/5 | Reject |

**Pick**: Frobenius-Preserving Equal-Row-Norm Classifier.

It has measurable accepted row-radius variation, no arbitrary temperature or new parameter, correct invariants for nonzero rows, and a plausible post-pooling location. Evidence is mechanistic rather than diagnostic: balanced classes need not have equal optimal margins, and the map changes tangential conditioning plus globally coupled radial gradients. Seven of 32 synthetic initial argmaxes change, so this is a meaningful trajectory treatment.

Simplex regularization is gauge-correct but its decay-gradient norm matching is only a magnitude heuristic and the accepted mean angle is already near target. Momentum reset directly affects only 44 updates and relies on unsupported path amplification.

Corrections adopted: Frobenius equality holds at each given raw state but does not preserve the accepted scale trajectory; success cannot separate visible radius removal from changed conditioning/shared-scale coupling; a miss closes only the exact differentiable RMS map and algebraic equivalents, not other normalization/geometry families.
