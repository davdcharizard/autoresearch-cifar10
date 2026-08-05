# Idea Review EXP-036

Cross-model review was intentionally unavailable in this strictly offline/local
session. The required fallback local idea critic reviewed the shuffled
finalists, all proposals, goal, system diagnosis, history, constraints, and
accepted source.

## Prioritized Feedback

1. **All candidates treat classifier geometry as a hypothesis, not a diagnosis.** Near-zero tail loss proves training separation but does not show that head capacity or radial logits cause test error. Frame the pooled treatment narrowly as a cheap nonlinear remapping of accepted features.
2. **The pooled residual MLP is distinct from EXP012/014 and has the strongest local bridge, but scale/init are unsupported operating points.** Post-pooling placement avoids the spatial bottleneck; nonzero Kaiming initialization avoids delayed exact-zero opening. Measure initial branch/direct norm ratio, logit perturbation, and backbone/classifier/head gradient norms as diagnostics without tuning from them.
3. **The cosine classifier bundles uncalibrated changes.** Feature and weight normalization, bias removal, scale 10, and altered whole-backbone gradient magnitude are inseparable; a miss closes only that exact system.
4. **Classifier under-decay has clean attribution but weak direction and ceiling.** EXP007 is not an exact retry, yet the train/test gap and severe loss regression argue against less regularization, while only 0.13% of decayed parameters changes.
5. **No finalist violates hard constraints or rerolls the scored seed.** The MLP's prospective isolated seed is acceptable because it initializes genuinely new tensors, restores global RNG, and permits no alternate.

## Scored Verdict

| Candidate | Evidence / reasoning | Potential impact |
|---|---:|---:|
| Scaled Pooled-Feature Residual MLP Head | 3.5/5 | 3.5/5 |
| Fixed-Scale Cosine Classifier | 2.5/5 | 4/5 |
| Exclude Only Classifier Weight From Decay | 2/5 | 1.5/5 |

## Pick

**Scaled Pooled-Feature Residual MLP Head.** It best extends the only locally
positive capacity-plus-invariance interaction, spends capacity after costly
spatial processing, preserves the direct path, and differs materially from
failed spatial bottlenecks and zero-initialized branches. Advance only the
fixed `128 -> 64 -> 128`, ReLU, scale-0.1, seed-36036 design with diagnostics,
the >=130-pass gate, and complete neighboring-family closure after a valid miss.
