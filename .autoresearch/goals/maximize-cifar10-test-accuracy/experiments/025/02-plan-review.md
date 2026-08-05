# Adversarial Plan Review - EXP-025

1. **The causal thesis is narrow and conflicts with the broad exposure history.** The only intervention versus EXP-017 is roughly 3.4 recovered passes, while prior faster variants did not reliably improve top-1. A threshold crossing could be noise rather than a general exposure effect.
2. **Seed 17017 is outcome-selected.** Reusing the exact favorable EXP-017 gate seed is defensible only as treatment identity, not as evidence independent of seed noise; the result must remain a single fixed-draw closure rather than motivate seed changes or reruns.
3. **The preflight must exercise production construction.** Importing `train.py` does not call `main`; the plan must expose gate attachment through the real `WideResNet` constructor rather than duplicate construction logic in the preflight.
4. **A 94.17 result has no variance estimate.** The fixed-seed, one-run protocol cannot distinguish a one-image crossing from stochastic noise, although rerunning would violate the goal's anti-reroll discipline.
5. **Treatment identity must cover init-time RNG consumption.** Removing diagnostics must not shift the gate weight draws; compare against the archived EXP-017 seed oracle explicitly.
6. **The scored run needs a realized exposure gate.** A short timing projection can overestimate full-run throughput, so require the actual `num_steps` to represent at least 137 passes as part of hypothesis verification.

## Disposition

- Adopt concerns 3, 5, and 6 as execution-hardening changes.
- Retain the experiment despite concerns 1, 2, and 4 because the approved brainstorm explicitly frames it as a low-ceiling, single-draw closure. Seed 17017 reproduces the prior treatment rather than searching seeds; no rerun or seed change is permitted, and a miss closes the hypothesis.
