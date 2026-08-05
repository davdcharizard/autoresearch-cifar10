# Adversarial Plan Review - EXP-022

1. **Second-pass gradient hygiene is underspecified.** Explicitly clear gradients after deriving the perturbation and verify the optimizer receives the pure gradient at perturbed weights, not an accumulated first-plus-second gradient.
2. **Timing must measure the alternating production cadence.** EXP-021 validated isolated helper semantics and dense-step cost only; it did not validate end-to-end accuracy, and alternating execution may have different overlap behavior.
3. **Feasibility thresholds conflict.** A 90% retention floor against 141.9 passes implies at least 127.71 passes, not 127.
4. **Timing gates use unverified 300-second literals.** Read `TIME_BUDGET_S` and derive transition expectations as 0.65x and 0.90x the actual value.
5. **Transition semantics need an explicit one-shot condition.** Log only on the first even-parity step that actually invokes SAM.
6. **Inline cadence logic is hard to preflight.** Put the progress/parity predicate in a small pure helper and test the strict boundary directly.
7. **A passing fixed-seed score may rest on few late evaluations.** Record this residual robustness limitation in analysis; it cannot alter the goal-defined single-run criterion.
