# Claude Opus Adversarial Result Audit — EXP-017

## Recomputed Decision

Two accuracy-blind preflight failures occurred inside `production_trace`, before paired timing and before the sole JSON print. The single preregistered preflight repair was consumed by correcting the sample-count assertion from two to three. The metric command was never launched, so there is no `best_test_acc` to compare with 95.33. The verdict guide therefore maps the outcome to `crash` with metric `NaN`, not `invalid` or `no-improvement`.

## Findings

- The one-repair rule prohibits a second preflight repair and metric run. The separate deterministic-smoke import repair should be identified explicitly as outside the decisive preflight ledger.
- `crash`/`NaN` is correct because no result exists and no hard constraint was violated.
- Zero test iteration and zero accuracy computation are supported. The original evaluator/test-loader objects were constructed during module import before guard replacement, so claims must distinguish construction from iteration/evaluation.
- Retaining 1,024 detached CUDA loss scalars is sufficient to invalidate the allocation-stability assertion as evidence about candidate state, but does not prove the absence of another leak.
- The Gradient Centralization hypothesis remains untested and this leaf must not be used as negative evidence against GC.
- Charged metric-training time consumed is zero because the metric command never launched.

## Agent Resolution

Adopted all material corrections above. The reviewer additionally said “EXP002 remains best at 95.23”; that is incorrect global context. EXP002 is the parent reference, while the existing goal-wide best remains EXP011 at 95.61.

Correct tree verdict: **`crash`**, metric **`NaN`**.

AUDIT_VERDICT: PASS
