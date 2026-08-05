# Plan Review EXP-038

Offline fallback adversarial review:

1. A miss at classifier decay `0` and `1e-3`, with accepted `5e-4`, cannot formally close intermediate values or schedules because the response may be non-monotonic and schedules are distinct interventions. The plan must limit its conclusion to the two tested one-sided perturbations and deprioritize nearby static tuning.
2. Doubling the decay coefficient doubles only the coupled decay contribution, not necessarily the full update or realized norm shrinkage after gradient and momentum terms. The oracle language must distinguish these quantities.
3. A realized exposure below 127 passes remains a valid primary-metric observation under the goal contract, but cannot support a mechanism-level conclusion and must not be rerun.

The allocation, counts, independent Nesterov oracles, timing gate, score command, and source scope are otherwise sound.
