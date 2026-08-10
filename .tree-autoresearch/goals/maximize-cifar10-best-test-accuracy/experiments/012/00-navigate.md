# Navigation - Experiment 012

## Search Intent

Exploit the new best lineage while looking for a low-overhead representation or calibration mechanism with plausible effect beyond the 95.49 EMA-tail plateau.

## Chosen Base
**011** on `br-000` @ 95.61 - exploit

EXP-011 is the fresh global-best tip, has no failed children, retains 25,798 steps, and adds a validated full-state EMA without material charged overhead. Its analysis identifies substantial memory headroom and a stable but lower 95.49 tail plateau, making it the highest-payoff base for an additive mechanism rather than immediately tuning EMA horizon. (`tree.sh show 011`; `experiments/011/04-analysis.md`)

## Alternatives Considered

- 004 - Four failed children already surround its 95.40 tip; EXP-011 is a successful additive extension and should receive at least one exploitation child before backtracking.
- 002 - Could support another architecture fork, but its 95.23 metric and failed architecture/augmentation children make the required path back to the 95.71 threshold longer.
- BASE/001 - Broad exploration remains possible, but the current branch has positive momentum and no saturation evidence at node 011.

## Policy Influence

The policy favors branch momentum while penalizing failed-child pileups. The single branch's new tip has positive momentum and zero failed children, so exploitation of EXP-011 is preferred; no executable policy hook imposed another constraint.
