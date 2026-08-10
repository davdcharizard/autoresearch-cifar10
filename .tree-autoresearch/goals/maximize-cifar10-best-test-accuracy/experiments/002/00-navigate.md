# Navigation - Experiment 002

## Search Intent
Exploit the new global-best WRN branch by targeting the remaining generalization gap with a low-overhead additive training intervention.

## Chosen Base
**001** on `br-000` @ 94.62% - exploit

EXP-001 is the only non-root successful node, the global-best tip, and has no failed children. Its late training loss approached zero while test accuracy plateaued at 94.62%, and its analysis identifies front-loaded mixed-sample regularization as the highest-confidence next direction.

## Alternatives Considered
- BASE - a fresh fork would discard the validated 3.11-point architecture/schedule gain before the winning branch has shown saturation.

## Policy Influence
The default policy favors branch momentum and penalizes failed-child pileups. EXP-001 has strong positive momentum and zero failed children, so exploitation has the highest expected payoff.
