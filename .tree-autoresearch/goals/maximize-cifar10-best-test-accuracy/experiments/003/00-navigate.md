# Navigation - Experiment 003

## Search Intent
Exploit the successful CutMix branch by finding a better regularization operating point without disturbing the validated architecture and time schedule.

## Chosen Base
**002** on `br-000` @ 95.23% - exploit

EXP-002 is the global-best tip, improved its parent by 0.61 points, and has no failed children. Its preregistered CutMix probability/alpha/cutoff were hypotheses rather than tuned values, so local refinement has higher expected payoff than discarding the winning lineage.

## Alternatives Considered
- 001 - useful for comparing alternative mixed-sample methods, but it would discard the demonstrated 0.61-point CutMix gain before this mechanism shows saturation.
- BASE - the time-aware WRN lineage remains far stronger and unsaturated.

## Policy Influence
The default policy favors positive branch momentum and low failed-child counts. EXP-002 has both, making continued exploitation the policy-aligned choice.
