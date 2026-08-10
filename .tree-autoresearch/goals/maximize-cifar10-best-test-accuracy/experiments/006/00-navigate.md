# Navigation — Experiment 006

## Search Intent

Exploit the global-best WRN/CutMix/SAM stack while moving away from EXP-005's repeat-view data bottleneck toward an orthogonal representation or architecture mechanism.

## Chosen Base
**004** on `br-000` @ 95.40% — exploit

EXP-004 remains the sole branch tip and global best. Its validated period-two clean-tail SAM added 0.17 points while preserving 25,560 steps, and it now has only one failed child. EXP-005 failed because its half-overlap DLB recipe halved new-image introduction; that mechanism-specific failure does not indicate that EXP-004 is saturated for representation changes. Growing from 004 preserves every accepted gain and the independent-image stream.

## Alternatives Considered

- **002** — would discard EXP-004's validated +0.17-point SAM gain without evidence of incompatibility with the next representation lever.
- **001** — would additionally discard the +0.61-point CutMix improvement.
- **BASE** — would abandon the modern architecture and all validated training improvements.
- **005** — ineligible terminal no-improvement leaf; its DLB code remains inspectable but is not a valid base.

## Policy Influence

The policy weighs momentum against failed-child pileups. EXP-004 has the best metric and only one failed child, below any reasonable saturation threshold, so its expected payoff remains highest. No search-policy hook imposes an alternative.
