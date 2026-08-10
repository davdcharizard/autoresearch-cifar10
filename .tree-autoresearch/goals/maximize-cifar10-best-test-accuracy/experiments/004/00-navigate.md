# Navigation — Experiment 004

## Search Intent

Exploit the strongest validated training stack while exploring a qualitatively different generalization mechanism after scalar CutMix/drop-path tuning failed confirmation.

## Chosen Base
**002** on `br-000` @ 95.23% — exploit

EXP-002 is the sole successful branch tip and the accepted global best. It combines the high-throughput time-aware WRN with validated front-loaded CutMix, leaving substantial memory headroom for a new mechanism. Its only failed child, EXP-003, rules out a narrow regularization-strength grid rather than the parent stack itself.

## Alternatives Considered

- **001** — would discard EXP-002's confirmed +0.61-point CutMix gain without evidence that the new direction conflicts with CutMix.
- **BASE** — would abandon the validated +3.72-point architecture and recipe improvements from EXP-001 and EXP-002.
- **003** — ineligible because it is a no-improvement terminal leaf.

## Policy Influence

The policy favors branch momentum while accounting for failed-child pileups. EXP-002 has strong momentum and only one mechanism-specific failed child, so extending it has the best expected payoff.
