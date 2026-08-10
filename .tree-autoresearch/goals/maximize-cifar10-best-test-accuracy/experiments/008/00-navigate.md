# Navigation - Experiment 008

## Search Intent

Exploit the strongest validated training lineage while moving away from low-yield optimizer and augmentation substitutions toward an additive representation mechanism with a plausible detectable effect.

## Chosen Base
**004** on `br-000` @ 95.40% - exploit

EXP-004 is the global best and only branch tip. It combines the validated WRN rewrite, front-loaded CutMix, and clean-tail periodic SAM while retaining 25,560 steps. Its three failed children tested unrelated self-distillation, augmentation substitution, and ASAM-package changes; none removed evidence for EXP-004 itself. Branching here preserves every successful mechanism and makes an additive architectural intervention interpretable.

## Alternatives Considered

- **002** - avoids interaction with late SAM, but discards a validated +0.17-point gain and would need a larger architectural improvement merely to recover the global best.
- **001** - offers a cleaner architecture-only base, but gives up both the +0.61 CutMix and +0.17 SAM gains without evidence that they block representation improvements.
- **BASE** - maximal exploration but an inefficient restart when the current WRN lineage is substantially stronger and has ample memory headroom.

## Policy Influence

The policy asks for judgment balancing branch momentum and failed-child pileups. EXP-004 has three failed children, so EXP-008 deliberately changes mechanism class rather than making another nearby optimizer or augmentation adjustment; its global-best status and lack of any alternative successful branch still make it the highest-payoff base.
