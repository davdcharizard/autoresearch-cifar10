# Navigation — Experiment 022

## Search Intent

Exploit the global-best stack with one remaining systems-grounded capacity intervention, rather than spend another run on optimizer smoothing, scalar loss calibration, or intermediate supervision.

## Chosen Base
**011** on `br-000` @ 95.61 — exploit

EXP011 is the global tip and has fewer failed children than the now more crowded EXP004/EXP002 interiors. Its EMA tail plateau is stable enough that a genuine capacity lift should remain visible after trajectory averaging. EXP014 proved that a stage-3 width change is operationally sound and memory-cheap but rejected width 320 solely at a 1.161x latency gate without measuring accuracy; the separately preregistered width-288 taper remains a distinct, unresolved multiple-of-32 operating point with an approximately 1.076x systems prior.

## Alternatives Considered

- **004** — cleaner live-model attribution and lower 95.50 threshold, but it now has six failed children; EXP021 just rejected the leading low-cost representation idea there.
- **002** — seven failed children and completed GC close-out make another optimizer/representation fork poor value.
- **001** — could support a broader architecture fork, but discards validated CutMix/SAM/EMA progress and would need multiple successes to approach the global best.
- **011 with EMA-horizon tuning** — cheaper, but likely sub-threshold and weakly grounded after four failed narrow children.

## Policy Influence

The policy weighs tip momentum against failed-child pileups. Four failed children indicate caution, but EXP011 remains less crowded than the relevant interiors and width 288 is explicitly distinguished from every failed child by EXP014's no-accuracy, width-specific feasibility result. This is a bounded final exploit of the tip, not a generic local tune.
