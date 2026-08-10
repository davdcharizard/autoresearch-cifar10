# Navigation — Experiment 021

## Search Intent

Leave the crowded optimizer-projection and scalar-loss space and test a materially new, low-forward-cost representation-learning mechanism that can later compose with trajectory averaging if it earns a stable gain.

## Chosen Base
**004** on `br-000` @ 95.40 — explore/exploit balance

EXP004 retains the validated WRN, CutMix, and clean-tail periodic SAM package but predates EXP011's EMA. It is the cleanest successful parent for a training-only representation intervention: the formal threshold is 95.50, evaluation remains on the live deployable model, and any gain can later be tested with EMA rather than being hidden or confounded by shadow-state behavior. This lineage already produced the global best through EXP011, so the base has demonstrated momentum despite several failed siblings.

## Alternatives Considered

- **011** — the global-best exploit choice, but four failed children and a 95.49 EMA plateau indicate local saturation; adding a training-only auxiliary mechanism directly would entangle its effect with shadow ownership and raise the first test's threshold to 95.71.
- **002** — cleanest optimizer baseline, but it now has seven failed children and the GC close-out supplied no reason to spend another experiment there.
- **001** — offers broad architecture/augmentation freedom, but discards the proven +0.61 CutMix gain and creates a larger recovery burden.
- **BASE** — maximally exploratory but would abandon the successful WRN representation and is unjustified without evidence of a fundamentally different architecture family.

## Policy Influence

The policy asks for expected payoff after weighing momentum and failed-child pileups. EXP011 is strongest numerically but locally crowded; EXP002 is more crowded still. EXP004 is an interior successful node whose next representation direction is distinct from its failed augmentation, geometry, and optimizer children, making it the best risk-adjusted fork point.
