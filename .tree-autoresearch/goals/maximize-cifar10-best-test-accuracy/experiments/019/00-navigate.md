# Navigation — Experiment 019

## Search Intent

Explore a low-overhead optimizer-conditioning direction that remains scientifically unanswered, while avoiding another direct trajectory smoother after EXP018's stable Lookahead regression.

## Chosen Base
**002** on `br-000` @ 95.23 — explore

EXP002 is the clean WRN + front-loaded CutMix launch point beneath SAM and EMA. Its child EXP017 established reference-ordered Gradient Centralization math but crashed in a disposable allocation harness before timing or accuracy, so GC remains untested rather than disproven (`tree.sh children 002`; `experiments/017/04-analysis.md`). EXP018 then validated the exact fixed-device-scalar and post-state-baseline harness pattern needed to correct that procedural failure (`experiments/018/04-analysis.md`). Starting below SAM/EMA preserves causal isolation and avoids nested smoothing.

## Alternatives Considered

- **011** — exploit the 95.61 global tip, but four failed children and the 95.49 tail plateau argue against adding another weakly isolated mechanism before GC is resolved.
- **004** — retains successful SAM without EMA, but would confound GC with a two-pass optimizer geometry intervention.
- **001** — offers a less regularized fork, but discards the validated +0.61-point CutMix gain without a specific reason.

## Policy Influence

The soft policy asks for judgment balancing branch momentum, failed-child pileups, and unexplored directions. With no executable hook, the untested GC question at interior node 002 outweighed immediate exploitation of the four-failure global tip.
