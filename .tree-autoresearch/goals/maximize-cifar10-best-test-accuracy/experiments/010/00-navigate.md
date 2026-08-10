# Navigation - Experiment 010

## Search Intent

Explore a higher-effect architecture reallocation that preserves the validated CutMix recipe while avoiding the loader and multi-kernel overheads exposed by EXP-008 and EXP-009.

## Chosen Base

**002** on `br-000` @ 95.23 - explore

EXP-002 retains the lineage's largest recipe gain (+0.61 from front-loaded CutMix) but precedes the clean-finish SAM addition, so an architecture change can be tested without confounding it with extra late two-pass work. It has only two failed children versus four at global-best EXP-004, and its failed architecture child (EXP-009) rejected a particular multi-launch attention implementation on latency rather than testing architecture capacity itself (`tree.sh show 002`; `tree.sh children 002`; `experiments/009/04-analysis.md`).

## Alternatives Considered

- 004 - global best at 95.40, but four failed children and a diminishing +0.17 terminal gain indicate local saturation; architecture throughput changes would also alter its SAM dose.
- 001 - clean architecture launch point, but discarding EXP-002's validated +0.61 CutMix gain lowers expected payoff without improving isolation materially.
- BASE - would discard both the successful time-aware WRN and CutMix advances.

## Policy Influence

The soft policy favors balancing branch momentum against failed-child pileups. With the only branch tip carrying four failures, an interior fork from EXP-002 provides the strongest unexplored architecture direction while preserving proven training improvements. No hard policy hook is present.
