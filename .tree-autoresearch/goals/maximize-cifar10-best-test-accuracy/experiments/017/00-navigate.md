# Navigation — Experiment 017

## Search Intent

Explore a different generalization/representation path after repeated failures at the EMA tip and its SAM parent, while retaining the strongest inexpensive early-training recipe.

## Chosen Base
**002** on `br-000` @ 95.23 — explore

EXP-002 provides the validated WRN-16-4 plus front-loaded CutMix core, which gained 0.61 points and preserved clean late convergence (`experiments/002/04-analysis.md`). Its successful SAM child led to EXP-011, but EXP-004 now has six failed children and EXP-011 has four, indicating local saturation across output loss, classifier geometry, augmentation, architecture, and averaging. EXP-002's three failed children rule out a narrow CutMix/drop-path scalar sweep, four-gate SE, and one compute-neutral stage-depth move, but leave other low-overhead representation and optimization mechanisms open. Its local threshold is 95.33.

## Alternatives Considered
- **011** — global best at 95.61, but four distinct failed children and a required 95.71 bar make another direct micro-intervention low expected value.
- **004** — retains successful SAM, but six failed children now cover a broad set of nearby mechanisms and two preflight rejections.
- **001** — clean WRN parent has wider conceptual freedom, but discards the validated 0.61-point CutMix gain without a specific incompatible mechanism.

## Policy Influence

The soft policy asks for balance between branch momentum, failed-child pileups, and unexplored directions. With only one branch and heavily explored later nodes, an interior fork from EXP-002 is the best exploration bet without returning to the much weaker root.
