# Navigation — Experiment 023

## Search Intent

Explore an alternative generalization branch from the validated time-aware WRN backbone, avoiding the saturated CutMix/SAM/EMA lineage and the now-closed dense-width direction.

## Chosen Base
**001** on `br-000` @ 94.62 — explore

EXP001 has no failed direct children and preserves the 3.11-point architecture/schedule gain while exposing a clean near-zero-train-loss generalization gap. Its only child is the successful CutMix lineage, so a distinct regularization or representation mechanism can form a genuine alternative branch rather than another crowded tip tweak. This sacrifices 0.99 points of inherited metric versus EXP011 but gains substantial search freedom and a lower 94.72% local threshold.

## Alternatives Considered

- **011** — global best, but five failed children now cover augmentation, classifier geometry, loss calibration, and dense width; another local exploit has poor expected value.
- **004** — six failed children and its remaining mechanisms mostly inherit the same CutMix/SAM assumptions.
- **002** — seven failed children across architecture placement, attention, optimizer, and gradient transforms make it the most saturated interior.
- **BASE** — no failed children, but discards the large validated WRN/time-schedule gain and would require a new architecture package to recover three points.

## Policy Influence

The policy asks for branch momentum balanced against failed-child pileups. With only one branch and saturated descendants, EXP001 is the highest-performing extendable node with zero failed direct children and therefore the best exploration point.
