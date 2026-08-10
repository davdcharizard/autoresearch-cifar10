# Navigation — Experiment 020

## Search Intent

Run one sharply diagnosed mechanism ablation that preserves GC's literature-supported convolutional projection while removing the disproportionately strong classifier projection implicated by EXP019.

## Chosen Base
**002** on `br-000` @ 95.23 — explore

EXP019 definitively rejected full convolution+classifier official-order GC but exposed a specific unresolved split: convolution directions lost 41.89% norm while the classifier lost 93.21%, and the ECCV paper states convolution-only GC is sufficient for small-resolution CIFAR (`experiments/019/04-analysis.md`; `knowledge/papers/gradient-centralization.md`). Returning to clean EXP002 isolates classifier eligibility without SAM/EMA confounding. This is a distinct preregistered projection rule, not a metric retry or coefficient tune.

## Alternatives Considered

- **011** — exploit the global tip, but four failed children and nested SAM/EMA would obscure whether classifier exclusion rescues GC.
- **004** — retains successful SAM but would confound convolution-only GC with a second optimizer-geometry mechanism.
- **002 with raw-gradient GC** — preserves L2 common-mode directions too, changing two factors at once; classifier-only eligibility is the cleaner next ablation.
- **Representation intervention from 002** — broader upside but weaker immediate evidence than the measured 93.21% classifier removal.

## Policy Influence

The policy weighs failed-child pileups against unexplored directions. Although EXP002 now has six failed children, EXP019 generated unusually specific mechanism evidence and an authoritative small-image eligibility recommendation, justifying one final focused fork before moving away from this base.
