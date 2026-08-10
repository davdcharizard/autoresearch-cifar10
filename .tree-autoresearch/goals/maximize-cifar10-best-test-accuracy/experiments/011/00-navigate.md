# Navigation - Experiment 011

## Search Intent

Exploit the global-best training package with a low-overhead late-iterate averaging mechanism that targets checkpoint variance without replacing validated data, representation, or SAM exposure.

## Chosen Base

**004** on `br-000` @ 95.40 - exploit

EXP-004 is the global best and its clean-tail periodic SAM improves both accuracy and loss. Although it has four failed children, none tested model-weight averaging; EXP-006 specifically recommends preregistered EMA as an orthogonal next step after its final four checkpoints varied by 0.15 points. Averaging can retain the full 2-2-2 representation, CutMix stream, and SAM cadence with no extra model forward, directly addressing the current system-understanding question about late-iterate variance (`tree.sh show 004`; `experiments/004/04-analysis.md`; `experiments/006/04-analysis.md`).

## Alternatives Considered

- 002 - would isolate averaging from SAM and has three failed children rather than four, but discards the validated +0.17 clean-tail optimizer gain and starts 0.17 points below the best.
- 001 - removes both CutMix and SAM, abandoning two successful mechanisms without improving the averaging test's relevance.
- BASE - would discard the full successful lineage and is inappropriate for a late-solution mechanism.

## Policy Influence

The soft policy penalizes the four-failure pileup at EXP-004, but those failures cover self-distillation, augmentation substitution, ASAM, and CPU RandAugment rather than weight averaging. The mechanism-specific evidence and direct match to the measured tail limiter outweigh the generic saturation penalty. No hard hook is present.
