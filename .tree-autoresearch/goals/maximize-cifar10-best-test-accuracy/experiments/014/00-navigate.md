# Navigation — Experiment 014

## Search Intent

Exploit the strongest validated lineage while selecting a mechanism distinct from EXP-011's two failed children. Preserve the global-best WRN + CutMix + SAM + EMA package and target its stable late generalization plateau.

## Chosen Base
**011** on `br-000` @ `95.61` — exploit

EXP-011 is the sole branch tip and global best. Its `+0.21`-point EMA gain retained `25,798` optimizer steps, while its two failed children tested unrelated spatial erasure and fixed-scale classifier geometry; neither discredits low-overhead loss or state-averaging changes. A third, mechanism-distinct child remains higher expected value than discarding the validated EMA gain.

## Alternatives Considered

- **004** — offers the pre-EMA online-training package, but would forfeit a validated `+0.21`-point improvement and already has four failed children.
- **002** — provides a wider architectural fork point, but discards both validated clean-tail SAM and EMA gains and has three failed children beyond its successful SAM child.
- **001/BASE** — useful only for a substantially new architecture direction; current evidence still favors improving stable generalization on the mature branch.

## Policy Influence

The soft policy prioritizes momentum, failed-child pileups, and unexplored directions. EXP-011 has the highest metric and only two heterogeneous failed children, so its momentum outweighs the modest pileup; no executable hook imposed additional constraints.
