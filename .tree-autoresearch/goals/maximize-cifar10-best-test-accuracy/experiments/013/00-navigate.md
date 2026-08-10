# Navigation - Experiment 013

## Search Intent

Exploit the strongest validated package while changing mechanism family after the first failed child: target the 95.49 EMA plateau with a low-overhead loss, averaging, or calibration intervention that does not repeat Cutout's spatial-erasure overlap or production-dose miss.

## Chosen Base

**011** on `br-000` @ 95.61 - exploit

EXP-011 remains the sole extendable tip and global best, with a stable late EMA mean of 95.493125, negligible EMA overhead, and only one failed child (`tree.sh show 011`; `experiments/011/04-analysis.md`). EXP-012 reached 95.52 but is a terminal no-improvement leaf; it discredits only the exact full-probability complementary Cutout package and leaves loss calibration, EMA-horizon, and cheap representation directions open (`experiments/012/04-analysis.md`).

## Alternatives Considered

- **004** - still extendable and has diverse evidence, but abandoning the validated +0.21 EMA package would lower the starting metric and repeat older search territory.
- **002** - offers architectural exploration, but its existing architecture children did not improve accuracy and the current objective is better served by exploiting the stronger 011 package.
- **001/BASE** - useful only for a fundamentally new branch; no current evidence justifies discarding WRN, CutMix, SAM, and EMA together.

## Policy Influence

The soft policy favors momentum while accounting for failed-child pileups. EXP-011 has the best metric and only one failed child, so another tip extension is preferred; no executable hook imposed a hard restriction.
