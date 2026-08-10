# Navigation — Experiment 018

## Search Intent

Recover the highest-information untested direction from EXP017 while keeping mechanism attribution below SAM and EMA. Use a fresh experiment protocol so corrected feasibility instrumentation cannot leak test information or inherit a spent repair allowance.

## Chosen Base
**002** on `br-000` @ 95.23 — explore

EXP002 is the validated front-loaded CutMix parent and the same isolation point chosen for GC. Its EXP017 child is a `crash/NaN` caused entirely by temporary preflight bookkeeping before timing or metric access; Claude explicitly judged reference-ordered GC untested (`experiments/017/04-analysis.md`). Retrying from EXP002 therefore answers the original optimizer-geometry question cleanly, with a 95.33 local threshold and without confounding the successful SAM/EMA mechanisms.

## Alternatives Considered

- 011 — the 95.61 global-best tip has four failed children and is saturated; stacking unvalidated GC onto SAM+EMA would weaken causal attribution and require 95.71.
- 004 — SAM is successful at 95.40, but GC should first demonstrate an isolated effect before composition with a second gradient intervention.
- 001 — offers broader exploration, but abandoning the validated CutMix gain is lower expected payoff than resolving the untested EXP017 hypothesis.

## Policy Influence

The soft policy asks for momentum-versus-failed-child judgment. The tip's four failed children favor exploration, while EXP017's protocol crash is not research evidence against another EXP002 fork. No executable hook constrained the choice.
