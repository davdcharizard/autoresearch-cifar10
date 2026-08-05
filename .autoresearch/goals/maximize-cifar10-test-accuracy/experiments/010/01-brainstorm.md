# Brainstorm EXP-010 (Sweep)
**Created**: 2026-06-29

<!-- Goal/metric/constraints: goals/maximize-cifar10-test-accuracy/01-definition.md. Baseline 96.38 (EXP-008), bar ≥96.48, in 04-results.tsv. -->

## Experimental History Review

- **Current best 96.38% (EXP-008)**: DavidNet/ResNet-9 + whitening + ReZero + EMA + flip-TTA + (cutout12 + light RandomErasing), SGD-Nesterov lr 0.4, wd 5e-4, LS 0.2, time-based triangular one-cycle, 300s budget (~150 epochs).
- **EXP-009 (Muon, no-improvement, 94.11%, −2.27pp)**: replaced SGD on the conv weights with from-scratch **Muon** (Newton-Schulz orthogonalized momentum, ns=3, airbench weight-renorm replacing L2) at **peak LR 0.24** (airbench's value); fc/BN/α on unchanged SGD. Result: **divergence-and-recovery** — net climbed to 77.7% by ep3, then COLLAPSED to ~random (ep25-100 ~10-20%) through the high-LR phase, and re-learned only as LR→0 in the tail (ep138 94.11, still rising). The Muon implementation is VERIFIED CORRECT (clean late recovery proves NS numerics, weight-renorm, EMA, dual-optimizer wiring all sound; CUDA smoke passed). **The single open variable is the Muon peak LR.** (refs: experiments/009/04-analysis.md, knowledge/references/muon-optimizer.md)
- **Diagnosis (EXP-009)**: airbench's peak LR 0.24 does NOT transfer to our schedule. With weight-renorm pinning ‖conv‖=√out, the per-step rotation at peak is `lr·‖update‖/‖p‖ ≈ lr·√min(m,n)/√out` ≈ **~24%/step at lr 0.24** for the wide convs. airbench tolerates 0.24 only because its schedule is **8 epochs, no warmup, no high-LR plateau**; our 150-epoch triangular one-cycle holds near-peak LR for ~dozens of epochs → BN destabilizes and the net collapses. The fix is to lower the peak so the sustained-high-LR rotation is gentle enough to stay stable while still annealing fully.
- **Untried**: the Muon peak LR magnitude. EXP-009 pre-registered "lower PEAK_LR_MUON ~2-3× (≈0.08-0.12)" as the high-confidence next step.

## Sweep Plan

**Rationale**: EXP-010 is a parameter sweep, not a new method — the algorithmic decision (Muon on the conv weights, airbench weight-renorm, ns=3, fc/BN/α on SGD) is settled and its implementation is verified correct by EXP-009; the *only* thing that broke the metric was the scalar **peak LR**, and finding its stable-and-well-annealed value is the single highest-value move. The early-epoch trajectory is an unusually cheap, unambiguous diagnostic for this knob (divergence shows as a collapse toward ~10% test-acc by ep25, visible at ~100s wall), so the sweep can rank candidate LRs with SHORT diagnostic runs (kill ~ep30) and then spend one full 300s confirmation run on the best stable candidate — keeping total cost modest. No seed re-rolling (seed fixed at 42 across all trials); this is method-hyperparameter tuning, explicitly in-scope per the goal. The target recovery (94.11 → >96) is far above the ~0.1pp noise floor, so the sweep verdict is robust to run-to-run jitter.

| param | range / choices | scale | mechanism (how it moves the metric) |
|-------|-----------------|-------|-------------------------------------|
| `PEAK_LR_MUON` | {0.03, 0.06, 0.10, 0.14} | ~log | Sets the per-step orthogonalized-conv-weight rotation at the schedule peak (`≈lr·√min/√out`). Too high (≥~0.16, cf. 0.24 diverged) → sustained high-LR plateau destabilizes BN → collapse to ~random. Too low (~0.03) → stable but slow convergence → under-annealed finish below baseline. A mid value keeps the long high-LR phase stable AND fully anneals the tail → recovers from 94.11 toward/above the 96.48 bar. |

All other knobs held at EXP-009 values (the verified-correct Muon recipe): `PEAK_LR_SGD=0.4` (unchanged from EXP-008 for the fc/BN/α group, stable there), momentum 0.9 shared (NS normalizes update scale so momentum affects smoothing not magnitude — not the divergence cause), `MUON_NS_STEPS=3`, weight-renorm, no Muon WD; EMA/whitening/TTA/augmentation/schedule/seeds byte-identical to EXP-008.

**Proposed optimizer**: grid (4 values) — **rough trial budget**: 4 short diagnostic runs (each killed at ~ep30 / ~130s wall, ranked by ep25 test-acc = does it stay stable, ≳90%, vs collapse toward ~10%) + 1 full 300s confirmation run of the best stable candidate (and a 2nd confirmation if the top two diagnostics are close). ≈4×130s + ≈2×450s ≈ 24 min total.
