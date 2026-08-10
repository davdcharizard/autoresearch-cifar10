**AUDIT: PASS — evidence is internally consistent, accuracy-blind, and correctly classified. Correct tree verdict: `crash`, metric `NaN` (valid pre-metric preflight rejection, EXP-014 precedent).**

## Gate recomputation from `PREFLIGHT_JSON`

All medians, drifts, dispersions, and projections recompute exactly from the raw round data; every reported number matches.

| Gate (preregistered in 02-plan.md M3) | Recomputed | Threshold | Result |
|---|---|---|---|
| Parent drift ordinary / cutmix / sam | 0.009309 / 0.012005 / 0.009301 | ≤ 0.03 | PASS |
| Ratio MAD/median ordinary / cutmix / sam / weighted | 0.002632 / 0.001163 / 0.000515 / 0.001326 | ≤ 0.01 | PASS |
| Weighted charged median ratio | 1.160794 (round ratios 1.15965, 1.16079, 1.16956, 1.15925, 1.21887 → median confirmed; round-1 hand-recompute from raw ms gives 1.15965 ✓) | ≤ 1.10 | **FAIL** |
| Charged-path max ratio | ordinary 1.16737, cutmix 1.27270, sam 1.24896 | ≤ 1.13 | **FAIL** (all three paths) |
| Projected steps | early 20857/1.161156 = 17,962.28; late 4941/1.161904 = 4,252.50; total 22,214.78 | ≥ 23,200 | **FAIL** |
| Projected epochs | floor(22,214.78/195) = 113 | ≥ 118 | **FAIL** |
| Projected EMA samples | 160 × 4252.50/4941 = 137.705 | ≥ 140 | **FAIL** |
| Conservative projected total | 301.1 + 146.8 × (113/133) × 1.167767 = 446.75 s | < 600 s | PASS |

The JSON `gates` block encodes exactly this pass/fail vector — no discrepancy between raw data, computed gates, and what 03-execute.md transcribed. The early/late ratio derivations use the exact preregistered EXP-011 step weights (10512/10345/2470/2471), and eval timing (ratio 1.167767) is correctly excluded from charged dose and used only in the total-time projection.

## Integrity checks

- **No accuracy exposure**: the JSON contains only conditioning losses/norms (finite, non-selective, explicitly permitted), timing, memory, and EMA update counts. No evaluator call, test accuracy, or `run.log` exists. The evaluator-raises monkeypatch protocol was honored.
- **No rerun/correction after first timing datum**: single decisive preflight, harness hash recorded before launch; the two harness fixes (SHA namespace, `sys.path`) occurred in the earlier CPU verifier, before any timing datum — permitted by the plan's bounded-correction rule.
- **Transcription fidelity**: every figure in 03-execute.md (medians, drifts, dispersions, projections, joint peak 728.967 MiB, conditioning values, fifth-round outliers 1.272699/1.248961) matches the JSON. The claim that the median rejection is independent of the round-5 outliers is correct — dropping round 5 leaves the weighted median ≈1.160 and the ordinary path max 1.16737 still over 1.13.
- **Cross-experiment claim**: width 288's 1.160794 vs EXP-014 width 320's 1.160975 matches 014's report; the "MAC interpolation prior falsified" learning is supported.

## Classification

The failure is a valid, integrity-clean pre-metric feasibility rejection: no scope/evaluator/seed corruption (→ not `invalid`), no accuracy measured (→ not `no-improvement`), no complete metric result (→ `crash`/`NaN` under the tree's mechanical convention, exactly as EXP-014 was encoded). 03-execute.md's stated classification is correct.

**Blockers found: none. False classifications found: none. Verdict to insert: `crash`, metric `NaN`, base node 011 unchanged as global best (95.61%).**
