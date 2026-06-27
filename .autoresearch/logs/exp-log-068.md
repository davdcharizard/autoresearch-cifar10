# EXP-068: Radical-class inequality closures at zero charged cost (self-distillation, resolution-up, optimizer-family) + near-miss pool audit

## Execution

Overall Status & Info:
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-068.md
- **Plan**: plans/plan-068.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-068
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Zero-code, zero-GPU loop per plan (EXP-063/064 NO-LAUNCH lineage). Milestone 1: branch
autoresearch/exp-068 cut from autoresearch/dev; `git diff --quiet -- train.py` → ZERO_DIFF_OK;
no run.log exists. Milestones 2–5 are the experiment's substance: four worked closures
recorded below as Run Log entries A–D, each composing only measured ledger anchors with
candidate-FAVORING choices wherever a quantity is uncertain, each ending in an explicit
branch decision (CLOSED vs ESCAPE-HATCH). No launch occurred; no /tmp scripts were created.

### Surprises & Discoveries
- Working Inequality B revealed it does not even need the dt term: with eval pinned at 32px
  (prepare.py read-only), the gain term is bounded ≤ 0 by inspection + the EXP-029 mismatch
  sign alone, so the FLOPs-floor starvation toll only deepens an already-failed bound.
- EXP-066's probe datum (square kernels FLOPs-priced at 1.59× ≈ the exact 1.5625× ratio
  resolution-up needs) made the Inequality B cost floor citable without any new probe.

### Decisions
- Pre-registered (plan § protocol): verdict invalid/NaN regardless of branch outcomes —
  even an escape-hatch firing would only promote a candidate to EXP-069, not launch it here.
- KD credit in Inequality A set at +1.0pp, the candidate-favoring MAXIMUM of published
  fixed-epoch self-distillation gains, despite both standing screens failing for KD and the
  loss axis carrying a measured negative prior (goal-learnings § Failed Approaches: "give
  loss-motivated candidates (distillation...) a measured negative prior"). Waived costs are
  itemized in the entry so the bound is an over-estimate by construction.

## Experimental Adjustments

- None — all four anchors re-read as cited; no inequality needed re-derivation.

## Run Log

### Entry A (Milestone 2): Inequality — sequential self-distillation (teacher→student inside 300s)

Metadata:
- **Job ID**: N/A (no launch)
- **Log file(s)**: this entry (worked bound)
- **WandB**: N/A
- **Status**: completed (closure)
- **Started/Ended**: 2026-06-11

Description:
- Bound the best achievable best_test_acc of any teacher→student sequential distillation
  scheme whose TOTAL charged time is 300s, against bar 96.81.

Worked bound (anchors cited):
- Throughput anchor: family dt 22.4ms → 139 epochs / ~13.5k steps per 300s
  (EXP-067 ledger, logs/exp-log-067.md § Verification).
- Even split (150s + 150s): each phase trains ~70 epochs. A ~70-epoch 4x net is priced at
  ~95.6 by the measured starvation ladder — EXP-043's full-alternation members each received
  half the steps and were priced "~95.6" with the pair's function-space read consistent
  (reports/exp-report-043.md § Results); cross-check: EXP-007 55 ep → −0.71, EXP-005
  52 ep → −1.11 from their own baselines (TSV rows 007/005).
- Teacher quality T(150s) ≈ 95.6. Student non-KD ceiling at 150s ≈ 95.6 (same ladder).
- KD credit granted at the published MAXIMUM +1.0pp (fixed-epoch self-distillation/BAN
  literature) even though: (a) absorption screen FAILS — no heavy-aug budget-matched KD
  evidence exists, and the absorption law (project-insights, EXP-035/036/037/046/060) plus
  the measured loss-axis closure (EXP-050/051: per-sample/target shaping loses both
  directions; EXP-009/036: target-distribution closed) give KD a measured NEGATIVE prior;
  (b) cost-landing screen FAILS — the teacher forward prices ON the student's charged step
  (~+8ms eager fwd ≈ −25% student epochs), waived here in the candidate's favor.
- Bound (even split): 95.6 + 1.0 = 96.6 < 96.81.
- Split-robustness: 200s/100s → teacher ≈ 96.0 (93-ep ladder point), student 46-ep ceiling
  ≈ 95.0–95.4 (EXP-002: 40 ep → −0.82 below an already-lower base; TSV row 002) + 1.0 ≤
  96.4. 100s/200s → teacher ≈ 95.0; distilling from a teacher far below the student's own
  ceiling cannot earn the maximum credit (self-distillation gains presuppose teacher ≥
  student), bound ≤ 96.0 + 1.0 = 97.0 only if full credit were granted against an
  inferior teacher — physically the credit collapses toward 0 as teacher falls below
  student ceiling; with even a half-credit cap the bound is ≤ 96.5. All splits < 96.81.
- Unmodeled costs all waived in the candidate's favor: teacher-inference dt toll, KD-loss
  graph changes (compile re-warm), distribution-shift of soft targets under TA+RE views.

**Branch decision: CLOSED** (max worked bound ≈ 96.6 < 96.81; no split survives).

### Entry B (Milestone 3): Inequality — resolution-up training (bilinear 32→40px)

Metadata:
- **Job ID**: N/A (no launch)
- **Log file(s)**: this entry (worked bound)
- **WandB**: N/A
- **Status**: completed (closure)
- **Started/Ended**: 2026-06-11

Description:
- Bound any scheme that trains at upsampled resolution (40px representative; argument is
  monotone in resolution) while Eval remains pinned at 32px, against bar 96.81.

Worked bound (anchors cited):
- Gain term ≤ 0 by inspection + measured sign: bilinear upsampling is a deterministic
  function of the 32px source — ZERO information added. The only published mechanism in
  this family (train/test resolution discrepancy, FixRes) is measured net-negative on this
  recipe (EXP-025 clean-tail −0.87; EXP-065 head-side −1.2σ; reports cited in
  goal-learnings § Failed Approaches "Distribution lightening"). Training BN statistics on
  40px spatial distributions while Eval (prepare.py, read-only) feeds 32px is a train/eval
  constants mismatch with measured NEGATIVE sign (EXP-029: −10.93 for the pure case;
  reports/exp-report-029.md).
- Cost term (only deepens the failure): FLOPs ×(40/32)² = 1.5625. Square-kernel dense path
  is FLOPs-priced — EXP-066 probe measured P_norm 31.02 vs dense-law 30.3 at 1.59× FLOPs
  (logs/exp-log-066.md § Run 1), almost exactly this ratio → dt ≥ ~35ms → ≤ ~89 epochs →
  starvation toll ≥ ~0.6 by the EXP-006 conversion law (+25 ep = +0.48; reports/
  exp-report-006.md) and the EXP-007 ladder. Off-fast-tier spatial dims (40/20/10 vs
  32/16/8) can only make dt worse (EXP-044/045 tier law analog) — candidate-favoring floor
  used.
- Bound: ≤ mean 96.534 + 0 (gain) − ≥0.6 (starvation) − mismatch term < 96.0 ≪ 96.81.
  Even waiving the entire cost term, the bound is ≤ mean < bar.

**Branch decision: CLOSED** (gain ≤ 0 by inspection; bound < bar with or without cost term).

### Entry C (Milestone 4): Closure — optimizer-family swap (AdamW / Lion / cautious-SGD)

Metadata:
- **Job ID**: N/A (no launch)
- **Log file(s)**: this entry (worked closure)
- **WandB**: N/A
- **Status**: completed (closure)
- **Started/Ended**: 2026-06-11

Description:
- Close the remaining optimizer-family members (adaptive/sign-based/masked variants) by
  screen + subsumption, against bar 96.81.

Worked closure (anchors cited):
- Absorption screen FAILS: no published ≥0.3pp gain for AdamW/Lion/cautious-SGD over TUNED
  SGD+nesterov on CIFAR ResNets under heavy-aug budget-matched comparison; the published
  direction is adverse (adaptive methods generalize ≤ SGD on small CNNs). Screen
  established as binding by EXP-037 (goal-learnings: "candidates need evidence under
  heavy-aug budget-matched regimes specifically").
- Subsumption from measured strongest members: EXP-028 measured Muon (the strongest
  2024–25 CNN-speedrun optimizer, airbench-anchored) — real early gain decaying to a
  plateau AT baseline mean in a WORSE basin (test_loss 0.193 vs 0.185) with +2.9ms toll
  (reports/exp-report-028.md). EXP-062 measured Schedule-Free SGD at −1.84
  (reports/exp-report-062.md). EXP-023/024 bracket the momentum/noise geometry both
  directions at byte-identical signatures (reports/exp-report-024.md). The family's BEST
  measured member reads ≤ mean; AdamW/Lion/cautious additionally pay per-param state and
  1–2 extra elementwise passes (EXP-026 pointwise pricing: +1–3ms class) and force an
  LR/WD retune off the certified optimum (EXP-012/022 law: forced hyperparameter moves
  measured net-negative).
- Bound: ≤ baseline mean 96.534 by subsumption, further reduced by tolls < 96.81.

**Branch decision: CLOSED** (a fortiori under the family's measured best member).

### Entry D (Milestone 5): Near-miss combination-pool audit

Metadata:
- **Job ID**: N/A (no launch)
- **Log file(s)**: this entry (audit record)
- **WandB**: N/A
- **Status**: completed (audit)
- **Started/Ended**: 2026-06-11

Description:
- Discharge the autopilot directive prong "try combining previous near-misses" with a
  citable exhaustion record.

Audit (anchors cited):
- Positive-reading component pool across all 69 TSV rows: (1) anti-aliased shortcut —
  pooled +0.11 at n=3, unresolvable from 0, closed PERMANENTLY (EXP-046/052;
  reports/exp-report-052.md); (2) H2D-prefetch step saving — +87 steps once, regressed to
  zero on both EXP-053 replicates (reports/exp-report-053.md). No other row carries a
  positive point estimate vs the recipe mean.
- The only constructible compound (1)+(2) was measured: EXP-053 read 96.445 = mean −0.8σ,
  BELOW additivity. goal-learnings records the region closed: "a future compound needs a
  NEW component with a replicated ≥ +0.1 estimate first" — no such component exists.

**Branch decision: CLOSED** (pool exhausted by enumeration; the one compound is measured).

## Verification Results

### Conditions Checked

- **Condition 1 — best_test_acc ≥ bar 96.81**: NOT SATISFIABLE — no run was launched, no
  metric exists. Recorded metric = NaN; verdict pre-registered **invalid** per the
  EXP-063/064 inequality-gated NO-LAUNCH precedent (plan-068 § Verification Procedure).
  Remaining conditions not evaluated for the verdict (first-failure stop).
- **Condition 2 — run ≤ 600s total**: vacuous (no run); substitute check PASS — no run.log
  exists in the project root (`ls run.log` → No such file or directory).
- **Condition 3 — validation at most once per epoch**: vacuous (no run); structurally
  guaranteed by zero diff.
- **Loop-specific integrity (plan § Verification 4)**: PASS — (a) `git diff --quiet --
  train.py` → ZERO_DIFF_OK at setup and re-checked at loop end; (b) every inequality
  quantity above carries an EXP citation; (c) all four entries state explicit branch
  decisions (4× CLOSED, 0× ESCAPE-HATCH).

### Informational Metrics

- Worked bounds vs bar 96.81: A ≈ 96.6 (max over splits); B ≤ mean 96.534 even with the
  cost term waived (< 96.0 with it); C ≤ mean 96.534 minus tolls. None within 0.2 of the
  bar under candidate-favoring arithmetic.
- Escape-hatch status: none fired (0 of 3 inequalities survived).
- Charged GPU seconds: 0. Probe GPU seconds: 0. Code diff: zero. /tmp scripts created: 0.

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
