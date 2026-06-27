# Report EXP-069: Data-composition corner closure at zero charged cost
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-069.md
- **Plan**: plans/plan-069.md
- **Log**: logs/exp-log-069.md

## Goal
Maximize best_test_acc (%) within the fixed 300s charged budget. Baseline 96.71 @ 1990397;
bar 96.81 (= pooled mean 96.534 + 2.24σ̂, EXP-067). Closure loop in the EXP-063/064/068
NO-LAUNCH lineage, targeting the single uncited member of the goal-learnings residual
out-of-recipe space: training-set COMPOSITION (which examples train — distinct from order,
EXP-041, and from augmentation distribution, EXP-025/033/065).

## Idea & Hypothesis
Close the composition class by screens + arithmetic in three sub-classes — (i) label-noise
curation, (ii) subset selection / importance sampling / pruning, (iii) near-duplicate
handling — each with a candidate-favoring bound vs the bar, after a targeted adversarial
search for escape-hatch evidence. Hypothesis: all three close sub-bar; any surviving bound
promotes to the EXP-070 lead.

## Approach
Zero code diff (verified at setup and loop end), zero GPU seconds, no /tmp scripts. A
targeted adversarial WebSearch ran during brainstorming (not a full periodic sweep): the
strongest published counter-evidence — confident learning's ~+0.9pp CIFAR-10 cleaning gain
(arXiv 1911.00068) — dissolved on regime inspection: it is a 20–40% ADDED-synthetic-noise
result, while CIFAR-10's natural error rate is ~0.54% confirmed (54 test-set mislabels,
arXiv 2103.14749), two orders of magnitude lower.

## Execution
No runs, no retries, no errors. All three bounds plus the enumeration-completion check
worked first-pass; the abort criterion (an anchor failing on re-read) never fired.

## Results
- **Primary metric**: NaN (no launch; baseline: 96.71, delta: N/A)
- **Observations**:
  - **(i) Label-noise curation — CLOSED, bound ≤ 96.63**: absorption screen fails (no
    natural-rate ≥0.3pp evidence); LS 0.1 (measured flat, EXP-036) + TA+RE already supply
    the robustness mechanism; per-sample loss interventions measured negative both
    directions (EXP-050/051); effect ceiling ≤0.1pp at ≤1% natural noise even granting 3×
    the confirmed rate; honest identification also prices ON the charged budget
    (cost-landing fail), and the imported-list variant rides on externally trained models.
  - **(ii) Subset selection — CLOSED, bound ≤ mean**: the rare external result measured in
    OUR regime (budget-matched importance sampling, arXiv 2110.14283) is a published NULL;
    pool-shrinking is pressure-down on the bracketed axis (EXP-015, four-quadrant law);
    difficulty/confidence weighting is the measured anti-curriculum mechanism (EXP-051
    −7.8σ, EXP-065 −1.2σ). Time-keyed budget means selection cannot buy steps — only the
    sampling distribution changes, a closed axis.
  - **(iii) Near-duplicate handling — CLOSED, bound ≤ mean (removals strictly below)**:
    test-informed selection is leakage (banned); test-blind removal strictly deletes
    information about the distribution Eval pins; re-weighting inherits (ii)'s closures.
  - **Enumeration completion (the loop's ledger product)**: every member of the
    goal-learnings residual space ("data composition/order, objective shaping,
    architecture") now carries a closure citation — composition was the last.
- **Analysis**: The measured-ceiling reading reaches its strongest form: the residual-space
  enumeration written into goal-learnings when recipe-space first closed is now EXHAUSTED
  with citations, not assertions. Two methodological notes: (1) the targeted adversarial
  search pattern (one question, run at brainstorm time, looking specifically for
  escape-hatch evidence) is cheaper than a periodic sweep and directly serves the closure
  being attempted; (2) published gains must be regime-checked on NOISE RATE as well as
  augmentation and budget — the +0.9pp cleaning figure is real but lives at 20–40% noise,
  a regime two orders of magnitude away from this dataset's natural state.
- **Key Learning**: Training-set composition closes like every other axis — the recipe's
  existing mechanisms (LS + heavy aug) already absorb label-noise robustness, selection has
  a budget-matched published null, and duplicate removal is strictly information-removing
  against a pinned eval distribution.

## Verification
- **Conditions**: Condition 1 not satisfiable (no run; metric NaN) — verdict invalid as
  pre-registered; Conditions 2–3 vacuous, substitute checks PASS (no run.log; zero diff).
  Loop-specific integrity PASS: anchors all cited, three explicit CLOSED decisions, zero
  escape hatches, enumeration list complete.
- **Review Notes**: trustworthy — the only integrity surface is citation fidelity; all
  in-ledger anchors re-read this session, external claims grounded in the brainstorm's
  searched sources.
- **Verdict**: invalid
- **Verdict Basis**: inequality-gated NO LAUNCH (EXP-063/064/068 precedent) — no metric by
  design; the product is the composition closure plus enumeration completion.

## Unexplored Avenues
- A measured falsification of sub-class (i) would require the dominated run the bound
  exists to avoid; it becomes worthwhile only if heavy-aug budget-matched natural-noise
  cleaning evidence ≥0.3pp is published.
- Data ADDITION (synthetic/generated examples) was not separately bounded: generating
  in-budget costs charged time (cost-landing), and importing generated data rides on
  external models (pretrained-knowledge class) — both inherit existing closures, but a
  future loop could write the two-line bound explicitly if the class is ever nominated.

## Next Steps
1. The next periodic double-screened lit sweep is due ~2026-06-17 (one week after
   brainstorm-066's); until then, loops should target instrument work or any remaining
   explicit-bound gaps (e.g., the data-addition note above) at zero charged cost.
   Confidence high.
2. With the enumeration complete, define the steady-state idle-loop policy explicitly in
   goal-learnings: sweep on schedule, close on nomination, replicate only when a near-bar
   decision pends — so future sessions inherit the policy without re-deriving it.
   Confidence high.
3. If the 2026-06-17 sweep surfaces a double-screen survivor, run it through the pre-run
   inequality with a probe (EXP-064 internal-control pattern) before any launch.
   Confidence high.

## Exit Action Results
