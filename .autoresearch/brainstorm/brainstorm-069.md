# Brainstorm EXP-069
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- knowledge/README.md re-read: no existing entry covers training-set composition.
- **Targeted adversarial check this loop** (NOT a full periodic sweep — that remains due at a
  publication-scale interval, next ~2026-06-17): does any work show ≥0.3pp gain from curating
  CIFAR-10's NATURAL label noise?
  - Confident Learning (Northcutt et al., arXiv 1911.00068): the often-quoted ~+0.9pp
    CIFAR-10 cleaning gain is measured under 20–40% ADDED synthetic noise — not the natural
    regime. https://arxiv.org/abs/1911.00068
  - Pervasive Label Errors (arXiv 2103.14749): CIFAR-10 test set has 54 CONFIRMED mislabels
    (0.54%); the train set comes from the same labeling pipeline (~0.5–1% suspected). At
    these natural rates no ≥0.3pp cleaning gain is published.
    https://arxiv.org/abs/2103.14749 ; https://l7.curtisnorthcutt.com/label-errors
  - Standing published null already in-ledger: budgeted importance sampling / subset
    selection under fixed budgets (arXiv 2110.14283, surfaced in brainstorm-066) — supports
    the same closure for the selection side of composition.
- Conclusion: no escape-hatch evidence; the data-composition corner is closable by screen +
  arithmetic at zero charged cost.

## Experimental History Review

- 70 experiments. Baseline 96.71 @ 1990397; pooled n=5 mean 96.534, σ̂ 0.123, bar 96.81 =
  mean + 2.24σ̂ (EXP-067). Last improvement EXP-006. External transfer 0-for-21.
- **State after EXP-068**: the three radical classes (sequential KD, resolution-up,
  optimizer-family) are closed by measured-anchor inequalities; near-miss pool exhausted;
  in-scope files audited empty. exp-report-068 Next Steps: sweep only at publication
  interval; candidate inflow only via new double-screened evidence or anchor revision;
  prefer zero-charged-cost closures.
- **The one residual gap in the closure ledger**: goal-learnings § Patterns (GRADIENT-NOISE
  LAW entry) enumerates the out-of-recipe space as "data composition/order, objective
  shaping, architecture." Order is closed (EXP-041), objective shaping is closed
  (EXP-050/051 + EXP-009/036), architecture is closed everywhere (lattice/depth/head/
  kernel/attention/reparam) — but data COMPOSITION (WHICH examples train, vs how they are
  transformed or ordered) was never explicitly closed. EXP-025/033/065 changed the
  augmentation distribution, not the example set. This is the last enumerated member of the
  residual space with no closure citation.

## Candidate Ideas

### 1. Data-composition corner closure at zero charged cost (label-noise curation, subset selection, duplicate handling)
**Summary**: Close the training-set composition class by screens + arithmetic in the
EXP-063/064/068 NO-LAUNCH lineage. Three sub-classes: (i) label-noise curation (drop/fix
the ~0.5–1% naturally mislabeled train examples), (ii) subset selection / importance
sampling / pruning under the fixed budget, (iii) near-duplicate handling. For each, work a
candidate-favoring bound against bar 96.81 from the searched evidence + in-ledger anchors.

**Reasoning**: (i) Label-noise curation fails all three filters: absorption screen — the
published cleaning gain (+0.9pp) is a 20–40% ADDED-noise result; at the natural ~0.5–1%
rate no ≥0.3pp evidence exists, and LS 0.1 (a measured-flat constant, EXP-036) plus heavy
TA+RE already supply the noise-robustness mechanism cleaning would add; cost-landing
screen — identifying errors honestly requires trained-model predictions, which under this
goal's rules means spending charged seconds on ranking passes (importing an external
precomputed error list rides on externally trained models — pretrained-knowledge class,
and even granting it, the effect ceiling stays sub-screen); effect-size — correcting ≤500
of 50,000 labels under LS, where per-sample loss interventions are measured-negative
(EXP-050/051), is a ≤0.1pp-class move ≪ +0.28 required. (ii) Subset selection has a
published NULL precisely in the budget-matched regime (arXiv 2110.14283) and shrinking the
epoch's example count is pressure-down on a bracketed axis (EXP-015/033). (iii) Duplicate
REMOVAL can only hurt: ~3% of test images have train near-duplicates; the official train
set is the distribution Eval pins.

**Sources**: this file § Web Search (1911.00068, 2103.14749, 2110.14283); goal-learnings
§ Patterns (residual-space enumeration), § Failed Approaches (EXP-050/051, EXP-015/033,
EXP-036); exp-report-068 (NO-LAUNCH protocol).

**Estimated Effort**: low (zero GPU; artifact work only).

**Risk Assessment**: No metric movement possible (no launch); verdict pre-registered
invalid/NaN per precedent. Escape hatch: any sub-class whose honest bound clears 96.81 is
promoted to the EXP-070 lead instead of being forced closed.

### 2. Fresh periodic lit re-sweep under the double screen
**Summary**: Re-run the full screened sweep for new 2026 techniques.

**Reasoning**: Last full sweep ran in brainstorm-066 (~1 day ago) and returned empty; the
steady-state protocol sets sweeps at publication-scale intervals. This loop's targeted
adversarial check (above) doubles as partial coverage. Re-sweeping now has near-zero
expected yield.

**Sources**: brainstorm-066 § Web Search; exp-report-067/068 Next Steps.

**Estimated Effort**: low.

**Risk Assessment**: Burns a loop on a near-certain empty result one day after the last
sweep. Rejected on timing — due ~2026-06-17.

### 3. σ pooling n=5 → n=7 replicate pair
**Summary**: Two more zero-diff baseline draws to tighten σ̂.

**Reasoning**: exp-report-067 § Unexplored Avenues prices this as worthwhile only when a
near-bar decision hinges on the third decimal; none pends. Spends ~16 GPU-min on a digit
that changes no standing decision.

**Sources**: exp-report-067 § Unexplored Avenues.

**Estimated Effort**: low.

**Risk Assessment**: Safe but information-poor; dominated by Candidate 1 under steady-state
economics. Rejected.

## Idea Evaluation

**Evidence strength**: Candidate 1 rests on a fresh adversarial search (the strongest
published cleaning gain dissolves on regime inspection: added-noise, not natural), one
published budget-matched null already in-ledger, and measured in-ledger closures
(EXP-036/050/051/015/033). Candidates 2–3 are both priced by their own governing reports as
not-yet-due.

**Mechanism clarity**: Candidate 1's closure mechanism is the program's validated screen
arithmetic: no heavy-aug budget-matched evidence + cost lands on the charged step + effect
ceiling ≤0.1pp ≪ 2.2σ̂. It also has clean LEDGER value: it closes the last enumerated member
of the residual out-of-recipe space, making the closure enumeration complete.

**Expected impact**: None can move the metric this loop (measured ceiling). Candidate 1 is
the only one that changes the ledger state; its escape hatch is the only path that could
mint a launchable candidate.

**Risk profile / Feasibility**: Candidate 1 is zero-GPU, zero-code, trivially feasible, and
fails safest (its "failure" = a surviving bound = a real candidate found).

## Chosen Idea
**Selected**: Data-composition corner closure at zero charged cost

**Why this idea**:
It is the steady-state protocol's prescribed move (zero-charged-cost closure over
re-measuring closed interiors), it targets the single remaining uncited member of the
goal-learnings residual-space enumeration, and this loop's targeted search already
performed the adversarial step — the strongest published counter-evidence dissolved on
regime inspection (added-noise vs natural-noise).

**Hypothesis**:
All three composition sub-classes (label-noise curation, subset selection, duplicate
handling) close with candidate-favoring bounds strictly below bar 96.81 — curation ≤
mean + ~0.1 (effect ceiling at natural noise under LS+TA+RE), selection ≤ mean (published
budget-matched null + pressure-down bracketing), duplicate removal ≤ mean (strictly
information-removing vs the pinned eval distribution) — so no launch occurs and the
out-of-recipe residual space enumeration becomes fully closure-cited. Any surviving bound
instead fires the escape hatch and becomes the EXP-070 lead (falsification branch).
