# EXP-069: Data-composition corner closure at zero charged cost (label-noise curation, subset selection, duplicate handling)

## Execution

Overall Status & Info:
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-069.md
- **Plan**: plans/plan-069.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-069
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Zero-code, zero-GPU loop per plan (EXP-063/064/068 NO-LAUNCH lineage). Milestone 1: branch
autoresearch/exp-069 cut from autoresearch/dev; `git diff --quiet -- train.py` →
ZERO_DIFF_OK; no run.log. Milestones 2–5 below as Run Log entries (i)–(iii) plus the
enumeration-completion check — each composes the brainstorm-069 searched evidence with
in-ledger measured anchors, candidate-favoring on every uncertain quantity, ending in an
explicit branch decision.

### Surprises & Discoveries
- The duplicate-handling sub-class (iii) needed an integrity boundary stated explicitly:
  the SELECTION of which examples to keep must never be test-informed (leakage). Once that
  boundary is drawn, every legal move in the sub-class is information-removing or
  pressure-changing — both bounded ≤ mean without any new evidence.
- Sub-class (i)'s strongest published counter-evidence (+0.9pp from confident-learning
  cleaning) dissolved entirely on regime inspection during the brainstorm search: it is a
  20–40% ADDED-noise result, two orders of magnitude above CIFAR-10's natural rate.

### Decisions
- Pre-registered: verdict invalid/NaN regardless of branch outcomes (no run, no metric);
  any surviving bound promotes to the EXP-070 lead instead of launching here.
- The imported-error-list variant of sub-class (i) (hard-coding a precomputed mislabel list
  in train.py) is treated as pretrained-knowledge class (it encodes the predictions of
  externally trained models) — but the closure does NOT hinge on that classification: the
  effect-size ceiling alone (≤0.1pp at natural noise) fails the +0.28 screen.

## Experimental Adjustments

- None — all anchors re-read as cited; no bound required re-derivation.

## Run Log

### Entry i (Milestone 2): label-noise curation (drop/relabel ~0.5–1% natural train errors)

Metadata:
- **Job ID**: N/A (no launch) | **Log file(s)**: this entry | **WandB**: N/A
- **Status**: completed (closure) | **Started/Ended**: 2026-06-11

Description:
- Bound the best achievable gain from training on a curated 50k train set with natural
  label errors removed or corrected, against bar 96.81.

Worked bound (anchors cited):
- Absorption screen FAILS: the published cleaning gain (~+0.9pp, Confident Learning,
  arXiv 1911.00068) is measured under 20–40% ADDED synthetic noise; CIFAR-10's natural
  error rate is ~0.54% CONFIRMED in the test set (54 mislabels, arXiv 2103.14749) with the
  train set from the same labeling pipeline; no ≥0.3pp natural-rate cleaning result exists
  (brainstorm-069 § Web Search, adversarial check run this loop).
- Mechanism redundancy: LS 0.1 — measured FLAT on [0.1, 0.2] (EXP-036) — plus heavy TA+RE
  already supply label-noise robustness; per-sample loss-side interventions are measured
  negative in BOTH directions (EXP-050 −2.4σ, EXP-051 −7.8σ).
- Cost-landing screen FAILS: honest in-budget error identification requires trained-model
  predictions (ranking passes price ON the charged budget); the imported-list variant rides
  on externally trained models (pretrained-knowledge class) — and even granting it free and
  legitimate, the next line caps the effect.
- Effect-size ceiling: ≤500 of 50,000 labels (~1%) corrected, each diluted across ~13.5k
  steps under LS-capped targets → ≤0.1pp class ≪ +0.28 = 2.2σ̂ required (EXP-067
  arithmetic). Candidate-favoring: even granting 3× the natural rate, the ceiling stays
  < +0.28.
- Bound: ≤ mean 96.534 + ~0.1 = 96.63 < 96.81.

**Branch decision: CLOSED.**

### Entry ii (Milestone 3): subset selection / importance sampling / dataset pruning

Metadata:
- **Job ID**: N/A (no launch) | **Log file(s)**: this entry | **WandB**: N/A
- **Status**: completed (closure) | **Started/Ended**: 2026-06-11

Description:
- Bound any scheme that trains on a selected/weighted subset of the 50k examples under the
  fixed budget, against bar 96.81.

Worked bound (anchors cited):
- Published budget-matched NULL: budgeted importance sampling shows no gain under fixed
  budgets (arXiv 2110.14283; surfaced brainstorm-066, re-cited brainstorm-069) — this is
  the rare external result measured in OUR regime (budget-matched), and it is null.
- In-ledger mechanism closures: shrinking the per-epoch example pool is pressure-DOWN on
  the bracketed pressure axis (EXP-015 −0.30; pressure must be constant-on at the certified
  level through the last step — four-quadrant law EXP-025/033/065); selection WEIGHTING
  keyed to difficulty/confidence is the measured anti-curriculum mechanism (EXP-051 −7.8σ;
  EXP-065 easy-first −1.2σ). Both directions of "train on the informative subset" are
  measured losers.
- Throughput note: smaller epochs do not buy steps — the budget is time-keyed, steps/s is
  unchanged; selection only changes WHICH images fill the same 13.5k steps, i.e., the
  effective sampling distribution — a composition-side pressure change on a closed axis.
- Bound: ≤ mean 96.534 < 96.81.

**Branch decision: CLOSED.**

### Entry iii (Milestone 4): near-duplicate handling

Metadata:
- **Job ID**: N/A (no launch) | **Log file(s)**: this entry | **WandB**: N/A
- **Status**: completed (closure) | **Started/Ended**: 2026-06-11

Description:
- Bound removal or re-weighting of train-set near-duplicates (incl. the ~3% of test images
  with train near-duplicates, ciFAIR analyses), against bar 96.81.

Worked bound (anchors cited):
- Integrity boundary (stated, not used as a gain path): any selection USING test-set
  information is leakage and banned by the no-eval-gaming constraint — so the legal moves
  are test-blind removals/re-weightings only.
- Removal: strictly information-removing relative to the distribution Eval pins
  (prepare.py read-only); train-test near-duplicates are exactly the training examples
  most predictive of test points — removing them can only lower best_test_acc; removing
  train-internal duplicates re-weights the empirical distribution = composition-side
  pressure change on the bracketed axis (Entry ii mechanism).
- Re-weighting/duplication: same closed mechanism (EXP-051/065 weighting closures).
- Bound: ≤ mean 96.534 < 96.81 (removal variants strictly below).

**Branch decision: CLOSED.**

### Entry iv (Milestone 5): residual-space enumeration completion

Metadata:
- **Job ID**: N/A | **Log file(s)**: this entry | **WandB**: N/A
- **Status**: completed (audit) | **Started/Ended**: 2026-06-11

Record:
- goal-learnings § Patterns enumerates the out-of-recipe residual space as "data
  composition/order, objective shaping, architecture". Closure citations now exist for
  every member: data ORDER — EXP-041; objective SHAPING — EXP-050/051 + EXP-009/036;
  ARCHITECTURE — width/lattice (EXP-040/044/045), depth/allocation (EXP-008/017/034/061),
  head/routing (EXP-030/037/047), kernels (EXP-066), shortcuts (EXP-020/046/052), blocks
  (EXP-056), reparam (EXP-064), attention (EXP-037), ensembles (EXP-042/043/063); radical
  program classes — EXP-068; data COMPOSITION — this loop (entries i–iii).
- **The enumeration is COMPLETE: every member carries at least one closure citation.**

## Verification Results

### Conditions Checked

- **Condition 1 — best_test_acc ≥ bar 96.81**: NOT SATISFIABLE — no run launched, no metric
  exists. Recorded metric = NaN; verdict pre-registered **invalid** per the EXP-063/064/068
  NO-LAUNCH precedent (plan-069 § Verification Procedure). Remaining conditions not
  evaluated for the verdict (first-failure stop).
- **Condition 2 — run ≤ 600s**: vacuous; substitute check PASS — `ls run.log` → No such
  file or directory (checked at setup and loop end).
- **Condition 3 — validation once per epoch**: vacuous; structurally guaranteed (zero diff).
- **Loop-specific integrity (plan § Verification 4)**: PASS — (a) `git diff --quiet --
  train.py` → ZERO_DIFF_OK at setup and loop end; (b) every bound quantity carries an EXP
  citation or brainstorm-069 § Web Search URL; (c) entries i–iii each state an explicit
  branch decision (3× CLOSED, 0× ESCAPE-HATCH); (d) entry iv lists a closure citation for
  every residual-space member.

### Informational Metrics

- Worked bounds vs bar 96.81: (i) ≤ 96.63; (ii) ≤ 96.534; (iii) ≤ 96.534 (removal variants
  strictly below). None within 0.18 of the bar under candidate-favoring arithmetic.
- Escape-hatch status: none fired (0 of 3).
- Charged GPU seconds: 0. Probe seconds: 0. Code diff: zero. /tmp scripts: 0.

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
