# Report EXP-070: Gap-completion closure loop — infra/throughput angle + data-addition bound
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-070.md
- **Plan**: plans/plan-070.md
- **Log**: logs/exp-log-070.md

## Goal
Maximize best_test_acc (%) within the fixed 300s charged budget. Baseline 96.71 @ 1990397;
bar 96.81 (= pooled mean 96.534 + 2.24σ̂, EXP-067). Closure loop in the NO-LAUNCH lineage,
targeting the last two nominated-but-unbounded classes in the ledger: the infra/throughput
angle (never searched before — all prior sweeps were ML-technique-scoped) and data addition
(nominated in exp-report-069 § Unexplored Avenues).

## Idea & Hypothesis
Close both gaps with candidate-favoring bounds: (a) infra — establish whether any legal
execution-stack change offers the ≥11% numerics-IDENTICAL speedup that bar-clearing via the
conversion law requires; (b) data addition — bound each funding path (in-budget generation,
external import, augmentation-like synthesis). Hypothesis: both close sub-bar; any survivor
promotes to the EXP-071 lead (the infra side via an EXP-064-style probe).

## Approach
Zero code diff, zero GPU. The brainstorm performed the program's first infra-scoped search
(PyTorch 2.9 release notes/blog, GitHub releases, issue tracker). Both bounds then composed
that evidence with measured laws (EXP-006 conversion, EXP-021 numerics, EXP-048 overhead
floor, EXP-043 dilution, EXP-009/060 pressure closures) under standing anchor discipline.

## Execution
No runs, no retries, no errors; all anchors re-read clean; both entries plus the backlog
check completed first-pass.

## Results
- **Primary metric**: NaN (no launch; baseline: 96.71, delta: N/A)
- **Observations**:
  - **Bound A (infra/throughput) — CLOSED**: required gain = +0.28 = +15 epochs ≈ −2.4ms on
    the 22.4ms step ≈ 11% speedup that must be numerics-identical (EXP-021 measured that
    numerics-DIFFERENT speedups lose more than they convert, replicated). Availability:
    PyTorch 2.9's conv work is AArch64-scoped, no new CUDA conv path; the known 2.9.0 conv
    regression is 3D+AMP (irrelevant). The knob class that exists (cuDNN algo selection) is
    numerics-different by construction AND its marginals at an already-benchmarked,
    compile-floored config are an order below 2.4ms. Gain ceiling ≤ +0.05-class ≪ +0.28.
  - **Bound B (data addition) — CLOSED on all three funding paths**: in-budget generation
    prices on the charged meter (EXP-043 dilution: −0.9 per halving vs +0.3–0.5 gains);
    external/generated import crosses both the pretrained-knowledge boundary and TASK.md's
    isolation rationale (leakage class — banned, not merely sub-screen); augmentation-like
    synthesis is the measured-closed pressure axis in dose and type (EXP-009/060).
  - **Backlog check (the loop's ledger product)**: the closure chain EXP-068 (radical
    classes) → EXP-069 (composition + enumeration completion) → EXP-070 (infra + data
    addition) leaves NO nominated-but-unbounded class in the ledger. Next scheduled inflow:
    the periodic double-screened sweep due ~2026-06-17.
- **Analysis**: A ledger-sharpening came out of Bound A: EXP-048's 99.3%-kernel bound
  closes OVERHEAD recovery but not kernel-math substitution; the kernel-math side is closed
  by EXP-021 (numerics law) plus this loop's required-gain inequality (11% vs searched-empty
  availability). The two laws compose — neither alone closed the throughput escape, which
  is why this angle deserved its own search and entry. The program is now in its cleanest
  steady state: every named class carries a closure citation, the idle-loop policy is
  written down, and the only open channel is scheduled external inflow.
- **Key Learning**: The throughput escape is shut by composition — bar-clearing needs an
  11% numerics-identical speedup, the searched stack offers none, and the knobs that exist
  are numerics-different (the measured EXP-021 trap); data addition closes on all funding
  paths; the explicit-bound backlog is empty.

## Verification
- **Conditions**: Condition 1 not satisfiable (no run; metric NaN) — verdict invalid as
  pre-registered; Conditions 2–3 vacuous, substitute checks PASS (no run.log; zero diff at
  setup and loop end). Loop-specific integrity PASS: all quantities cited, both entries
  carry explicit CLOSED decisions, zero escape hatches, backlog chain enumerated with the
  2026-06-17 inflow date.
- **Review Notes**: trustworthy — citation-fidelity-only surface; primary sources fetched
  this loop, in-ledger anchors re-read clean.
- **Verdict**: invalid
- **Verdict Basis**: inequality-gated NO LAUNCH (EXP-063/064/068/069 precedent) — the
  product is two class closures plus the empty-backlog certification.

## Unexplored Avenues
- A future PyTorch/cuDNN release could ship a genuinely numerics-identical conv speedup
  (same algorithm, better implementation); the 2026-06-17+ sweeps should include the infra
  modality now that it exists — its closure is availability-dependent, unlike the
  arithmetic closures.
- Bound A's required-gain arithmetic (11%) is reusable for ANY future throughput proposal:
  cite it before probing.

## Next Steps
1. Hold the steady state until 2026-06-17, then run the periodic double-screened sweep
   INCLUDING the new infra modality alongside the ML-technique queries. Confidence high.
2. Until then, remaining idle actions per policy: anchor meta-audit only when anchors are
   stale (re-read clean three times this week — not yet), instrument replication only when
   a near-bar decision pends (none does). If a loop fires before the sweep date with
   neither condition met, the honest spend is the smallest useful ledger work available —
   do not manufacture GPU runs to look busy. Confidence high.
3. When the sweep runs, any survivor goes through the pre-run inequality + EXP-064
   internal-control probe before launch. Confidence high.

## Exit Action Results
