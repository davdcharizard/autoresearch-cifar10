# Brainstorm EXP-070
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- knowledge/README.md re-read (label-errors entry added EXP-069; nothing covers infra).
- **Infra-angle targeted check this loop** — a search modality never tried: all prior sweeps
  were ML-technique-scoped; this one asked whether PyTorch 2.9 / cuDNN / H20 offer a
  NUMERICS-PRESERVING throughput option (the one legal throughput escape under the EXP-021
  numerics-equivalence law, not excluded by EXP-048's overhead bound which covers only
  non-kernel time):
  - PyTorch 2.9 release notes/blog: compile improvements are suite-level and Arm/AArch64
    conv-focused; no new CUDA conv lowering or numerics-preserving speed option.
    https://pytorch.org/blog/pytorch-2-9/ ; https://github.com/pytorch/pytorch/releases/tag/v2.9.0
  - Known 2.9.0 conv issue is 3D-conv + AMP regression (irrelevant: our convs are 2D, and
    family dt 22.4ms is ledger-stable across the whole program).
    https://github.com/pytorch/pytorch/issues/166122
  - Remaining legal knob class (cuDNN algo selection, e.g. benchmark_limit): a different
    conv algorithm is faster-but-DIFFERENT arithmetic — exactly the measured EXP-021 trap
    (max-autotune: +10 epochs, −0.20). Required-gain inequality regardless: clearing the
    bar by conversion (+0.019/ep, EXP-006 law) needs ≥ +15 epochs ≈ −2.4ms/step ≈ 11%
    numerics-identical speedup — far beyond algo-benchmarking marginals at an
    already-cudnn.benchmark'd, compile-floored config (EXP-048: 99.3% kernel math).
- Next FULL periodic double-screened sweep remains due ~2026-06-17.

## Experimental History Review

- 71 experiments. Baseline 96.71 @ 1990397; pooled mean 96.534, σ̂ 0.123, bar 96.81 =
  mean + 2.24σ̂ (EXP-067). Last improvement EXP-006. External transfer 0-for-21.
- **State after EXP-069**: residual-space enumeration COMPLETE (composition/order, objective
  shaping, architecture — all closure-cited); radical classes closed (EXP-068); steady-state
  idle-loop policy recorded in goal-learnings (sweep on schedule, close on nomination,
  replicate only when a near-bar decision pends).
- **Explicitly noted open gaps** (exp-report-069 § Unexplored Avenues + this loop's search):
  (a) data ADDITION (synthetic/generated training examples) — inherits existing closures but
  was never explicitly bounded; (b) the infra/throughput angle above — searched this loop,
  closes by the numerics law + required-gain inequality, needs recording.

## Candidate Ideas

### 1. Gap-completion closure loop: infra/throughput angle + data-addition bound (zero charged cost)
**Summary**: Record two explicit closures in the NO-LAUNCH lineage: (a) **infra angle** —
no numerics-preserving speed option exists in the legal stack (searched above), and the
knob class that does exist (cuDNN algo selection) fails the pre-run inequality (needs ~11%
numerics-identical speedup; EXP-021 measured that numerics-DIFFERENT speedups lose more
than they convert); (b) **data addition** — in-budget generation prices its compute ON the
charged step (cost-landing fail); importing generated/external data rides on externally
trained models (pretrained-knowledge class, same boundary as EXP-069's imported-list
variant) and external-data import is also outside the task's data scope; augmentation-like
in-budget synthesis (mixing/erasing variants) is the measured-closed pressure/type axis
(EXP-009/060). With these, BOTH gaps noted in the ledger are closed and the explicit-bound
backlog is empty.

**Reasoning**: This is the steady-state policy's "close on nomination" branch applied to
the only two nominated-but-unbounded classes in the record. The infra angle is genuinely
new information (first infra-scoped search of the program; its closure needs the searched
evidence, not just old anchors). Pre-registered escape hatch: if either bound survives
honest arithmetic, it becomes the EXP-071 lead (the infra side would get an EXP-064-style
internal-control probe before any launch).

**Sources**: this file § Web Search (pytorch.org/blog/pytorch-2-9, GH releases v2.9.0,
GH issue 166122); EXP-021/048/006 (numerics law, overhead bound, conversion law);
exp-report-069 § Unexplored Avenues (data-addition nomination); EXP-009/060 (pressure/type
closures); goal-learnings § Protocol Findings (idle-loop policy).

**Estimated Effort**: low (zero GPU; artifact work).

**Risk Assessment**: No metric movement possible (no launch); verdict pre-registered
invalid/NaN per NO-LAUNCH precedent. Failure mode is sloppy bounds — mitigated by anchor
discipline; escape hatch prevents forced closure.

### 2. Full periodic lit sweep, early
**Summary**: Run the complete double-screened technique sweep now instead of ~2026-06-17.

**Reasoning**: Two days after the last full sweep, expected yield ≈ 0; the policy interval
exists precisely to avoid this spend. The infra-scoped check (done this loop) already
covers the only untried modality.

**Sources**: brainstorm-066 § Web Search; goal-learnings idle-loop policy entry.

**Estimated Effort**: low-medium.

**Risk Assessment**: Near-certain empty; weakens the policy's discipline. Rejected on
timing.

### 3. Load-bearing anchor meta-audit (σ̂ arithmetic, starvation ladder, absorption law, dt floor, max-statistic law)
**Summary**: Adversarially re-verify the five anchors most cited by closure loops against
their primary reports, looking for internal contradictions.

**Reasoning**: Hardens the closure ledger's foundations. But every anchor was re-read from
its source during EXP-068/069's anchor-discipline checks within the last day, and no bound
came within 0.18 of the bar — even a moderate anchor revision would not flip a verdict. A
meta-audit pays off only after time has passed or when a bound is near-bar.

**Sources**: exp-log-068/069 § Experimental Adjustments (all anchors re-read clean).

**Estimated Effort**: low-medium.

**Risk Assessment**: Redundant with checks just performed; defer until anchors are stale or
a near-bar bound appears. Rejected for now.

## Idea Evaluation

**Evidence strength**: Candidate 1's infra half rests on a fresh primary-source search
(release notes, issue tracker) plus two measured laws (EXP-021/048) and the measured
conversion law; its data half composes closures already validated in EXP-069. Candidate 2
is priced empty by its own governing policy. Candidate 3 re-does day-old checks.

**Mechanism clarity**: Candidate 1's bounds are arithmetic: required numerics-identical
speedup (11%) vs available marginals at a measured 99.3%-kernel floor; generation compute
vs the charged meter; import vs the pretrained-knowledge boundary.

**Expected impact**: None can move the metric (measured ceiling). Candidate 1 empties the
explicit-bound backlog — after it, the ledger has NO nominated-but-unbounded class, which
is the cleanest possible state for future sessions.

**Risk profile / Feasibility**: Candidate 1 is zero-GPU, zero-code, fails safest (a
surviving bound = a found candidate), trivially feasible.

## Chosen Idea
**Selected**: Gap-completion closure loop: infra/throughput angle + data-addition bound

**Why this idea**:
It applies the standing policy's close-on-nomination branch to the only two open
nominations, one of which (infra) required and received a genuinely new search modality
this loop. It leaves the ledger with an empty explicit-bound backlog until the scheduled
2026-06-17 sweep.

**Hypothesis**:
Both bounds close strictly sub-bar under candidate-favoring arithmetic: (a) infra — the
legal stack offers no numerics-preserving speedup ≥11% (searched: none exists; algo-class
knobs are numerics-different, EXP-021 measured-negative), so the throughput escape stays
shut; (b) data addition — every funding path (in-budget generation, external import,
augmentation-like synthesis) lands on a measured closure or a banned class, bounding the
gain ≤ 0 vs mean. If either bound survives, it is promoted to the EXP-071 lead with a
probe-gated inequality (falsification branch).
