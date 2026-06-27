# Brainstorm EXP-068
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- No new external sources this loop. The double-screened lit sweep (absorption screen:
  heavy-aug budget-matched evidence; cost-landing screen: price off the charged step) ran in
  brainstorm-066 (~1 day ago) and returned empty; per the steady-state protocol
  (exp-report-067 Next Steps #1) sweeps are periodic at publication-interval, not per-loop.
- **In-scope file audit completed this loop** (per the autopilot think-harder directive:
  "re-read the in-scope files for new angles"): pyproject.toml (full read — legal package
  space is matplotlib/numpy/pandas/pyarrow/torch 2.9.1/torchvision 0.24.1; the non-torch
  packages have no path to the metric; pretrained torchvision weights would be reward
  hacking, not training within budget), README.md (confirms 2×H20 hardware, the ResNet-20
  lineage, and the per-channel-mean normalization decision — consistent with the EXP-066-loop
  by-inspection closure of the Normalize-std gap; nothing unexploited), TASK.md (restates the
  standing hard constraints verbatim; nothing unexploited), prepare.py (re-confirmed
  read-only: Eval pins the test-input distribution). **No new resources found in scope.**

## Experimental History Review

- 69 experiments. Baseline 96.71 @ 1990397 (distribution top); pooled n=5 baseline draws
  {96.71, 96.59, 96.40, 96.53, 96.44} → mean 96.534, σ̂ = 0.123 (EXP-067); bar 96.81 =
  mean + 2.24σ̂; a candidate needs true effect ≥ +0.28 ≈ 2.2σ̂. Last improvement EXP-006.
  External transfer 0-for-21.
- **State**: measured-ceiling AUDIT-COMPLETE (EXP-066 closed the final structural corner;
  EXP-067 re-certified the decision bands with no drift). Steady-state protocol in force:
  spend GPU only on (a) candidates passing BOTH standing screens plus the pre-run inequality,
  probed before launch, or (b) periodic double-screened sweeps; otherwise prefer
  zero-charged-cost closures (probe-gated inequalities, by-inspection audits) over
  re-measuring closed interiors.
- **Near-miss combination pool (directive item "combine previous near-misses") — audited
  this loop, EXHAUSTED**: the only positive-reading components in 69 rows are the
  anti-aliased shortcut (EXP-046/052: pooled +0.11, unresolvable from 0 at affordable n,
  closed permanently) and the H2D-prefetch step saving (EXP-048: +87 steps, did not
  reproduce). EXP-053 already measured their compound (96.445 = at-or-below mean; no
  additivity). No unmeasured pair of near-misses exists.
- **Remaining formally-unclosed space** (directive item "more radical architectural
  changes"): three radical *program-level* classes were never explicitly closed because each
  is dominated by measured arithmetic rather than by a run — (1) sequential self-distillation
  (train teacher then student inside the 300s charged budget), (2) resolution-UP training
  (upsample 32→40px; EXP-031 closed only the DOWN direction), (3) wholesale optimizer-family
  swap (AdamW/Lion/cautious-SGD; only Muon and Schedule-Free were measured). These are the
  classes a fresh reader would flag as "untried"; the ledger should close them explicitly.

## Candidate Ideas

### 1. Radical-class inequality closures at zero charged cost (self-distillation, resolution-up, optimizer-family) + near-miss pool audit
**Summary**: A closure loop in the EXP-063/064 lineage (inequality-gated NO LAUNCH): close
the three remaining radical classes by pre-run inequalities built from measured anchors, and
record the near-miss combination-pool audit, all at zero charged GPU seconds.
- **Sequential self-distillation**: teacher gets ≤150s → ~70 epochs; EXP-043 measured the
  half-budget starvation toll at ~−0.9pp and EXP-002/007 give the same starvation slope; the
  student also trains ~70 epochs (same toll) and must beat the FULL-budget baseline by +0.28.
  Inequality: student ≤ (mean − 0.9) + KD_gain; published KD gains (+0.5–1.0pp) are
  fixed-epoch, never budget-matched-halved (absorption AND cost-landing screens both fail);
  even crediting +1.0 the bound lands ≈ 96.6 < 96.81. Closed by arithmetic.
- **Resolution-up (32→40px bilinear)**: deterministic upsampling adds ZERO information;
  FLOPs ×1.56 with off-lattice spatial sizing (EXP-044/045 mispricing law predicts ≥×1.5 dt)
  → ≤~90 epochs; the measured starvation slope (EXP-002/007: −0.7 to −0.9 per ~50 lost
  epochs) prices the class strictly negative with no offsetting information gain. Closed by
  inspection + measured pricing law.
- **Optimizer-family swap (AdamW/Lion/cautious-SGD)**: absorption screen fails — no
  published ≥0.3pp gain over tuned SGD for ResNet/CIFAR under heavy-aug budget-matched
  comparison; the published record runs the other way (adaptive methods ≤ SGD on small
  CNNs). Subsumption: EXP-028 measured the STRONGEST 2024-25 CNN-speedrun optimizer (Muon,
  airbench-anchored) at exactly baseline mean with a dt toll; weaker family members close
  a fortiori. EXP-062 closed the schedule-free branch. Closed by screen + subsumption.
- **Near-miss pool**: record the audit (pool = {EXP-052 +0.11 unresolvable, EXP-048
  non-reproducing}; compound already measured in EXP-053; pool exhausted).

**Reasoning**: This is literally the steady-state protocol's prescribed move when no
candidate survives the screens: zero-charged-cost closures over re-measuring closed
interiors. It also discharges all three prongs of the autopilot think-harder directive
honestly (papers → sweep stale by 1 day; in-scope files → audited this loop, empty;
combinations → pool exhausted; radical changes → the three classes above, closed by
inequality instead of by burning dominated runs). Each closure cites measured anchors
(EXP-043/002/007 starvation slope, EXP-044/045 pricing law, EXP-028/062 optimizer reads),
not vibes. Pre-registered escape hatch: if any inequality unexpectedly SURVIVES when worked
through with exact numbers, that class immediately becomes the next loop's launch candidate.

**Sources**: exp-report-067 § Next Steps (protocol); EXP-063/064 (inequality-gated NO LAUNCH
precedent, verdict invalid/NaN); EXP-043/002/007 (starvation anchors); EXP-044/045 (pricing
law); EXP-028/062 (optimizer reads); EXP-052/053/048 (near-miss pool).

**Estimated Effort**: low (no GPU; arithmetic worked against ledger anchors; artifacts only).

**Risk Assessment**: No metric movement possible (no launch) — verdict pre-registered
invalid/NaN per EXP-063/064 precedent. Failure mode is only wasted ideation if the
inequalities are sloppy; mitigated by requiring every bound to cite a measured anchor.
The escape-hatch branch converts any surviving inequality into a real candidate.

### 2. Further σ pooling (n=5 → n=7 replicate pair)
**Summary**: Two more zero-diff baseline draws to tighten σ̂ further.

**Reasoning**: The EXP-067 protocol, repeated. But exp-report-067 § Unexplored Avenues
already prices it: σ-of-σ̂ at n=5 is adequate for ±0.3-class screens; only worth it if a
near-bar decision hinges on the third decimal. No such decision is pending.

**Sources**: exp-report-067 § Unexplored Avenues.

**Estimated Effort**: low (2 × ~8 min GPU).

**Risk Assessment**: Safe but information-poor — spends ~16 min of GPU on a digit that
changes no standing decision. Rejected: dominated by Candidate 1 under the steady-state
economics.

### 3. Measure a law-closed residual interior (GhostBN / gradient clipping / TA dose-down)
**Summary**: Directly dose one of the never-individually-measured residuals.

**Reasoning**: Map-completion only. All three sit ON bracketed axes (BN-noise on the peaked
pressure axis; unspendable stability headroom; pressure-down). The EXP-049-class rule —
do not probe a bracketed optimum's interior — applies unchanged from brainstorm-067, where
this same candidate was rejected.

**Sources**: goal-learnings § Failed Approaches (EXP-015/029/033/038/039 closures).

**Estimated Effort**: low-medium.

**Risk Assessment**: Re-measures known laws; strictly less informative than Candidate 1 and
violates standing closure logic. Rejected.

## Idea Evaluation

**Evidence strength**: Candidate 1 rests entirely on measured anchors already in the ledger
(starvation slope measured three times, pricing law measured twice, optimizer family measured
at its strongest member) and follows the protocol adopted with high confidence in
exp-report-067. Candidate 2's own report prices it as not-yet-needed. Candidate 3 contradicts
standing closure logic.

**Mechanism clarity**: Candidate 1's mechanism is arithmetic — each class is shown dominated
by composing measured quantities (budget halving → starvation toll; FLOPs ratio → epoch loss;
strongest-member subsumption). Candidates 2–3 change no decision and test no mechanism.

**Expected impact**: None of the three can move the metric this loop (measured ceiling).
Candidate 1 is the only one that changes the LEDGER: it converts the three classes a fresh
reader would call "untried" into cited closures, and carries an escape hatch that can mint a
real launch candidate if any bound survives.

**Risk profile**: Candidate 1 is the safest possible loop (zero GPU, zero code). Candidate 2
is safe but redundant. Candidate 3 risks nothing but re-learns known results.

**Feasibility**: Candidate 1 is trivial to execute and produces complete artifacts.

## Chosen Idea
**Selected**: Radical-class inequality closures at zero charged cost (self-distillation,
resolution-up, optimizer-family) + near-miss pool audit

**Why this idea**:
It is the steady-state protocol's prescribed move for a no-surviving-candidate loop
(zero-charged-cost closures over re-measuring closed interiors), it discharges every prong
of the autopilot think-harder directive with citations instead of dominated runs, and it is
the only candidate whose output changes anything (the closure ledger + a possible escape-hatch
candidate). EXP-063/064 establish the recording precedent (inequality-gated NO LAUNCH,
verdict invalid, metric NaN).

**Hypothesis**:
Working each class's pre-run inequality through with exact ledger anchors yields a bound
strictly below the bar (96.81) in all three cases — self-distillation ≤ ~96.6 even with the
most generous published KD credit, resolution-up strictly negative (zero information gain at
≥×1.5 dt), optimizer-family ≤ baseline mean by subsumption of its measured strongest member —
so no launch occurs and all three classes close at zero charged cost. If any worked bound
instead clears the bar, that class is promoted to the EXP-069 launch candidate (falsification
branch).
