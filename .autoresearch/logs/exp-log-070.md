# EXP-070: Gap-completion closure loop — infra/throughput angle + data-addition bound (zero charged cost)

## Execution

Overall Status & Info:
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-070.md
- **Plan**: plans/plan-070.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-070
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Zero-code, zero-GPU loop per plan (NO-LAUNCH lineage). Milestone 1: branch
autoresearch/exp-070 cut from autoresearch/dev; `git diff --quiet -- train.py` →
ZERO_DIFF_OK; no run.log. Milestones 2–4 recorded below as Run Log entries A, B, and the
backlog check — composing the brainstorm-070 primary-source search (the program's FIRST
infra-scoped search) with measured laws, candidate-favoring throughout, each ending in an
explicit branch decision.

### Surprises & Discoveries
- Working Bound A sharpened a distinction the ledger had only implicitly: EXP-048's 99.3%
  bound closes OVERHEAD recovery, but a faster conv ALGORITHM would change kernel math
  itself — not excluded by EXP-048. The actual closure there is the EXP-021 numerics law
  (different kernels = different arithmetic = measured-negative) PLUS the required-gain
  inequality (≥11% numerics-identical speedup needed; nothing of the kind exists in the
  searched stack). The two laws compose; neither alone suffices.

### Decisions
- Pre-registered: verdict invalid/NaN regardless of branch outcomes; a surviving Bound A
  would get an EXP-064-style internal-control probe in the NEXT loop, not a launch here.
- TASK.md's isolation rationale ("isolate the experiments from any remote branches that
  might contain any information on how to improve the baseline, which would be a source of
  data leakage") is read as the task's data boundary for Bound B's import path: external
  data/model-derived artifacts are exactly the leakage class the isolation exists to block.

## Experimental Adjustments

- None — all anchors re-read as cited; no bound required re-derivation.

## Run Log

### Entry A (Milestone 2): Bound — infra/throughput angle (numerics-preserving speedup availability)

Metadata:
- **Job ID**: N/A (no launch) | **Log file(s)**: this entry | **WandB**: N/A
- **Status**: completed (closure) | **Started/Ended**: 2026-06-11

Description:
- Bound the metric gain available from any legal execution-stack change (compile options,
  cuDNN knobs, library versions within the pinned deps), against bar 96.81.

Worked bound (anchors cited):
- Required gain: clearing the bar via throughput needs true effect ≥ +0.28 = +15 epochs by
  the conversion law (+0.019/ep, EXP-006; reports/exp-report-006.md) ≈ −2.4ms on the
  22.4ms step ≈ **11% speedup that must also be numerics-IDENTICAL** (EXP-021: faster-but-
  different arithmetic measured −0.20 WITH +10 epochs delivered, replicated;
  reports/exp-report-021.md).
- Availability (searched this loop, brainstorm-070 § Web Search): PyTorch 2.9 release
  notes/blog show no new CUDA conv path (conv work is AArch64-scoped); the known 2.9.0
  conv regression is 3D+AMP — irrelevant (2D convs; family dt 22.4ms ledger-stable across
  the program). Sources: pytorch.org/blog/pytorch-2-9, github.com/pytorch/pytorch/releases/
  tag/v2.9.0, github.com/pytorch/pytorch/issues/166122.
- Knob class that does exist (cuDNN algo selection, e.g. benchmark_limit): a different
  algo is by construction different reduction-order arithmetic → EXP-021 class; and the
  candidate-favoring ceiling for algo-benchmarking marginals at an already-
  cudnn.benchmark'd, compile-floored config (EXP-048: charged step 99.3% kernel math,
  recoverable overhead ~0.15ms) is far below 2.4ms. Precision side bracketed flat
  (EXP-054 fp16 ≈ bf16 at equal speed; EXP-021 closed the coarser side).
- Note (ledger sharpening): EXP-048 closes overhead; EXP-021 + this inequality close the
  kernel-math side. Bound: available numerics-identical speedup ≪ required 11% → gain
  ≤ +0.05-class ≪ +0.28.

**Branch decision: CLOSED** (availability searched-empty; knob class numerics-different;
required-gain inequality fails by an order of magnitude).

### Entry B (Milestone 3): Bound — data addition (synthetic/generated training examples)

Metadata:
- **Job ID**: N/A (no launch) | **Log file(s)**: this entry | **WandB**: N/A
- **Status**: completed (closure) | **Started/Ended**: 2026-06-11

Description:
- Bound the metric gain from adding examples to the training set, by funding path, against
  bar 96.81.

Worked bound (anchors cited):
- Path 1 — in-budget generation: any generator trained or run within the 300s prices its
  compute ON the charged meter; diverting charged seconds from the model's own steps is
  the measured dilution arithmetic (EXP-043: −0.9 per halving vs +0.3–0.5 class gains;
  reports/exp-report-043.md). Even a "free" generator (e.g., the model itself, pseudo-
  labeling its own augmented views) adds forward passes to the charged step (cost-landing
  fail, EXP-064 law) and its labels carry no information beyond the model's own state.
- Path 2 — imported generated/external data: rides on externally trained models
  (pretrained-knowledge boundary — same classification as EXP-069's imported-list variant)
  and crosses the task's data/isolation boundary (TASK.md § Git Rules rationale: external
  improvement information = leakage). Banned class, not merely sub-screen.
- Path 3 — augmentation-like in-budget synthesis (mixes/erasures/policy transforms of
  existing examples): this is the measured-closed pressure axis in dose AND type (EXP-009
  mixup stacking −0.46; EXP-060 CutMix substitution null; four-quadrant pressure-profile
  law EXP-025/033/065). Already at the measured optimum.
- Bound: every funding path lands on a measured closure or a banned class → gain ≤ 0 vs
  mean 96.534 < 96.81.

**Branch decision: CLOSED** (all three funding paths).

### Entry C (Milestone 4): Explicit-bound backlog check

Metadata:
- **Job ID**: N/A | **Log file(s)**: this entry | **WandB**: N/A
- **Status**: completed (audit) | **Started/Ended**: 2026-06-11

Record:
- Closure chain: EXP-068 (radical classes: sequential KD, resolution-up, optimizer family;
  + near-miss pool exhaustion) → EXP-069 (data composition; residual-space enumeration
  completed) → EXP-070 (infra/throughput angle; data addition). **No nominated-but-
  unbounded class remains in the ledger.**
- Next scheduled candidate inflow: the periodic double-screened lit sweep due ~2026-06-17
  (goal-learnings § Protocol Findings, idle-loop policy). Until then the policy's
  remaining idle actions are instrument work (only if a near-bar decision pends — none
  does) or anchor meta-audit once anchors are stale (re-read clean three times this week).

## Verification Results

### Conditions Checked

- **Condition 1 — best_test_acc ≥ bar 96.81**: NOT SATISFIABLE — no run launched, no
  metric. Recorded metric = NaN; verdict pre-registered **invalid** per the
  EXP-063/064/068/069 NO-LAUNCH precedent. Remaining conditions not evaluated for the
  verdict (first-failure stop).
- **Condition 2 — run ≤ 600s**: vacuous; substitute check PASS — `ls run.log` → No such
  file or directory (setup and loop end).
- **Condition 3 — validation once per epoch**: vacuous; structurally guaranteed (zero diff).
- **Loop-specific integrity (plan § Verification 4)**: PASS — (a) ZERO_DIFF_OK at setup and
  loop end; (b) every quantity carries an EXP citation or brainstorm-070 URL; (c) entries
  A and B state explicit branch decisions (2× CLOSED, 0× ESCAPE-HATCH); (d) entry C
  enumerates the EXP-068→069→070 chain with next inflow 2026-06-17.

### Informational Metrics

- Bound A: available numerics-identical speedup ≪ required 11% (−2.4ms of 22.4ms);
  gain ceiling ≤ +0.05-class vs +0.28 required. Bound B: ≤ 0 vs mean on all three paths.
- Escape-hatch status: none fired (0 of 2).
- Charged GPU seconds: 0. Probe seconds: 0. Code diff: zero. /tmp scripts: 0.

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
