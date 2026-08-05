# Report EXP-007: Widen layer2 (8×8 stage) 256→384
- **Created**: 2026-06-28

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) within a fixed 300s training budget, editing only `train.py`. Baseline entering this experiment: **96.00%** (EXP-004, commit ae31206). Improvement bar: ≥96.10% (+0.1pp).

## Idea & Hypothesis
Chosen idea (cross-model reviewer's pick): widen the proven 8×8 middle stage `layer2` from 256→384 channels (toward airbench96's documented middle width), adding representational capacity where EXP-004 measured +0.13pp and where cuDNN kernels run at full speed (avoiding EXP-005's 4×4 penalty). The diagnosis was that we are capacity-bound near the ~96.0 ceiling with generous epoch budget to spend (we fit ~150 epochs vs airbench96's 37). Hypothesis: the added width lifts `best_test_acc` to ≥96.10%, with the capacity gain outrunning the throughput-driven epoch loss (projected ~115–130 epochs). Falsifier (pre-registered): best <96.10 with epochs ≤110 → under-annealing (→ try milder 256→320); or epochs adequate but accuracy flat → capacity-saturated / LR mismatch.

## Approach
Two edits in `ResNet9.__init__` (`train.py:150-151`): `layer2` `conv_bn(128,256)→conv_bn(128,384)` + `GatedResidual(256)→GatedResidual(384)`; `layer3` stem `conv_bn(256,512)→conv_bn(384,512)` to match. layer3 output stays 512 → `pool`/`fc=Linear(512,10)` untouched. PEAK_LR=0.4 and everything else held at the EXP-004 recipe (single-variable capacity test). Milestone-1 smoke confirmed shapes flow (layer2 384@8×8, layer3 512@4×4), `num_params == 9,997,235` exactly (hand-computed; +2.21M over EXP-004), and the `layer2[2]` GatedResidual is identity-init (α=0). Deviation from plan: none.

## Execution
One clean run on GPU 1, `timeout 600`, exit 0, wall 439.9s. No retries, no divergence. Training used the full 300.0s but fit only **94 epochs / 9069 steps** — far fewer than projected. img/s oscillated between ~20k peaks and 6–8k dips (vs EXP-004's steady ~26k), reflecting both the wider net's intrinsic cost and shared-host contention (GPU 0 busy).

## Results
- **Primary metric**: 95.85% (baseline: 96.00%, delta: **−0.15pp, −0.16%**) — peak == final at ep94.
- **Observations**:
  1. **Severe under-annealing — the pre-registered failure mode.** Only 94 epochs fit (≤110 cutoff), a −33% drop from EXP-004's 142–150. The widened layer2 (the `layer2[0]` conv at 16×16 widened 256→384, plus the two GatedResidual convs at 8×8 2.25× heavier, plus the layer3 stem 1.5×) cut throughput well below the ~115–130 projection; host contention compounded it.
  2. **The net was STILL CLIMBING at budget exhaustion — not capacity-saturated.** Tail: ep90 95.67 → ep92 95.79 → ep93 95.81 → ep94 95.85 (monotone rise, best==final). Most CIFAR fast-training accuracy lands in the low-LR tail (EXP-001); here that tail was truncated — the model never reached the flat, fully-annealed regime where 96.0 is set.
- **Analysis**: The hypothesis's *capacity premise is not refuted* — indeed the wider net's accuracy was rising and might have matched/exceeded 96.0 with more epochs — but its *prediction failed* because the throughput cost was larger than projected, so the capacity could not be annealed in 300s. This is a configuration/operating-point failure (too-aggressive a widen for the budget on this shared host), not evidence that width doesn't help. It mirrors EXP-005's lesson (capacity that costs too many epochs loses) but via width rather than a slow 4×4 kernel. The decisive lever here is the **capacity/epoch balance**: a 1.5× widen is too steep; a smaller step may sit on the right side of the tradeoff.
- **Key Learning**: Widening layer2 256→384 under-anneals at the 300s budget — it cut epochs 150→94 (accuracy still rising at the end), so the added capacity can't be realized; the capacity-vs-epochs balance, not capacity itself, is the binding constraint, pointing to a milder 256→320 step.

## Verification
- **Conditions**: C1 (clean run within wall guard) PASS — exit 0, one summary line, total 439.9s<600. C2 (full 300s budget + scope/integrity) PASS — training 300.0s, prepare.py byte-unchanged, only train.py changed (diff confined to the two width lines), no stray untracked .py, PEAK_LR=0.4 held, num_params 9,997,235 == smoke. **C3 (improvement ≥+0.1pp) FAIL** — best 95.85 < 96.10 (−0.15pp).
- **Review Notes**: Results trustworthy. Metric genuine (max per-epoch best 95.85 == summary, from `Eval.evaluate`, one eval/epoch, seeds unchanged). No reward-hacking — a legitimate architecture change in train.py. The −0.15pp is real (below even the noise floor's lower edge), driven by the measured 94-epoch under-anneal, not measurement error.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid run; necessary condition C3 failed (metric below bar). No constraint violation (not invalid); clean completion (not crash).

## Unexplored Avenues
- **Milder widen 256→320** (the pre-registered fallback) — recovers much of the epoch loss (~1.25× layer2 cost vs 1.5×) while still adding ~half the capacity; sits on a more favorable point of the capacity/epoch curve. This is the direct, evidence-backed next step.
- **Widen a CHEAPER stage** — adding width at layer1 (16×16) is FLOP-comparable but the 16×16 kernels may be more throughput-efficient than widening the layer2[0] 16×16→pool path; or widen only the GatedResidual branch (8×8) and leave the layer2[0] stem at 256 to limit the 16×16 cost. Untested whether a more surgical width placement preserves epochs.
- **Depth instead of width at layer2/8×8** (brainstorm Idea 3, a 2nd ReZero block) — adds ~1.18M params (vs this experiment's 2.21M) so a smaller epoch hit; still a capacity probe at the proven stage. Lower throughput cost than the 384 widen.
- **Note on confound**: epoch count is partly host-load-dependent (shared GPU 0). A future capacity experiment's epoch budget may differ; the 256→320 step is more robust to this because its throughput hit is smaller.

## Next Steps
1. **Milder widen layer2 256→320** (medium-high confidence) — the pre-registered fallback; directly tests whether a smaller capacity step lands on the right side of the capacity-vs-epochs tradeoff (target ~120–135 epochs). Highest-value next experiment.
2. **Second ReZero block at layer2/8×8** (medium confidence) — a depth-based capacity add with a smaller param/epoch cost (~1.18M) than the 384 widen; the alternative capacity placement if width keeps under-annealing.
3. **Throughput-free recipe alignment (GELU + cutout12)** (low-medium confidence) — brainstorm Idea 2, deferred; costs no epochs, so it sidesteps the under-annealing trap entirely, though each change is likely sub-noise alone. Best as a rider once a capacity step lands, or as a safe probe if capacity placements keep failing on epochs.

## Exit Action Results
<!-- No exit actions defined for this goal. -->
- None — no exit actions defined.
