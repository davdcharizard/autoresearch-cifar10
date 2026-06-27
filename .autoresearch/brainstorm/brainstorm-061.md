# Brainstorm EXP-061
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

Fresh external sweep per exp-report-060 Next Steps (the internal seam space is exhausted; check what the wall-clock-regime-matched lineage added post-2024):

- **cifar10-airbench current state** (https://github.com/KellerJordan/cifar10-airbench; https://kellerjordan.github.io/posts/speedrun/)
  Post-paper additions to the speedrun lineage: (1) **Muon optimizer** (94% record) — measured here EXP-028, closed (arrival-time benefit, zero plateau gain, worse basin); (2) **lookahead averaging** (every 5 steps) — weight-averaging class, closed EXP-011/032 (the time-keyed anneal already performs implicit iterate averaging); (3) **data filtering** (airbench96_faster).
- **airbench96_faster data filtering — source inspection** (https://raw.githubusercontent.com/KellerJordan/cifar10-airbench/master/airbench96_faster.py)
  Mechanism: online hard-example selection — pool batches of 1024, rank by per-sample loss, gradient step only on the hardest 512 (`mask[loss1.argsort()[-batch_size_masked:]]`), with masks precomputed by a small PROXY model (widths 32/64 vs main 128/384), refreshed every 4 steps. **Decisive datum: it is accuracy-NEUTRAL in its own regime — 96.00 @ 27.3s filtered vs 96.03 @ 34.7s unfiltered.** It buys TIME, not LEVEL. Under our fixed charged budget the time dividend must convert via extra epochs, which requires an unchanged data distribution (EXP-031) — but the mechanism IS a distribution change; and binary per-sample selection-by-loss is inside the EXP-050/051 per-sample-treatment closure (focal-class explicitly do-not-retry), with a noise-scale change on top (closed bidirectionally). Triple-screened.
- **Hive cifar10-speedrun (1.98s)** (https://github.com/hiverge/cifar10-speedrun)
  Improvement is Muon update vectorization — pure engineering of a closed optimizer class; nothing new for LEVEL.
- Conclusion: the external regime-matched frontier offers no law-passing LEVEL construction either. Ideation below therefore mines the internal record for unprobed DIRECTIONS of measured slopes — and finds one (EXP-017's allocation slope, never probed in its favorable direction).

## Experimental History Review

- **Current best**: 96.71 @ 1990397; family mean ≈ 96.57, σ ≈ 0.16; bar 96.81; TRUE effects need ≥ +0.3pp. 61 experiments, 6 improvements (all ≤ EXP-006), 54 consecutive closures.
- **EXP-060 closed the last open SEAM** (regularizer type): absorption now extends to augmentation TYPE; external transfer 0-for-17.
- **The one unprobed direction of a measured slope — depth allocation toward stage 1**: EXP-017 ([2,3,4] stage-3-heavy at equal FLOPs) lost −0.28 with a fully-converged plateau, and its mechanism reading **isolated "less stage-1 depth" specifically** as the deficit ("high-resolution early-stage blocks carry irreplaceable representation at shallow depth" — project-insights § Low). The mirror allocation **[4,3,2] (stage-1-heavy) was never run**. The ResNet stage-design invariant makes all blocks FLOPs-equal (~151M each: halved spatial × doubled channels), so [4,3,2] is FLOPs-neutral; the EXP-034 per-block law (~2.5ms/block, width-independent) predicts dt-neutral at 9 blocks; params DROP to 3,179,338 (−26%, verified arithmetically — the same formula reproduces the uniform net's 4,286,026 exactly).
- **Laws screen** for [4,3,2]: deferral — no new heat-priced components (fewer params, same kernel families, dt probe-checked); numerics — same dense 3×3 kernels on lattice widths {64,128,256}; noise — unchanged; heat/schedule — unchanged; max-statistic — a plateau-LEVEL question (the right kind); absorption — NOT an external import, internal reallocation (the 0-for-17 record does not apply); allocation-position is a recorded first-class variable (project-insights EXP-017 entry).
- **Adjacent negatives to respect**: depth-width plane closed for TOTAL depth changes (EXP-008/034: ±blocks cost dt and/or representation); width lattice fixed (64/128/256 unique); capacity INCREASES closed 6× — note [4,3,2] is a capacity DECREASE in params at constant FLOPs/depth, a never-visited corner.

## Candidate Ideas

### 1. Stage-1-heavy depth reallocation [3,3,3] → [4,3,2] at equal FLOPs and depth (lattice widths unchanged)
**Summary**: Change the per-stage block counts from uniform (3,3,3) to (4,3,2): four 64-wide blocks at 32×32, three 128-wide at 16×16, two 256-wide at 8×8. Total depth stays 20, every width stays on the hardware-fast lattice, every block keeps the standard BasicBlock topology and pad shortcuts. FLOPs are exactly preserved (stage invariant); params drop 4,286,026 → 3,179,338. Implementation is a ~3-line change (`_make_layer` already takes a per-stage count; pass (4,3,2)).

**Reasoning**: This is the only structural direction in the record with a measured FAVORABLE slope behind it. EXP-017 ran the exact mirror ([2,3,4]) under identical accounting and lost −0.28 with a converged plateau — and the analysis attributed the deficit specifically to the missing stage-1 block, not to the extra stage-3 capacity (which was +1.1M params of free capacity that returned nothing). If stage-1 high-resolution representation is the binding resource at depth 20 (consistent with EXP-008: depth loss hurts via representation; and with EXP-047: late-stage features are weak class-linearly), moving a block INTO stage 1 is the highest-prior structural move available. The cost side — losing one 256-wide block — is bounded by EXP-017's own observation that ADDED stage-3 capacity was worthless at the margin; the risk is asymmetric in our favor relative to any other untested structural move. A family-band read completes a clean 3-point allocation curve ([2,3,4] −0.28 / [3,3,3] mean / [4,3,2] X) and closes the allocation axis bidirectionally.

**Sources**: reports/exp-report-017.md (mirror + mechanism isolation); project-insights § Low (allocation POSITION first-class, EXP-017); EXP-034 per-block dt law; EXP-008 (depth-representation); param arithmetic verified this brainstorm.

**Estimated Effort**: low — ~3-line model change; standard probe + composite launcher; param assert updates (3,179,338).

**Risk Assessment**: (a) Diminishing returns: the 4th stage-1 block may add less than the 3rd carried — family-band read likely under the null. (b) Stage-3 capacity loss could dominate despite EXP-017's zero-marginal-value reading (WRN evidence is about WIDTH, kept intact here, not block count — but 2 blocks may under-process the 256-wide features) — a sub-floor read would close the axis from below, still informative. (c) dt surprise: per-block law says neutral, but stage-1 blocks carry 4× the activations of stage-3 blocks — MUST gate-probe before launch (protocol mandates dt-gating every shape change). (d) VRAM up somewhat (more 32×32 activations) — soft constraint, fine. No crash/invalid pathway.

### 2. Online hard-example selection (airbench96_faster-style data filtering: forward the batch, backward only the hardest half)
**Summary**: Per step, compute per-sample losses on the full 512 batch, run backward only on the hardest 256 — saving ~30% of step time (backward ≈ 65% of cost), converting to ~+40 epochs under the fixed budget.

**Reasoning**: The newest technique in the regime-matched lineage; would test whether informative-example compute allocation beats uniform.

**Sources**: airbench96_faster.py (inspected above); Selective Backprop (arXiv 1910.00762); RHO-loss class.

**Estimated Effort**: medium (masked backward interacting with the compiled graph; selection plumbing).

**Risk Assessment**: TRIPLE-SCREENED by measured closures: (1) binary loss-ranked selection is a per-sample loss treatment — the EXP-050/051 closure lists focal-class (upweight-hard) as do-not-retry, and EXP-051 measured −7.8σ for the continuous variant; (2) the time dividend converts only at an unchanged data distribution (EXP-031), which the mechanism itself violates; (3) gradient noise scale changes (closed bidirectionally). Decisively, the technique is accuracy-NEUTRAL in its OWN regime (96.00 vs 96.03) — it is a throughput trick, and the throughput→epochs conversion here is exhausted (EXP-048: charged step 99.3% irreducible; conversion law conditional). Not runnable on honest priors.

### 3. Lookahead slow-weights averaging (k=5, α=0.5, airbench-style)
**Summary**: Maintain slow weights updated every 5 steps as an interpolation toward the fast weights; eval the slow weights.

**Reasoning**: Present in the current airbench recipes; cheap to implement.

**Sources**: airbench96_faster.py; Zhang et al. 2019 (arXiv 1907.08610).

**Estimated Effort**: low.

**Risk Assessment**: SCREENED: lookahead is iterate averaging — the weight-averaging axis is closed from both ends (EMA −0.25 EXP-011; canonical SWA exactly at mean EXP-032) with the mechanism understood (the time-keyed cosine anneal already performs the equivalent averaging implicitly; solution smoothing moves calibration, not argmaxes). Also adds eval-contract plumbing (which weights `base_model` exposes per-epoch) for a zero-prior candidate. Not runnable.

## Idea Evaluation

Ideas 2 and 3 are screened out by measured closures before cost is even considered — both are members of classes with multiple negative or null measurements AND carry the airbench lineage's own evidence of accuracy-neutrality (Idea 2) or this project's direct mechanism explanation for why they cannot raise the plateau (Idea 3). Running either would re-measure a closure.

Idea 1 is the only candidate with a measured directional slope in its favor and no closure covering its direction. Evidence: the mirror experiment lost −0.28 under identical accounting with the deficit isolated to exactly the resource this candidate adds (stage-1 high-res depth), and the resource it spends (a stage-3 block) measured ≈ zero marginal value in that same mirror. Mechanism: clear — reallocate equal-FLOPs compute from where its marginal representational value measured lowest to where the record says it is highest. Expected impact: the only available candidate whose effect-size prior derives from an in-project measured slope (~0.3-class for one-block moves) rather than a heavily discounted external import. Risk: graceful in both failure directions (family-band completes the allocation curve; sub-floor closes from below); the single hard requirement is the dt gate-probe (per-block law predicts neutral but stage-1 blocks carry 4× activations — protocol mandates pricing every shape change). Feasibility: ~3-line model edit; everything else is the standard harness.

## Chosen Idea
**Selected**: Stage-1-heavy depth reallocation [3,3,3] → [4,3,2] at equal FLOPs and depth

**Why this idea**:
After EXP-060 the catalogued axes, seams, and the external regime-matched frontier are all measured-closed or accuracy-neutral; the one direction never probed is the favorable side of EXP-017's allocation slope — the only structural variable in the record with measured evidence pointing TOWARD a specific untested configuration. It passes every standing law without exception (FLOPs-equal, lattice widths, depth 20, no external import, no noise/heat/numerics/loss change), costs ~3 lines, and is informative under every pre-registerable outcome: it either clears the bar, completes a 3-point allocation curve closing the axis bidirectionally, or closes it from below.

**Hypothesis**:
Reallocating one block from stage 3 (256-wide, 8×8, ~1.18M params, marginal value measured ≈ 0 in EXP-017's mirror) to stage 1 (64-wide, 32×32, the resource whose removal cost −0.28) will raise best_test_acc to ≥ 96.81 IF stage-1 high-resolution representational depth is the binding constraint at depth 20 under this recipe — the model gains a fourth round of full-resolution feature refinement at zero FLOPs/dt cost. If the allocation curve is flat-topped at uniform (diminishing returns on the 4th stage-1 block ≈ the marginal value of the 3rd stage-3 block), the run reads family band [96.41, 96.73], completing the allocation bracket. If stage-3 block COUNT (unlike its added params) is load-bearing for processing the final features, the run reads < 96.41 and the axis closes from below.
