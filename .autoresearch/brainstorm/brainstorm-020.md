# Brainstorm EXP-020
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **SWA — Izmailov et al. 2018 (knowledge/papers/swa.md)**: the SWA solution quality depends on the constant
  LR used in the averaging phase. Too high → the iterate strays far from good minima (each snapshot is low
  top-1, so the average's top-1 is capped even if it is flatter/lower-loss); too low → the iterate barely
  moves and the average reproduces a single near-settled point (the EXP-006 no-op regime). There is an interior
  sweet spot. The paper also recommends BN-recompute for the average (already implemented in EXP-019).
- No new external search needed — EXP-019 established the mechanism works on this project; this loop tunes its
  one free hyperparameter (the floor LR) toward the sweet spot.

## Experimental History Review

Current best = **96.22%** (EXP-012). 19 experiments; ~12 axes closed. Binding constraint: generalization at
fixed k=4 capacity in 300s.

**Directly relevant — EXP-019 (the immediately prior loop, a −0.25pp near-miss)**: proper SWA with a constant
**0.05**-LR averaging tail (start 0.75, 24 snapshots, BN-recomputed) → **95.97%**, the LOWEST test loss in the
project (0.1788 << baseline 0.195). The SWA mechanism ENGAGED exactly as theorized (recovered the un-annealed
raw iterate from 91.8% → 95.97% via averaging) and the curve was STILL RISING at the final epoch — but it fell
0.25pp short of the cosine-to-0 top-1 baseline. Diagnosis (exp-report-019): the 0.05 floor was likely too HIGH
— the raw iterate cratered to 91.83% (ep 67) at that LR, so each averaged snapshot was individually low-top-1,
capping the average. The flatter/lower-loss-but-not-higher-top-1 outcome is a known SWA signature.

Goal-learnings now records weight-averaging as Medium "near-closed (count 2: EXP-006 EMA-on-cosine-to-0 no-op;
EXP-019 proper SWA −0.25pp)", with the explicit remaining refinements: **lower SWA_LR (≈0.02), later start
(≈0.85), or a cyclic-LR tail** — "the best remaining lead (EXP-019 was only −0.25pp, curve still rising)."

Closed axes (do NOT revisit): capacity k>4, LR-peak (0.2 optimum), block-order/pre-act, activation, SE
attention, weight decay, more-epochs, auto-aug policy, aug-strength-down, label-mixing aug (Mixup/CutMix).
Untried gap this loop targets: the SWA floor-LR sweet spot — a single-variable refinement of a proven near-miss.

## Candidate Ideas

### 1. SWA with a lower constant-LR floor (SWA_LR 0.05 → 0.02)
**Summary**: Repeat EXP-019's proper-SWA setup EXACTLY (cosine PEAK_LR→floor over [5%,75%], constant floor for
the final 25%, per-epoch weight averaging, truncated BN-recompute, SWA model evaluated in the tail), changing
ONLY the floor LR from 0.05 to **0.02**. Everything else identical: SWA_START_FRAC=0.75 (keeps the ~24-snapshot
tail that was still improving), BN_RECOMPUTE_BATCHES=50, k=4 recipe, compile, seed 42. A single-variable change.

**Reasoning**: EXP-019 was a −0.25pp near-miss whose SWA curve was still rising, and the diagnosis points
squarely at the floor being too high (raw iterate at 0.05 cratered to 91.8%, capping snapshot quality). Lowering
the floor to 0.02 (a) makes each snapshot individually higher-top-1 (closer to a well-converged solution, since
0.02 is near where baseline cosine sits around ~80% of training) while (b) keeping enough movement (0.02 = 10%
of peak) for the average to still land in a flat region rather than collapsing to a single point. This targets
the exact knob the literature identifies as controlling SWA solution quality. Clean attribution: any delta vs
EXP-019's 95.97 is attributable to the floor.

**Sources**: exp-report-019.md (diagnosis + Next Steps); goal-learnings Medium weight-averaging entry;
knowledge/papers/swa.md (floor-LR sweet-spot); project-insights Medium SWA entry.

**Estimated Effort**: low — a one-constant change to the already-implemented EXP-019 code (SWA_LR=0.05→0.02).

**Risk Assessment**: If 0.02 is too low the iterate barely moves → the average approaches a single
constant-0.02 endpoint, which lacks cosine-to-0's final anneal-to-~0 sharpening → likely lands ~96.0–96.1
(still no-improvement, between EXP-019's 95.97 and the no-op regime). Fails gracefully. Assumption: 0.02 sits
above the no-op floor but below 0.05's over-exploration — plausible but not guaranteed; this run pins it down.

### 2. Cyclic-LR SWA tail (SGDR-style sawtooth instead of a constant floor)
**Summary**: Replace the constant-floor tail with a short cyclic LR (e.g. sawtooth between ~0.05 and ~0.005),
snapshotting the average at each cycle minimum — the original SWA paper's stronger variant.

**Reasoning**: Cyclic LR samples more diverse, more-separated minima, so their average can be flatter/better than
a constant-LR tail's correlated samples. Could exceed the constant-floor result.

**Sources**: knowledge/papers/swa.md; exp-report-019 Unexplored Avenues.

**Estimated Effort**: medium — add a cyclic LR phase + snapshot-at-cycle-minimum logic to the tail.

**Risk Assessment**: More moving parts (cycle length, amplitude, snapshot timing) → more ways to mis-tune within
one shot; the constant-floor result was already close, so the marginal benefit of cyclic over a well-tuned
constant floor is uncertain. Higher implementation risk than Idea 1 for similar expected ceiling.

### 3. Lookahead optimizer (weight interpolation, keeps cosine-to-0 intact)
**Summary**: Wrap SGD in Lookahead (Zhang et al. 2019): maintain fast weights (inner SGD) and slow weights;
every k steps set slow += α(fast − slow) and reset fast = slow. Keep the original cosine-to-0 schedule.

**Reasoning**: A different weight-averaging sub-mechanism that operates THROUGHOUT training and is compatible
with cosine-to-0 — so it sidesteps the SWA tension (no need to give up the annealed endpoint). Targets
generalization variance, compute-neutral.

**Sources**: Lookahead (arXiv:1907.08610); project-insights Medium SWA entry; exp-report-019.

**Estimated Effort**: medium — manage slow/fast weight copies and periodic interpolation in the loop.

**Risk Assessment**: Same weight-averaging FAMILY the goal-learnings flag as near-closed; reported CIFAR gains
are typically small and often within noise, so expected top-1 lift is marginal. Most likely a within-noise null.

## Idea Evaluation

**Evidence strength**: Idea 1 has the strongest, most specific evidence — it refines a PROVEN −0.25pp near-miss
along the exact axis the literature names as controlling SWA quality, with a clean single-variable change. Ideas
2 and 3 are more speculative (cyclic tuning surface; Lookahead's marginal CIFAR gains).

**Mechanism clarity**: Idea 1 has the clearest, most directly-testable mechanism (floor-LR → snapshot quality
↔ movement trade-off). Idea 2's mechanism is sound but multi-knob. Idea 3's is sound but its expected magnitude
is small.

**Expected impact**: Idea 1 highest expected value among the three for clearing +0.1pp, given EXP-019 was only
0.25pp short and still rising. Idea 2 comparable ceiling but higher mis-tune risk in one shot. Idea 3 likely null.

**Risk profile**: All fail gracefully (no-improvement). Idea 1 has the smallest blast radius (one constant) and
the cleanest attribution.

**Feasibility**: Idea 1 is trivial (one constant on already-working code); Ideas 2/3 are medium.

Conclusion: **Idea 1 (SWA_LR → 0.02)** dominates on evidence specificity, mechanism clarity, attribution
cleanliness, and feasibility. It is the highest-EV single shot at converting the EXP-019 near-miss. If it also
falls short, the weight-averaging axis is conclusively closed for this budget (Idea 2/3 become low-priority).

## Chosen Idea
**Selected**: SWA with a lower constant-LR floor (SWA_LR 0.05 → 0.02)

**Why this idea**:
EXP-019 proved SWA engages on this model (project-lowest loss 0.1788) and missed the bar by only 0.25pp with the
SWA curve still rising — the single most promising near-miss in the project's recent history. Its own diagnosis
isolates the floor LR as too high (the 0.05 raw iterate cratered to 91.8%, capping snapshot quality). Lowering
the floor to 0.02 is the direct, single-variable, literature-aligned test of that diagnosis: it should raise
each snapshot's individual top-1 while preserving enough iterate movement for the average to remain flat.

**Hypothesis**:
Re-running EXP-019's SWA setup with the constant-LR floor lowered from 0.05 to 0.02 (all else identical) will
raise best_test_acc above EXP-019's 95.97 and plausibly past the 96.32 bar, because the averaged snapshots will
be individually higher-top-1 (LR nearer the well-converged region) while still moving enough for flat-region
averaging to help. If instead it lands ~96.0–96.1 (movement too small → approaching a single constant-0.02
endpoint without cosine-to-0's final sharpening), the SWA floor sweet spot does not clear cosine-to-0 at this
budget and the weight-averaging axis is conclusively closed.
