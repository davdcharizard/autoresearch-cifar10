# Brainstorm EXP-019
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **94% on CIFAR-10 in 3.29 Seconds** (arXiv 2404.00498 — https://arxiv.org/html/2404.00498v2; code https://github.com/KellerJordan/cifar10-airbench)
  The first conv is initialized as a PATCH-WHITENING transform: filters = eigenvectors of the patch covariance (scaled to give identity output covariance), with the second half as NEGATIONS of the first so information survives the following activation. Used in every airbench variant (93–96%). Crucially, this evidence comes from a WALL-CLOCK-BUDGET regime (seconds-scale speedruns) — the same metric semantics as ours, unlike the fixed-epoch sources that failed to transfer in EXP-016/017/018.
- **hlb-CIFAR10** (https://github.com/tysam-code/hlb-CIFAR10)
  Independent speedrun lineage using the same whitening-init first conv; corroborates the technique is load-bearing across implementations, not an airbench idiosyncrasy.
- **Understanding the Covariance Structure of Convolutional Filters** (arXiv 2210.03651)
  Trained first-layer filters converge toward data-statistics-aligned structures — whitening init starts the net where training would otherwise spend early heat getting to.
- **In-project (knowledge/papers/bag-of-tricks-zero-gamma.md, EXP-018)**: init interventions priced in early heat (stability headroom) INVERT under fixed wall clock; only init changes that ADD early learning escape the objection. Whitening init is precisely that: information at init, zero expressivity removed.

## Experimental History Review

- **Current best**: 96.71 @ 1990397 (EXP-006 recipe). Thirteen consecutive no-improvements (EXP-007…018).
- **Closed axes**: all training constants (certified single-knob optimum); schedule in every dimension; uniform capacity both ways; per-stage depth allocation (EXP-017); regularization both sides; batch; EMA smoothing; zero-γ init (EXP-018: −0.99pp — identity-at-init is capacity deferral; promoted the deferral insight to project-insights HIGH with 7 confirmations across 4 mechanism classes).
- **The sharpened selection criterion** (EXP-017+018): the 300s regime rewards ONLY interventions that increase what is LEARNED PER UNIT OF SCHEDULE HEAT. Free params (EXP-017) and easier optimization (EXP-018) both failed; the early high-LR phase is the most valuable real estate in the run and nothing may defer or waste it.
- **Untried gaps**: (a) information-adding init — whitening first conv, explicitly exempted from the EXP-018 closure (goal-learnings Low: "only init changes that ADD early learning escape the objection"); (b) width asymmetry 64/128/320 (exp-report-017/018 runner-up — preserves early depth but pays ~14 epochs); (c) cosine LR floor (flagged in exp-report-018 with a reward-hacking caution: must raise the plateau MEAN, not its variance).

## Candidate Ideas

### 1. Whitening init for conv1 (data-derived eigenvector filters, learnable)
**Summary**: Initialize the stem conv (3→64, 3×3) with patch-whitening filters computed at startup from the training data: sample ~100k random 3×3 patches from `train_set.data` (normalized with our mean/std), compute the 27×27 patch covariance, eigendecompose, and set filters to v_i/√(λ_i+ε) — 27 whitening directions plus their 27 NEGATIONS (airbench's trick so ReLU/BN preserves both signs) = 54 filters, the remaining 10 keep Kaiming init. Weights stay LEARNABLE (freezing would change the optimizer surface; airbench freezes, but our least-invasive variant keeps training dynamics untouched). Architecture, graph, dt, params all unchanged — init-only, same perfect-attribution property as EXP-018. Startup cost ~1–2s (one covariance + 27×27 eigh), far inside the wall-clock margin.

**Reasoning**: This is the unique remaining idea that passes the sharpened criterion: it ADDS learning at init (the stem starts as the decorrelating, variance-equalized feature extractor that training otherwise spends early heat discovering — arXiv 2210.03651), rather than deferring capacity (EXP-018) or buying unspendable stability. The evidence base has the right METRIC SEMANTICS for once: airbench/hlb are wall-clock speedruns on this exact dataset, where whitening init is load-bearing across two independent lineages. Following BN (bn1) does not erase the value — BN normalizes per-channel mean/var but not the cross-channel decorrelation and basis alignment whitening provides.

**Sources**: arXiv 2404.00498 § first-layer whitening (+ cifar10-airbench code); github.com/tysam-code/hlb-CIFAR10; arXiv 2210.03651; goal-learnings § Failed Approaches Low (EXP-018 exemption clause); project-insights § High (deferral).

**Estimated Effort**: medium (~25–30 lines: patch sampling, covariance, eigh, weight assembly; numerical care with ε and dtype).

**Risk Assessment**: (1) Wash-out risk: at 139 epochs (vs airbench's ~10) the stem has ample time to learn good filters anyway — the init advantage may converge to within-noise; this is the most likely failure mode and it is graceful. (2) Scale mismatch: 1/√λ filters produce unit-variance outputs while Kaiming-init bn1 expects different input scale — bn1 absorbs this by construction. (3) The 10 leftover Kaiming filters break the ± pairing for 5 directions — minor, the top-27 directions carry the variance. (4) Implementation bug risk (covariance over wrong axis, channels_last interaction) — mitigated by an analytic pre-launch check (print filter-output covariance ≈ I on a patch sample) before the timed run.

### 2. Width asymmetry 64/128/320 ([3,3,3], stages intact)
**Summary**: Widen stage 3 only: 256 → 320 (= 5×64, aligned); ~+1.9M params, total FLOPs ~+17% ⇒ ~125–130 epochs (must measure compiled dt — EXP-007 says inductor gains shrink with width).

**Reasoning**: Preserves early-stage depth (the EXP-017 failure isolate) while adding late-stage capacity. But it spends epochs — the binding resource — and the EXP-017 lesson (capacity location is priced steeply; more params at MORE epochs still lost) plus the EXP-012 ceiling (epochs trade ~1:1 against trajectory quality) bound the upside near zero.

**Sources**: reports/exp-report-017.md § Unexplored Avenues; goal-learnings § Failed Approaches High; project-insights § High (alignment, deferral).

**Estimated Effort**: low (2-line diff + dt gate).

**Risk Assessment**: Graceful but low-information — a fourth point on a thrice-measured curve; likeliest outcome is a −0.1 to −0.4pp converged deficit.

### 3. Cosine LR floor (anneal to lr_min ≈ 0.008 instead of ~0)
**Summary**: `lr_at` returns `lr_min + (PEAK_LR − lr_min) · 0.5(1+cos(πq))` — the tail keeps a small constant heat so late epochs continue refining instead of freezing.

**Reasoning**: Could raise the plateau LEVEL if baseline's near-zero tail under-trains the last ~10 epochs. But EXP-016 measured the opposite sign (hotter tail = still climbing = worse), the floor ADDS total heat (closed axis, hot side ×2), and the goal-learnings EXP-011 insight warns this family courts variance-increasing reward hacking — it must raise the plateau MEAN, which nothing in our data suggests (baseline's final evals are flat at peak, not decaying).

**Sources**: reports/exp-report-016.md (tail arithmetic); goal-learnings § Failed Approaches Medium (heat) + EMA entry (variance bar); exp-report-018.md § Next Steps caution.

**Estimated Effort**: low (1-line diff).

**Risk Assessment**: High probability of re-measuring the closed heat axis; worst case flagged as metric-gaming-adjacent. Weakest candidate.

## Idea Evaluation

**Evidence strength**: Idea 1 is the only candidate whose external evidence comes from the SAME metric regime (wall-clock CIFAR-10 speedruns, two independent code lineages) — the transfer-condition failure that killed EXP-016 (final-value theory), EXP-017 (fixed-epoch populations at other depths), and EXP-018 (fixed-epoch stability tricks) does not apply. Ideas 2 and 3 are extrapolations against measured in-project negative results.

**Mechanism clarity**: Idea 1 targets exactly the resource the last two experiments identified as decisive — learning per unit of early heat — by making the stem start where training otherwise spends early heat arriving. Idea 2's mechanism must beat a measured 1:1 exchange rate. Idea 3's mechanism contradicts the EXP-016 measurement.

**Expected impact**: Idea 1's honest range is wide (wash-out at 139 epochs vs airbench's 10 is plausible), but its upside is the only one not bounded by an in-project negative measurement. Ideas 2–3 have upper bounds near or below the +0.1pp bar.

**Risk profile**: all three fail gracefully; Idea 1 has implementation-bug risk handled by a pre-launch covariance check; Idea 3 additionally risks an integrity-flagged (reward-hacking-adjacent) framing.

**Feasibility**: Ideas 2–3 are trivial; Idea 1 is ~30 lines with numerical care — acceptable.

Idea 1 wins on the criteria that have actually predicted outcomes in this campaign (metric-regime-matched evidence + heat-efficiency mechanism).

## Chosen Idea
**Selected**: Whitening init for conv1 (data-derived eigenvector filters, learnable)

**Why this idea**:
It is the only remaining direction that satisfies the campaign's hardest-won lesson — interventions must increase learning per unit of early schedule heat (project-insights High; EXP-017/018) — and the only one whose external evidence comes from the same wall-clock metric regime (airbench/hlb speedruns on this dataset, where it is load-bearing in two independent lineages). It is init-only with an identical compiled graph, inheriting EXP-018's perfect-attribution property, and its failure mode is a clean graceful wash-out that would close the init axis entirely.

**Hypothesis**:
With conv1 initialized as 27 patch-whitening eigenvector filters + their negations (10 Kaiming-init filters retained) and everything else byte-identical to baseline @ 1990397, the early trail is FASTER than baseline-family runs (test_acc at eps 1–10 above EXP-017's 63.76@5 / 75.06@8 — the opposite sign of EXP-018's 35.26@5), throughput signatures are unchanged (~22.4ms, ~139 epochs, params 4,286,026), and the converged plateau lands at best_test_acc ≥ 96.81%. If the early gain washes out to a plateau at-or-below baseline, the init axis is closed in both directions (expressivity-removing AND information-adding) and structural ideas are exhausted at this recipe's scale.
