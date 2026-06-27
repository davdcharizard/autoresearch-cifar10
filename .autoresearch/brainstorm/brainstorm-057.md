# Brainstorm EXP-057
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- No new external search this loop. Per exp-report-056, the standard-modernization audit is COMPLETE (every published-toolkit entry measured; external transfer 0-for-16), so candidate generation moves to constructions derived from the project's own laws rather than literature imports. Re-read of the in-scope files (train.py, prepare.py, README.md, TASK.md) for new angles found nothing unexploited: std=(1,1,1) normalization is eval-locked in prepare.py, the README's per-channel-mean deviation is BN-absorbed (per-pixel/per-channel norm variants rejected in brainstorm-056), and TASK.md adds no new degrees of freedom.
- Model knowledge anchors used below (no fetch needed): label-smoothing logit-equilibrium arithmetic (Szegedy et al. 2016 / Müller et al. 2019 — LS ε=0.1 on 10 classes fixes the optimal true-vs-wrong logit gap at ln(0.91/0.01) ≈ 4.51); "Don't decay the learning rate, increase the batch size" (Smith et al. 2018, arXiv:1711.00489); decoupled weight decay SGDW (Loshchilov & Hutter 2019, arXiv:1711.05101).

## Experimental History Review

- Current best: 96.71 @ 1990397 (EXP-006 recipe, certified by EXP-049 constant audit). Bar = 96.81. σ context (EXP-027): recipe mean ≈ 96.57, σ ≈ 0.16; one-draw detection needs TRUE effect ≥ +0.3. **51 experiments, 50 consecutive non-improvements.**
- State of the frontier after EXP-056: every catalogued axis is measured-closed — throughput (99.3% kernel floor, EXP-048), heat both directions (EXP-016/049), noise level both directions (EXP-011/022/023/024), loss geometry both directions (EXP-050/051), numerics both directions (EXP-054), averaging/ensembling (EXP-031/032/036), regularization dose (saturated), structural classes content AND order (EXP-026/…/056), kernel lattice {64,128,256} (EXP-045), tail pressure two-sided — data side (EXP-025/033) and parameter side (EXP-055).
- Standing laws that filter candidates: absorption (16 external imports nulled under TA+RE — mechanism must be something heavy augmentation cannot supply); deferral (+1ms ≈ −7 epochs ≈ −0.08pp); two-sided tail pressure (neither data nor any parameter subset may stop moving before budget end); max-statistic (only plateau LEVEL/length matter).
- Technically-open corners on the books (exp-report-056 Next Steps, brainstorm-056 runner-ups): (a) per-layer constants on the ONE layer the WD-with-BN equilibrium argument does not cover — fc, the only BN-free scale-sensitive layer (loss-geometry negative prior); (b) late batch-size schedule — changes the noise SCHEDULE, not its level (three adjacent negative closures).
- Protocol assets: GPU probe (~90s) is mandatory for any graph change; composite launcher with dual gates + watchdog at /tmp/exp046_composite.sh; replicate-pair MEAN protocol for any read ≥ 96.81.

## Candidate Ideas

### 1. Decouple the classifier from weight decay (fc.weight WD 5e-4 → 0)
**Summary**: Split `fc.weight` out of the decay param-group so the classifier trains with WD = 0 while every conv weight keeps 5e-4. One mechanism, one constant, ~5 lines in the optimizer setup; model graph, schedule, loader, and loop are byte-identical — **no torch.compile graph change, so no GPU probe is needed** (the optimizer is eager; param-group membership is invisible to inductor).

**Reasoning**: The selective-WD recipe decays all ndim>1 weights identically. For conv weights, BN makes WD scale-irrelevant (the equilibrium argument that pre-refuted LARS): WD only sets effective-LR equilibrium. The fc layer is the single exception — no BN follows it, so WD on fc does real work: it monotonically shrinks the logit scale. Label smoothing ε=0.1 fixes a finite optimal logit gap (≈ 4.51); if WD 5e-4 holds the classifier BELOW that equilibrium, the trained model pays a permanent margin deficit that augmentation cannot supply (absorption-law compatible: this is a label-space/classifier-scale mechanism, not a feature-view mechanism). Removing fc WD is NOT the forced-margin dose that failed in EXP-050 — it removes an extraneous constraint and lets CE+LS find its own equilibrium, which is self-limiting (the LS loss gradient reverses once targets are matched, so no runaway). Law check: dt identical (zero graph/loop change), heat identical (LR schedule untouched), noise identical, tail pressure maintained (fc keeps training to the end), numerics identical. Closure value is high: after this run, every parameter class in the model (BN/bias no-decay, conv weights 5e-4, fc) has had its WD treatment individually measured — the per-layer-constant corner closes regardless of sign.

**Sources**: brainstorm-056 Idea 3 (documented-weak entry); goal-learnings EXP-050/051 (loss-geometry closure — the negative prior), EXP-015 (WD account), brainstorm-055 Idea 2 (LARS/WD-equilibrium rejection); LS equilibrium arithmetic (Müller et al. 2019).

**Estimated Effort**: trivial — ~5-line param-group edit, CPU sanity (param-group membership assert + 3-step smoke), no probe, standard composite launch.

**Risk Assessment**: Safest failure mode on the books: byte-identical training signatures guaranteed by construction (dt, epochs, steps unchanged), so any metric move is attributable. Worst case is a sign-down read (< 96.41) meaning fc WD's margin cap was itself load-bearing regularization — that too closes the corner from below. Main assumption at risk: WD may not bind the fc scale at all (CE+LS equilibrium may already be reached), in which case the read is mean-band and the corner closes as redundant.

### 2. Late batch-size schedule: 512 → 1024 at p ≥ 0.75, LR unchanged (carried from brainstorm-056)
**Summary**: Keep batch 512 until progress 0.75, then feed 1024-sample steps (concatenate two loader batches; dual-warmup both graph shapes in the uncharged warmup), LR schedule untouched — a tail-only noise-scale reduction stacked on the cosine anneal.

**Reasoning**: The only un-bracketed noise DOF left: the noise law measured constant LEVELS (both directions lose) but never a late-only SCHEDULE. Published as Smith et al. 2018 (fixed-epoch setting). Under fixed time the step-cost saving is tiny (~+0.03pp equivalent), so the live variable is pure tail-noise annealing — which the cosine already performs via lr→0; this doubles a pressure that is already applied. Three adjacent closures read negative (EXP-022 constant-1024 at two LR rules, EXP-024 horizon trades, EXP-055 tail-conversion).

**Sources**: brainstorm-056 Idea 2 (verbatim frontier entry); Smith et al. 2018 (arXiv:1711.00489); goal-learnings EXP-012/022/023/024, EXP-055 report.

**Estimated Effort**: medium — loader stitching, dual-shape compile warmup, GPU probe mandatory (graph shape change), watchdog band revision for the 1024-shape tail.

**Risk Assessment**: Graceful (mean-band or sign) but carries real infra surface (two compiled shapes, recompile risk at the switch — EXP-055's silent-no-op lesson) for a mechanism with three adjacent negative priors. Dominated by Idea 1 on effort, risk, AND closure value per run.

### 3. Decoupled (lr-independent) weight decay over the anneal (SGDW-style)
**Summary**: Replace coupled WD (decay term = lr·λ·w, which vanishes as lr→0) with a constant decoupled decay applied directly in the loop, so norm pressure persists through the tail and the effective-LR (lr/||w||²) anneal is reshaped to be shallower late.

**Reasoning**: The one WD DOF besides per-layer placement: its TIME profile. Coupled WD self-anneals with the cosine; decoupling holds it constant — a tail-dynamics change that (unlike EXP-055's freeze) maintains full tail pressure. However, by the BN scale-invariance account, reshaping the effective-LR trajectory is a schedule-family/heat intervention in disguise, and the heat axis plus schedule family are measured-closed both directions. SGDW is also arguably a published-toolkit entry, weakening the "outside the toolkit" rationale.

**Sources**: Loshchilov & Hutter 2019 (arXiv:1711.05101); goal-learnings heat/schedule closures (EXP-016/049); van-Laarhoven-style effective-LR account used in the LARS rejection (brainstorm-055).

**Estimated Effort**: low-medium — loop edit (manual `p.data.mul_(1-λ_d)` per step on decay params inside the charged region, ~0 dt but must be verified), CPU sanity, probe advisable since the step content changes.

**Risk Assessment**: Mechanism reduces to an axis that is closed both directions — the clearest expected-null of the three; charged-region edit adds a small dt risk for nothing the laws leave open.

## Idea Evaluation

Evidence strength: all three are expected-null under standing closures — that is the honest state at 50 nulls; selection therefore weighs mechanism distinctness, closure value per run, and risk. Idea 1's mechanism is the most distinct: fc is provably the ONE parameter class where the WD-equilibrium argument (which closed LARS and makes conv-WD a pure effective-LR knob) does not apply, so its WD constant is the last individually-unmeasured constant in the recipe — and the LS-equilibrium arithmetic gives a concrete, falsifiable way it could bind. Idea 2's live mechanism (tail noise annealing) duplicates what the cosine already does and carries the largest infra surface (dual compiled shapes). Idea 3's mechanism collapses into the closed heat/schedule axis by the scale-invariance account. Expected impact is small for all; risk ordering is 1 ≪ 3 < 2 (Idea 1 changes nothing the compiler, loader, or timer can see). Closure value: Idea 1 completes per-parameter-class WD coverage in one trivial run; Idea 2 closes the noise-schedule corner but at medium effort; Idea 3 closes a corner that is already closed by argument. Feasibility strongly favors Idea 1: no probe, byte-identical signatures by construction, terminal branches on every outcome.

## Chosen Idea
**Selected**: Idea 1 — Decouple the classifier from weight decay (fc.weight WD 5e-4 → 0)

**Why this idea**:
It is the highest closure-value-per-cost run available on the measured frontier: the only individually-undosed constant left in the recipe (the per-layer WD treatment of the single BN-free, scale-sensitive layer), with a concrete falsifiable mechanism (WD-induced logit-scale shrinkage vs the CE+LS equilibrium gap ≈ 4.51), passing every standing law by construction — zero dt/heat/noise/numerics change, full tail pressure, classifier-scale mechanism outside augmentation's reach — and a guaranteed-attributable read because training signatures are byte-identical to the family. The loss-geometry negative prior (EXP-050/051) is acknowledged and priced into the branches: any outcome (up, mean-band, or down) terminally closes the corner.

**Hypothesis**:
If WD 5e-4 binds the classifier below the CE+LS logit equilibrium, removing decay from fc.weight lifts the converged plateau level and best_test_acc reads ≥ 96.81 (TRUE effect ≥ +0.3), to be confirmed by replicate-pair MEAN. Expected under the loss-geometry prior: a family-shaped read in [96.41, 96.73] at byte-identical signatures (dt 22.0–22.8ms, 138–140 epochs, params 4,286,026), closing per-layer WD coverage as redundant; a read < 96.41 instead identifies fc WD's margin cap as load-bearing regularization and closes the corner from below. All branches terminal.
