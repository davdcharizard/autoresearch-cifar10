# Adversarial Review — EXP-028

## Prioritized Feedback (most important first)

### 1. PNM — the scale-match proof covers the one regime PNM is *not* used in (core mechanistic gap)
The derivation forces `q_pnm(t)·s_t = q_sgd(t)` **only for a constant gradient**. But PNM's entire purpose is to amplify *stochastic* (changing) gradient noise via the `-previous` term — precisely the regime the proof does not bound. The proposal concedes this ("Scale matching fixes coherent signal magnitude, not instantaneous update norm"). So on real batches the update norm can exceed matched-SGD by the noise-amplification factor, which is the exact mechanism that produced EXP-020/022 candidate-only concentration. The safety gate (5× control / 10× median caps, >95% class-share veto) is well-targeted at this — but note the *expected outcome is a preflight veto → invalid*, not a scored miss. **Fix/mitigation:** report the empirical changing-gradient update-norm ratio distribution over the 200-batch corpus as a first-class result, and pre-commit that a median changing-gradient ratio >~1.3 is an abort (not just the 5× spike cap), so you learn *why* it vetoes rather than just that it did.

### 2. PNM — horizon mismatch attacks the goal's protected asset (short strong phase)
The paper's 5.01→4.48 gain is over 200 epochs. Noise amplification helps escape sharp minima *over long training*; in a 300 s / ~68-pass budget it more likely adds variance during the fragile strong phase that `02-system-understanding.md` names as the accuracy limiter and that EXP-010's 89.73% switch fit must be preserved. The proposal's own switch floor of 88.73% budgets a ~1-point fit *loss* — a tell that the mechanism fights the thing the goal cares about. **Mitigation:** this is inherent to the bet; the honest move is to state that a switch in [88.73, 89.73) with a top-1 miss is the *expected* failure signature, not a surprise.

### 3. PNM — this is a third optimizer-path attempt in a high-mortality family, but is *materially* distinct (not a disqualifying retry)
EXP-020 (Nesterov) and EXP-022 (Lookahead) both died at concentration gates, and the learnings promote "unwarmed optimizer-path changes create lower-loss class transients" to Medium. Critically, this is **not** a blind retry: (a) first-step update = exactly `d_1` (no Nesterov 1.9× amplification), (b) no parameter pullback (the Lookahead failure cause), (c) EXP-020's report *explicitly* recommended a scale-matched PNM revisit, and (d) the failed-approach insight literally prescribes "gate new optimizer paths on immutable production batches" — which this does. So it clears the "already-failed, retried unchanged" bar. Keep this distinction explicit in the plan so it isn't mistaken for a rerun.

### 4. GC — the implemented mechanism is weaker than the cited theorem (evidence-mechanism gap)
The proposal is commendably honest that placing GC *before* unchanged coupled decay yields `P(g)+λw`, so the paper's invariant-filter-mean theorem does **not** hold (decay still shrinks the mean). That means the ECCV evidence — already from CIFAR-100, larger nets, 200 epochs, wd 5e-4 — supports a *different* operator than the one being run. The cited gain does not transfer cleanly. **Mitigation:** frame the hypothesis strictly as "remove common-mode data-gradient drift," and don't lean on the paper's accuracy numbers as a point estimate; they're directional at best.

### 5. GC — modest ceiling vs. a hard +0.10 gate; real no-improvement risk
Published GC gains are typically ~0.1–0.5% on longer schedules. On a 300 s horizon against a 94.25% gate, the most likely scored outcome is the 94.15–94.24 dead zone. It's the safest candidate but also the one most likely to produce a clean, attributable, *sub-gate* result. That's not a flaw in reasoning — just a low expected delta. **Mitigation:** none available without changing the lever; accept it as a low-variance probe.

### 6. GC — 38 tiny kernels on a 10.9 ms step vs. a 1.01 timing gate is tight
EXP-003 shows a "cheap" change lost 6.7% exposure on this fast net; kernel-launch overhead matters disproportionately here. The self-imposed ≤1.01 ratio (26,629-step floor) is strict and could invalidate on timing alone before any accuracy signal. This is the right gate to have, but flag it as a live invalidation path. **Mitigation:** consider a single fused `_foreach_` mean/sub across the 19 conv grads to cut launch count — but that's explicitly forbidden by the proposal's own no-fused-kernel rule, so the tight gate stands as a genuine risk.

### 7. LeakyReLU — the mechanism is self-undermined and the evidence is the weakest (near-fatal on reasoning)
The stated premise is "reduce dead/dying ReLU features," yet the proposal itself states BN "weakens the premise that dying ReLUs are the current limiter" and that slope-0.01 "preserves signed evidence only weakly." So it targets a limiter it cannot establish exists, with a change small enough that its own author doubts the effect. The cited He et al. evidence is for *learned* PReLU on ImageNet and explicitly does not transfer to fixed-slope CIFAR ResNet-20. **Mitigation:** to make this credible you'd need a diagnostic first showing measurable dead-channel prevalence in the accepted run — absent that, the hypothesis has no legs.

### 8. LeakyReLU — touches every block from step 1, in the family that repeatedly suppressed the strong phase
Preactivation (−2.85) and zero-gamma (−3.25) both suppressed switch fit; LeakyReLU is milder but same family, and it additionally makes the final pooled feature *signed*, changing classifier geometry and potentially cancelling localized CutMix evidence under GAP (the proposal admits this). Unclear-signed upside + known-risky family = weak bet. **Mitigation:** none that preserves the idea's identity.

---

## Scored Verdict

**Scale-Matched PNM (idea-01)**
- Evidence & reasoning — **7/10**: Only candidate with a directly comparable *published accuracy* improvement (5.01→4.48 on CIFAR ResNet-18), plus a rigorous scale derivation and a genuine, documented distinction from EXP-020/022 — but horizon transfer is weak and the proof covers only constant gradients.
- Potential impact — **8/10**: Highest ceiling; a real optimizer-generalization mechanism with external precedent, if it survives the concentration gate.

**Conv-only Gradient Centralization (idea-02)**
- Evidence & reasoning — **7/10**: Sound, unusually honest reasoning; forward-graph-preserving and RNG-neutral, so it protects the fragile strong phase by construction — but the implemented `P(g)+λw` operator is weaker than the theorem it cites.
- Potential impact — **5/10**: Mild regularizer; most-likely scored outcome sits in the 94.15–94.24 no-improvement band, and a tight 1.01 timing gate can invalidate it outright.

**Fixed LeakyReLU 0.01 (idea-03)**
- Evidence & reasoning — **3/10**: Self-undermining premise (BN negates the dying-ReLU rationale), weakest external support (learned-PReLU/ImageNet doesn't transfer), effect the author calls "weak."
- Potential impact — **3/10**: Small, unclear-signed change in a block-wide family that has repeatedly suppressed the strong phase; also perturbs classifier geometry.

### Pick: **Scale-Matched PNM (idea-01)**

It wins on the two scored axes the objective rewards: it carries the **strongest direct evidence** (the only proposal with a published accuracy gain on the same task family) and the **highest impact ceiling**, and its mechanistic case is careful — the scale derivation, the exact-first-step property, and the absence of parameter pullback make it *materially* different from the EXP-020/022 failures rather than a disqualifying retry (indeed, EXP-020's report requested exactly this). GC is the sounder-floor, lower-variance probe and the natural hedge, but its ceiling is capped by a modest mechanism and an operator that no longer matches its cited evidence — a well-argued exploratory bet with real upside should not lose to a safe incremental one on caution alone. LeakyReLU is a distant third and is effectively self-refuting.

Two conditions the PNM plan must honor for the pick to pay off, both already scaffolded in the proposal: (1) treat the changing-gradient update-norm distribution over the immutable EXP-022 corpus as a pre-registered abort criterion, not just the 5× spike cap — that is where its core risk (feedback #1) actually lives; and (2) report the switch checkpoint honestly against 89.73%, since a fit drop into [88.73, 89.73) with a top-1 miss is the *predicted* failure signature of noise amplification on this short horizon, not an anomaly.
