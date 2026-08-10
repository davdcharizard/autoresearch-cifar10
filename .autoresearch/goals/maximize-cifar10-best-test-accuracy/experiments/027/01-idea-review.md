# Adversarial Review — EXP-027 Finalists

I read all three proposals (`idea-01` CutMix-off bridge, `idea-02` Option-B projections, `idea-03` FP32 channels-last), the goal definition, system understanding, learnings, and results TSV. Prioritized feedback first, then the scored verdict.

---

## Prioritized Feedback

### 1. Idea-03 (channels-last): the accuracy mechanism contradicts the goal's own diagnosis — the more serious flaw than the well-flagged speed risk
The proposal is admirably honest about link #1 (FP32 32×32 kernels "may gain nothing"), but the *fatal* problem is link #2. Its sole accuracy mechanism is "extra exposure raises accuracy." The system understanding explicitly refutes this as the limiter: the accuracy bottleneck is "generalization under a short strong-view phase," not step count (`02-system-understanding.md` §Current Bottleneck). Direct counter-evidence: EXP-007 gained **+1.25 points with 29.2% *fewer* updates** (capacity beat exposure), and EXP-013 rejected batch-256 (more images) on throughput. The Headroom Assessment states outright: "additional exposure has not yet been causally tested at the accepted recipe." So even in the ~lucky world where the ≥3% timing gate passes, funding ~800 extra tail steps onto an already-smooth trajectory that "finishes at its best" (EXP-010) has almost no expected accuracy effect.
- **Path to improve:** this only becomes worthwhile if the freed time funds a *capacity* change (which the diagnosis says works) rather than raw steps — but that is a different, attribution-destroying experiment. As a standalone metric-mover it is dominated; its realistic best case is a cheap timing veto that produces no accuracy run at all.

### 2. Idea-02 (Option-B): the nearest local experiment is counter-evidence, not support, and the idea doesn't address *why* it failed
EXP-017 ran a near-identical learned `1×1`-Conv+BN transition projection. It **improved switch fit (89.73→90.20)** yet **worsened final NLL to 0.2024 and peaked 94.09 — below the gate.** The proposal's differentiator is "no pooling / preserves the `::2` lattice," and it candidly concedes that if Option B "repeats EXP-017's higher switch fit and worse late NLL, the learned projection/BN basis is the likely harmful component." That is the more parsimonious reading of EXP-017: the harm tracks the *learned projection+BN*, which Option B keeps, not the pool, which it drops. There is no evidence offered that pooling (rather than the projection) caused the NLL regression. This also sits inside a broad recurring failure family — EXP-012 full-preactivation, EXP-015 zero-gamma, EXP-017/021 transition shortcuts — where transition/representation changes trade switch fit for worse generalization in this 300s recipe.
- **Gap between mechanism and limiter:** the diagnosis says a representation candidate "must preserve EXP-010's healthy 89.73% switch fit." EXP-017 already *cleared* that and still lost on NLL. Option B's mechanism (better channel transport → higher switch fit) attacks a bar that isn't the binding one.
- **Path to improve:** its genuine value is *attribution* (isolating lattice-preservation from pooling), not the metric. If pursued, frame and defend it as a diagnostic; for maximizing `best_test_acc` its expected direction is null/negative.

### 3. Idea-01 (CutMix-off bridge): targets the right limiter, but the "adaptation debt" it repairs is not demonstrated at the accepted p=0.5 — and its cited recoveries come from a different intervention
This is the strongest of the three, but its central assumption is unproven and its own counter-evidence section says so: EXP-010 "already ends at its best," switch fit (89.73%) and first-weak (93.16%) are healthy, so there may be *no debt to repair* and "ending CutMix early could simply discard useful late regional examples." Critically, the supporting recoveries (EXP-011, EXP-026 snapping back to ~93.4% at the first hard checkpoint) occurred after removing **all** strong augmentation at 80% from an **over-regularized** state — not after removing only CutMix while N1/M7 continues at LR 0.1 from a *correctly*-regularized state. The mechanism is transported across two changed conditions.
- **Secondary risks:** the 70% boundary is a single unswept point that the protocol forbids re-tuning, so a miss is low-information (near-zero learning for one seed); and the intervention conflates removing pasted pixels with removing soft labels (acknowledged, unavoidable in one run).
- **Why it's still the best bet:** it attacks exactly the question the learnings name as live — "the regularizer-to-refinement boundary … is the live augmentation question" (`01-brainstorm.md`, echoed in learnings) — with the densest directly-relevant local evidence (EXP-005/010/011/026), adds **no GPU path** so exposure is protected, and has bounded, valid downside.
- **Path to improve:** motivate the boundary from EXP-010's own trajectory (where does regional supervision stop adding value?) rather than a round 70%, and pre-register the switch/first-weak diagnostics (already done) so even a miss discriminates "no debt" from "debt but hard N1/M7 can't repair it." The implementation is heavier than the idea (forkserver shared `Value`, prefetch-drain provenance) — that complexity is justified given the constraint against a 70% reseed, but it is the main execution risk.

### 4. Cross-cutting: single-seed, 0.10-point gate makes all three weak causal evidence
Every proposal correctly notes a bare 94.25% pass is ~10 examples and one seed. This doesn't change the ranking (it applies equally), but it means the tiebreaker is *expected direction and mechanism-limiter alignment*, which favors idea-01.

---

## Scored Verdict

**Idea-01 — CutMix-Off / RandAugment-On Refinement Window**
- Evidence & reasoning: **7/10** — attacks the explicitly-named live limiter with dense on-point local evidence (EXP-005/010/011/026); docked because the "debt at p=0.5" premise is undemonstrated and the recovery evidence is transported from a different (all-aug-off, over-regularized) regime.
- Potential impact: **6/10** — modest but real ceiling (point pred 94.32 vs 94.25 gate), lands directly on generalization/strong-fit which the diagnosis says bounds the metric; no exposure risk.

**Idea-02 — Original Option-B Strided Projections**
- Evidence & reasoning: **5/10** — rigorous initialization/attribution design and a fresh, clean point, but the nearest neighbor (EXP-017) is counter-evidence and the idea offers no reason the projection+BN (which it keeps) wasn't the culprit; sits in a repeatedly-failing transition-change family.
- Potential impact: **4/10** — high *scientific* value (pooling-vs-projection attribution) but low *metric* expected value; nearest result improved switch fit and still missed on NLL.

**Idea-03 — FP32 Channels-Last**
- Evidence & reasoning: **3/10** — two independent unproven links, and the accuracy mechanism (more steps) directly contradicts the goal's diagnosis (exposure not the limiter; EXP-007 gained with fewer steps, batch-256 rejected).
- Potential impact: **2/10** — most likely outcome is a cheap timing veto and no accuracy run; conditional on passing, extra tail steps are unlikely to move a generalization-bound metric.

### Pick: **Idea-01 (CutMix-Off / RandAugment-On Refinement Window)**

It is the only finalist whose mechanism aligns with the diagnosed limiter (strong-phase generalization / the regularizer-to-refinement boundary) *and* whose supporting evidence is dense and directly on-topic, while risking nothing on the systems side (no new GPU path, exposure preserved). Idea-03 is a fail-fast systems probe whose accuracy story the goal's own analysis contradicts. Idea-02 is the most scientifically interesting and best-engineered for attribution, but its closest prior experiment already ran essentially this mechanism and lost on NLL — its expected direction for the *metric* is null-to-negative, so it loses to idea-01 on merit despite the stronger craftsmanship. Idea-01's main weakness (undemonstrated debt at p=0.5, arbitrary 70% boundary) caps its ceiling but does not threaten validity, making it the highest expected-value shot at clearing the +0.10 gate.
