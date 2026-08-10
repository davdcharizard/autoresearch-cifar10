# Adversarial Review — EXP-038 Finalists

I read the brainstorm, the goal definition, the system understanding, the experiment learnings, and the three proposals (idea-01 channels-last, idea-02 batch-192, idea-06 cosine classifier). Prioritized feedback first, then the scored verdict.

---

## Prioritized Feedback

### 1. Idea-02 (batch-192) bets on the one lever this goal's history has repeatedly disproven — **most severe**
The proposal's mechanism is "more image exposure." But `02-system-understanding.md` explicitly flags this as untested (*"additional exposure has not yet been causally tested at the accepted recipe"*), and the accumulated history points the other way: EXP-007 won with **70.76% of prior steps** by adding *capacity*, not exposure (`03-experiment-learnings.md` Patterns); EXP-026 raised steps to 27,268 and got *worse*; EXP-011 more mixing, worse. Meanwhile idea-02 **discards ~25% of optimizer decisions** (26,898 → ~20,058 by its own math), which is precisely what the diagnosis warns against: *"EXP013 shows raw image throughput is insufficient if optimizer decisions fall too sharply."* It trades away the quantity the recipe depends on (frequent decisions on a short schedule) for the quantity history says is not the limiter. Path to improve: it can't easily be rescued within its own thesis — the honest move is to demote it. If pursued, it must argue why *this* exposure differs from EXP-026's, which it does not.

### 2. Idea-02's LR 0.15 sits adjacent to the goal's #1 recurring failure
`03-experiment-learnings.md` High-importance recurring failure: *"Global-LR optimizer-path interventions destabilize early class geometry even when first-step or coherent scale is controlled"* (EXP-020/022/028). Preserving batch/LR ratio (NeurIPS 2019) rules out the *under-scaled* variant but does **not** bound curvature or momentum overshoot at 0.15 — the proposal concedes this in Principal Risks. Its immutable-corpus safety gate is the right mitigation, but the prior is unfavorable. Additionally the 50%-longer momentum image horizon changes response speed exactly at the abrupt 80% switch that **must** preserve the 89.73% switch fit; the pre-registered 87.08% floor is a good guard, but the mechanism actively threatens the invariant.

### 3. Idea-01 (channels-last) attacks the systems bottleneck but not the accuracy limiter — the exposure→accuracy link is untested and its ceiling is capped
This is the mechanism/limiter gap. `02-system-understanding.md` says the accuracy bottleneck is *generalization under a short strong phase*; idea-01 does nothing to that — it only feeds ~807 extra steps into an exposure→accuracy path the same doc calls causally untested. Two compounding problems: (a) the schedule is keyed to **elapsed progress**, so the extra updates are finer-grained samples of the *same* annealed curve, and a meaningful share land in the low-LR tail where EXP-002 measured only a **0.01-point** best-vs-final gap — near-zero marginal value; (b) the proposal's own point estimate is **94.25%, sitting exactly on the +0.10 gate** (10 test images). Even a clean timing pass is close to a coin flip. Path to improve: add analysis of *where* in the schedule the extra 807 updates fall — if they are tail steps, the accuracy hypothesis is near-dead on arrival and the idea should be reframed as pure infra with no accuracy claim.

### 4. Idea-06's central admitted gap: output-RMS matching ≠ Jacobian matching, and the safety replay horizon is far too short for the actual failure mode
The proposal is candid that `dL/dh` carries a `1/||h||` factor and `dL/dw_k` a `1/||w_k||` factor, and that coupled `1e-4` decay slowly shrinks raw row norms which the inverse-norm Jacobian then **amplifies** — a *late-training* instability. But the immutable-corpus gate replays only **200 strong + 64 weak steps (264 total)** against a **~26,898-step** production run. The dangerous amplification is a slow-decay effect that may not surface in 264 steps. This is the single most important refinement: the gate can pass while the real risk lives at step ~15k. Path to improve: extend the replay horizon materially, or add a row-norm-floor / update-ratio survival statistic explicitly designed to catch monotonic drift, not just first-step spikes (which is what EXP-014/031/034 taught — *"optimizer geometry became extreme after the first step"* — but here the failure is slower, so a first-step-style screen is the wrong instrument).

### 5. Idea-06 risks the CutMix signal that the entire frontier rests on
Deleting feature magnitude (`F.normalize` on pooled features) forces clean and CutMix-mixed samples onto the same radius. EXP-010's whole +0.60 gain is regional CutMix (`03-experiment-learnings.md` Patterns), and the diagnosis names preserving the 89.73% switch fit as mandatory. If magnitude encodes mixture ambiguity, this specifically damages the strong phase. The proposal acknowledges this and plans separate strong/weak reporting — good, but I'd elevate the 89.73% switch-fit floor to a hard pre-registered gate as idea-02 did, not just a reported number.

### 6. Idea-01 is the best-engineered but its result is a net-implementation effect, not a clean counterfactual
Changed cuDNN reduction order perturbs the seed-42 trajectory, so a pass/fail on the 10-image gate conflates layout numerics with exposure. The proposal states this honestly (Risks). Non-fatal, but it means even a *pass* doesn't validate the registered exposure mechanism — which the decision rule correctly flags as requiring adversarial analysis. Good discipline; just underscores the low informational yield.

### Non-issues (credit where due)
- All three correctly replace the dense-tail trigger with a fixed 19-look schedule — a fairness control matching EXP-010, not a reward-hack. Good.
- Idea-06's scale is **not** a retry of any failed approach: EXP-014/031 were max-pooling readouts, EXP-034 a conv reparam, EXP-037 stem-centering with an arbitrary scale-8. A cosine *classifier* with RMS-matched temperature is genuinely untried and directly fixes EXP-037's diagnosed scale mismatch. Distinctness is real.
- All three reuse registered immutable corpora with hash verification and control-qualified denominator-safe gates — the EXP-035 lesson is internalized across the board.

---

## Scored Verdict

**Idea-01 — FP32 Channels-Last**
- Evidence/reasoning: **7/10** — provable systems mechanism (official PyTorch, direct hit on the 75.46% backward limiter) and the most rigorous gating, but the evidence is for *speed*, and the accuracy link runs entirely through an untested exposure causality.
- Impact: **3/10** — ceiling capped at ~3% exposure; point estimate sits *exactly* on the gate; extra steps largely land in the near-flat tail; likely neutral on tiny FP32 tensors.

**Idea-02 — Batch-192 Linear-Scale Knee**
- Evidence/reasoning: **4/10** — NeurIPS batch/LR grounding is real, but the core exposure hypothesis contradicts this goal's own repeated results (EXP-007/011/026) and the idea discards updates the diagnosis says matter most.
- Impact: **4/10** — modest, and it bets on the disfavored mechanism while adding LR-0.15 downside and momentum-horizon risk at the protected 80% switch.

**Idea-06 — Output-RMS-Matched Cosine Classifier**
- Evidence/reasoning: **6/10** — strongest novel reasoning (RMS calibration removes EXP-037's arbitrary-scale flaw without labels/LR retuning) and the only finalist attacking the diagnosed accuracy limiter; docked for the admitted Jacobian mismatch under fixed LR 0.1 and a safety replay far shorter than the production horizon.
- Impact: **7/10** — highest representation ceiling among the three; directly targets generalization/confidence under strong views; asymmetric downside but a sound, mechanistically-grounded upside.

### Pick: **Idea-06 — Output-RMS-Matched Cosine Classifier**

It is the only finalist whose success mechanism aligns with the *actual* bottleneck (`02-system-understanding.md`: the accuracy limiter is generalization under the short strong phase, not systems cost). Idea-01 attacks the systems limiter with a provable but low-ceiling speedup whose accuracy payoff runs through an explicitly-untested exposure link and whose point estimate is pinned to the gate — a safe near-null. Idea-02 is the weakest: it sacrifices ~25% of optimizer decisions (the thing the diagnosis says matters) to buy raw exposure (the thing this goal's history repeatedly shows does *not* help), while nudging LR toward the recurring collapse family. Idea-06 is the well-reasoned exploratory bet the guidance says should not be risk-discounted into irrelevance: it is genuinely distinct, it fixes a specific diagnosed flaw from EXP-037, and its real risk (late-run inverse-norm drift) is identifiable and can be de-risked by extending the safety-replay horizon per Feedback #4 before spending the single scored run. Highest merit-weighted expected value of the three.
