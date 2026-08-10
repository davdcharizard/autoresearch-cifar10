Based on my review of the EXP-038 plan against the goal definition and the chosen idea (idea-06, cosine classifier), here is the prioritized concern list.

---

## Concerns — EXP-038 plan (Output-RMS-Matched Cosine Classifier)

### 1. [Scope] The evaluation-cadence rewrite is a second, independent code change not part of the chosen idea — it confounds attribution and breaks baseline comparability
**Section:** Code Changes bullet 2; Configuration Changes "Evaluation opportunity control"; Milestone 1 ("fixed 19-look fairness control").

The chosen idea (`01-brainstorm.md`, idea-06) is *solely* the cosine classifier. The "19 elapsed-progress evaluation opportunities" control was designed for the *speed-changing* ideas (idea-01 channels-last, idea-02 batch-192), where epoch count shifts. It is imported here via the Adversarial Review Response's rationale ("normalized-head overhead cannot alter max-metric observation count"), but the cosine head is not a speed intervention. Bundling an evaluation-schedule rewrite with the classifier change means:
- Any measured Δ in `best_test_acc` cannot be cleanly attributed to the classifier vs. the cadence change.
- The candidate is measured under a **new cadence** while the moving baseline (94.15 at `7c1e7d8`) was measured under the **old "open-ended per-epoch dense-tail"** cadence. The plan never re-runs the baseline under the new cadence, so the +0.10 comparison is not apples-to-apples. This is exactly the "evaluation changes" concern the goal statement (`01-definition.md` §Goal Statement) singles out. Either drop the cadence change for this idea, or re-measure the baseline under identical cadence.

### 2. [Invalid/ambiguous gate] "Exactly 19 unique once-per-epoch evaluations" is likely arithmetically infeasible and its "progress" unit is undefined
**Section:** Configuration Changes "Evaluation opportunity control"; Verification step 8 ("exactly19 unique once-per-epoch evaluations").

The schedule is 4 early points + **15 points in [0.8,1.0]**, with **one call per distinct epoch**. Getting 15 *distinct* evaluations inside the last 20% of the run requires ≥15 distinct epochs in that window (≈75+ total epochs). A 300s CIFAR ResNet-20 run typically completes far fewer epochs; if the last 20% contains <15 epochs, the 15 points collapse under once-per-epoch and you get **<19 total**, failing the summary-integrity gate on an otherwise valid run — or you evaluate the same epoch twice, violating the hard constraint "Do not run validation more than once per epoch" (`01-definition.md`). Additionally, "elapsed-progress" is never pinned to a unit (wall-clock fraction? step fraction? epoch fraction?), so the "exactly 19" assertion is unverifiable as written. Pin the unit and confirm the actual epoch count supports 19 distinct once-per-epoch looks before committing to the gate.

### 3. [Over-strict verdict — false negative on the goal] The mechanism/fit gate can discard a run that satisfies every goal necessary condition
**Section:** Verification step 9 ("a preregistered mechanism/fit verification miss is no-improvement"; "mechanism support requires >=89.0"); Milestone 3.

The goal's necessary conditions (`01-definition.md` §Necessary Conditions) are only: exceed baseline by ≥0.1 pp, no crash, within time budget. Nothing requires the *mechanism* (switch-fit ≥89.0%) to hold. As written, a run with `best_test_acc ≥ 94.25` but switch-fit <89.0% is declared **no-improvement** — discarding a genuine metric win because an experimenter-added proxy failed. This over-constrains beyond the goal and risks throwing away a real improvement. Mechanism-fit should be recorded as *informational*, not a veto on a passing primary metric.

### 4. [Precondition unverified] The entire calibration/replay/timing edifice depends on immutable EXP022/028 corpora existing on disk with the stated SHAs
**Section:** Code Changes ("hashed EXP022 batches"); Verification steps 3–4 (SHA `e04dc2…`, `ffefe9…`, RMS `2.7600300312042236`, scale `22.786916732788086`).

The frozen `COSINE_SCALE` and every safety gate are derived from registered EXP022/EXP028 batch dumps. The plan asserts their SHAs but does not confirm the files are present (only untracked `data/` is visible in git status). If these corpora are absent or hash-mismatched, calibration, both replays, and timing all abort before production. This is a hard precondition that should be checked *first*, not mid-verification. It is also a single point of failure: the whole result rests on one hard-coded float whose provenance is a separately-coded RMS computation.

### 5. [Safety gate gives false confidence] The radial-collapse veto uses a linear log-row-norm projection over a self-accelerating nonlinear process
**Section:** Abort Criteria "Long replay effect-specific bounds" ("last-half log-row-norm linear projection to 26,898 steps"); Milestone 2/Verification step 5.

The core mechanism risk (correctly flagged in `01-brainstorm.md`): with `normalize(W)`, the loss gradient has no radial component, so `weight_decay=1e-4` shrinks row norms unopposed, and the effective tangent LR grows as norm shrinks — i.e., collapse *accelerates*. A **linear** extrapolation of log-row-norm from a 5,120-step replay to the ~26,898-step horizon will *underestimate* late-stage collapse. A candidate can pass the replay projection and still die in the real run. The downside is bounded (a wasted production run → no-improvement, not a false improvement), but the plan presents this projection as a "prospective safety veto" with more authority than a crude linear model warrants.

### 6. [Under-specified apparatus / tunable gates] Extensive verification controllers are treated as existing helpers, and their bounds are hand-tuned envelopes that risk being unfalsifiable
**Section:** Code Changes bullet 3 ("Ignored EXP038 controllers"); Abort Criteria (short/long/timing bounds); Verification steps 2–6.

The FP64 oracle, construction controller, 264-step and 5,120-step replays, and seven-pair timing harness are substantial code that must be *written* correctly, yet the plan asserts what they "prove" as if they exist. The abort bounds (`5x accepted-control envelope`, `2x first-256-step medians`, `row max/min <=3`, `>95% one-class share`) are numerous and hand-picked with no concrete definition of the "accepted-control envelope." Bounds that loose and numerous can be tuned post-hoc to pass and rarely fire — giving false assurance rather than a real gate. None of this validates the production metric; it only gates go/no-go. Recommend concretely specifying envelope definitions and pre-registering them before any replay is run.

### 7. [Correctness dependency] The unused-bias invariant relies on SGD silently skipping grad-None params
**Section:** Code Changes bullet 1 ("bias remains stored but receives no gradient/update"); Verification step 2 ("bias `grad is None`/bitwise unchanged").

Because the forward uses `F.linear(features, weights)` with no bias, `bias.grad` stays `None` and standard SGD skips it (no weight-decay applied to a None-grad param). This is correct for the default optimizer path, but it is a fragile invariant: any `foreach`/fused SGD path or a decoupled weight-decay implementation could touch the stored bias and drift it, silently changing the model vs. the accepted construction. The plan does add a bitwise-unchanged check (good) — just ensure that check runs against the *actual* optimizer path used in production, not only the construction controller.
