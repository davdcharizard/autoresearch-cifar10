Based on my review of the plan, proposal, brainstorm, goal definition, and `train.py`, here is the prioritized concern list.

---

## Prioritized Plan Concerns — EXP-028 (Scale-Matched PNM)

### 1. The median update-ratio veto (≤1.30) is in direct tension with PNM's own mechanism — likely forcing "veto-or-null"
**Where:** Milestone 2 / Abort Criteria (plan lines 17, 66, 72); idea-01 "Algebraic and immutable safety gates."

The stated mechanism is "amplify useful stochastic-gradient noise." Working the recurrence through: coherent signal is matched to SGD (coefficient `q_sgd`), but the *fresh* per-step gradient component enters with steady-state coefficient `2·a·s_t/√5 → 2·0.19·22.36/2.236 ≈ 3.8×` relative to SGD's `1×`. That ~3.8× amplification of the newest (noisiest) gradient direction **is** the intervention. The preflight then vetoes if the median changing-gradient update/control ratio exceeds 1.30 (plus 5×/10× spike gates). So the two outcomes are: (a) PNM genuinely amplifies noise → likely trips the median/spike veto and aborts, or (b) it passes the gate because amplification was negligible → the mechanism is neutered and no gain is expected. The plan never defines a quantitative window in which amplification is simultaneously large enough to help and small enough to pass. The 1.30 / 5× / 10× thresholds are asserted (imported from the review), not derived from any measured SGD-vs-PNM distribution or tied to the class-concentration harm they proxy for. This is the most fundamental risk to the experiment producing any interpretable positive result.

### 2. Self-imposed validity gates (`num_steps ≥ 26091`, `training_seconds ∈ [299.9,300.2]`) exceed the goal's necessary conditions and can discard a genuine win
**Where:** Verification step 7 (plan line 124), Milestone 4 (line 27) vs. goal necessary conditions (definition lines 25–28).

Step 7 lists `num_steps >= 26091` and the tight `299.9 <= training_seconds <= 300.2` as **protocol-validity** requirements. The goal only requires `best_test_acc` > baseline+0.1, no crash, and total < 10 min — it imposes no step-count floor. Because training is wall-clock-budgeted, a slightly slower optimizer produces fewer steps at the same wall time; if PNM lands at, say, 26,000 steps but reaches ≥94.25% within budget, this plan declares the result **invalid** and throws it away — spending the single permitted production run on a result it then refuses to score. Step count belongs in Informational Metrics, not the validity gate. Same for the training_seconds window: it is inherited baseline behavior, but promoting it to a hard validity condition risks invalidating an otherwise-conforming run over a sub-second timing artifact.

### 3. Contradictory production finite-gradient contract between proposal and plan
**Where:** idea-01 implementation step 1 ("require every gradient dense, finite, FP32, and present") vs. plan Code Changes line 38 ("Do not add synchronous per-parameter finite checks to production").

A finite check (`isfinite().all()`) forces a device sync every step, which — under the time budget — directly reduces exposure and would need to be present in the timing controller to be representative. The proposal mandates it; the plan forbids it in production (and its reject-list on line 38 silently drops "finite"). These must be reconciled explicitly: state that production does **no** per-step finite scan (finiteness is preflight-only), and confirm the timing controller therefore also omits it — otherwise the exposure gate measures a different optimizer than production runs.

### 4. Exposure projection mixes a historical constant with a fresh timing ratio
**Where:** Milestone 3 / Verification step 5 (plan line 111): `floor(26898*control_mean/candidate_mean) >= 26091`.

`26898` is a step count from a prior run, but `control_mean`/`candidate_mean` are measured now against a freshly-spawned SGD control. Thermal/scheduling drift between the historical baseline run and today's timing session enters the projection linearly. The plan does mitigate this ("actual production steps remain conclusive"), so this is a lower-severity methodological wobble — but the ≤1.03 median-ratio gate and this projection are redundant checks of the same quantity and could disagree; note which one governs the abort.

### 5. The production scale-diagnostic verification is self-referential
**Where:** Verification step 7 (plan line 124): "finite scale diagnostics agreeing with the formula"; diagnostics section (line 40).

`pnm_first_scale` / `pnm_final_scale` are printed by the same code under test, so their "agreement with the formula" confirms only that the print path runs — it cannot catch a wrong recurrence or parity-indexing bug in production. This is acceptable *only* because the preflight (Milestone 2) is the real algebra check; the plan should not treat the production scale print as independent verification of correctness.
