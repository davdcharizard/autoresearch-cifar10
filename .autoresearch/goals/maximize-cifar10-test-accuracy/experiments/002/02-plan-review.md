I've read the plan, goal definition, brainstorm, and the actual `train.py`/`prepare.py` (`TIME_BUDGET_S=300`, baseline `93.38` confirmed). Here is the prioritized adversarial concern list.

---

## Prioritized Plan Concerns — EXP-002 (Early Mixup + Hard-Label Tail)

### 1. The pass threshold (93.48%) sits inside single-run noise, and `best_test_acc` is a max-over-evals selection — a "pass" may not be a genuine improvement
*Targets: Verification Procedure steps 1 & 7; Abort Criteria; goal's 0.1pp rule.*
- The required margin is exactly `baseline + 0.10 = 93.48%`, reported at 2 decimals (`best_acc:.2f`). On CIFAR-10's 10k test set, 0.10% = **10 images**. Typical WRN seed-to-seed test-accuracy std is ~0.15–0.25% — i.e. **larger than the entire required margin**.
- Seed rerolling is banned, so there is exactly **one draw**. Worse, `best_test_acc` is the **maximum over ~26 evaluations** (line 244). The max of ~26 noisy evals is upward-biased by roughly one std above the mean, so a run whose true generalization equals baseline can still spike ≥93.48% at one eval point. That is precisely the "passes the measurement without genuinely advancing the objective" failure mode.
- Mixup also perturbs the CUDA RNG stream (new `randperm` + Beta draws), so the entire trajectory — including the hard-label tail — differs from EXP-001 by more than "just mixup." A ±0.10 delta is not cleanly attributable to the intervention. The plan nowhere acknowledges this; it treats one number crossing 93.48% as dispositive.

### 2. The brainstorm's pre-registered noise-band contingency was dropped from the plan
*Targets: brainstorm "Review" (line 83) vs. plan Abort Criteria / Verification.*
- The brainstorm explicitly commits to "pre-register a 50% cutoff follow-up if a near-flat result occurs with normal throughput" and to "log realized steps/epochs to rule out throughput loss." The plan carries the step/epoch logging (Informational Metrics) but **not the near-flat contingency**. Given concern #1, the most likely outcome is a result in the 93.40–93.55 band, and the plan has no decision rule for it — it will be force-classified pass/fail on a coin-flip-precision number.

### 3. The 130-pass throughput floor permits a ~12% epoch reduction that confounds any null/negative result
*Targets: Milestone 2 / Abort Criteria (line 47).*
- EXP-001 ran ~146 dataset passes; the gate accepts down to 130 (~88%). Mixup adds a Beta sample, a permutation, a tensor interpolation, and a **second cross-entropy term** per step during 65% of training. If the run lands near baseline, you cannot distinguish "mixup is neutral" from "mixup helped but the ~12% throughput loss erased it." The gate protects a *positive* verdict but leaves a *null/negative* uninterpretable — and given concern #1 a null is a likely outcome.

### 4. The plan does not preserve the runtime non-finite-loss guard on the new mixup loss path
*Targets: Milestone 1 / Code Changes (line 30).*
- Current code raises on non-finite loss at line 209–210 for the single-CE path. The plan replaces the mixup-phase loss with `lambda*CE(out,y) + (1-lambda)*CE(out,y_perm)` but only checks finiteness in the **smoke test** (Milestone 2), not in the live loop. The tail keeps the guarded original path; the mixup phase (65% of the run) would be **unguarded** unless the plan explicitly wraps the new loss in the same `torch.isfinite` check. State this in Code Changes.

### 5. The throughput gate projects from mixup-only step timing, not realized epochs
*Targets: Milestone 2 (line 14), Abort Criteria (line 47).*
- Milestone 2 times **100 mixup steps** and projects ≥130 passes over 300s. But only 65% of the run is mixup (slow); the 35% hard-label tail runs at EXP-001 speed (fast). Projecting the whole budget from the slow phase **underestimates** realized passes — a conservative gate that could abort a genuinely viable run. The realized `num_epochs`/`num_steps` are already emitted to `run.log`; the throughput judgment should be made from those post-hoc, not from a pre-run mixup-only extrapolation.

### 6. Two non-discriminating checks are dressed as verification
*Targets: Verification step 6 (line 62).*
- `training_seconds ≈ 300` and `total_seconds ≤ 600` are satisfied **by construction** (the loop runs to `TIME_BUDGET_S=300`; `timeout 600s` enforces the outer bound). They validate nothing about the intervention and shouldn't be counted as evidence the experiment worked — only as liveness guards. Minor, but don't let them pad the pass rationale.

### 7. Minor: transition-window thresholds are inconsistent
*Targets: Verification step 5 (63–68%) vs. Abort Criteria (line 49, "missing by 70%").*
- Verification requires the `Mixup disabled` message within 63–68% counted time; the abort rule tolerates up to 70%. A transition firing at 68–70% would satisfy the abort rule yet fail verification step 5. Reconcile to one window. (Step granularity is ~0.003%/step, so the transition will land at ~65.0% in practice — this is a spec inconsistency, not a likely failure.)
