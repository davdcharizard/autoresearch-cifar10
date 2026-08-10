I've read the plan, the goal definition, the brainstorm, the proposal (`idea-01.md`), the results index, and the actual `train.py`. Below is a prioritized, grounded adversarial concern list on execution soundness.

---

## Prioritized concerns (hardest / most fundamental first)

### 1. The ≤1.01 timing gate is likely unachievable with the *naive per-parameter Python loop* the plan specifies — the experiment may fail its own no-go gate on kernel-launch overhead alone
**Section:** Code Changes line 36 ("perform `slow.add_(fast - slow, alpha=...)` then `fast.copy_(slow)` for each ordered pair"); Milestone 2 timing gate (line 20) / Abort Criteria (line 60): "candidate/control … at most 1.01."

The model has ~59 parameter tensors (conv1 + bn1 + 9 blocks × 6 + fc). The specified per-pair loop launches roughly 3 kernels/tensor (`fast-slow` subtract, `add_`, `copy_`) ≈ ~175 tiny kernel launches per sync, every 5 steps ≈ ~35 launches/step. With a `torch.cuda.synchronize()` on every step (line 248), these are launch-bound: ~35 × ~5µs ≈ ~175µs on an ~11ms step ≈ **~1.5%**, which *exceeds the 1.01 ratio*. The intervention could therefore veto itself at the timing stage even though its GPU math is trivial — and worse, the executor may be tempted to "rescue" the timing gate by quietly changing the loop, which the plan does not authorize or specify. **Actionable fix:** mandate fused `torch._foreach_sub_/_foreach_add_/_foreach_copy_` (2–3 kernels total per sync) in the Code Changes, and state that as the intended implementation so the timing gate measures the real design, not a strawman.

### 2. The projected-steps exposure gate anchors on EXP-010's 26,898 measured under past conditions and on an unspecified step-mix
**Section:** Milestone 2 (line 20) / Verification step 5: `floor(26898 * control_mean / candidate_mean) >= 26629`.

Two problems. (a) The formula assumes the *fresh* control run reproduces EXP-010's 26,898 steps; if current machine state makes control faster/slower than when 7c1e7d8 ran, the projection is biased and can green-light a candidate that the real run then undershoots on, or veto one that would pass. (b) The timing harness measures "production-region steps" but production is 80% slow strong-phase (RandAugment+CutMix collate) and ~20% faster weak-tail; if the 1,000 measured steps are drawn from one regime, the *absolute* projection is skewed even if the candidate/control ratio is fine. This is backstopped by the production hard gate `num_steps >= 26629` (Verification step 7), which is good — but the plan should say explicitly that the timing projection is advisory and require the timing steps to span both augmentation regimes (or measure the ratio only, not project absolute counts from a stale constant).

### 3. The "`training_seconds` ≈ 300" verification check is vacuous and gives false comfort
**Section:** Milestone 4 (line 30) / Verification step 7: "`training_seconds` is approximately 300."

The loop is time-budgeted (`while total_training_time < TIME_BUDGET_S`, line 213) — it *always* stops at the budget regardless of Lookahead overhead. So `training_seconds ≈ budget` is true by construction and can never detect that Lookahead ate steps. The only real exposure signal is `num_steps`. Also, the plan asserts the budget is ~300s but never confirms `TIME_BUDGET_S` from `prepare.py`; if it differs, the "≈300" and "total_seconds < 600" assertions are mis-calibrated. Drop the vacuous check or replace it with the meaningful `num_steps` comparison, and read the actual budget from `prepare.py`.

### 4. The preflight validates the mechanism on *materialized representative* batches, not the seed-42 trajectory the real run executes
**Section:** Milestone 2 (lines 15–17); Verification step 4.

The plan (correctly, learning from EXP-019) materializes 200 batches once and feeds both arms identical tensors instead of replaying forkserver seeds. But that means the concentration (<95%) / loss-EMA (<1.5×) gates certify Lookahead is *generally* safe on production-distribution data — they cannot certify the *actual* seed-42 run won't hit a concentration transient, because forkserver replay of the real trajectory is impossible. This is adequately backstopped by production monitoring (Milestone 3, line 25), but the plan's language ("prove … safety") overstates what the preflight establishes. State plainly that preflight is a representative-sample screen and production monitoring is the real safety net for the scored trajectory.

### 5. BatchNorm running-stats vs. parameter mismatch is an uncontrolled eval-noise source that can eat the ~0.1pp margin
**Section:** Code Changes line 36 ("Do not touch … BatchNorm buffers"); Configuration Changes.

Lookahead's `fast.copy_(slow)` resets *parameters* (conv/BN-affine/fc) to the slow point but leaves BN `running_mean/var` on the fast trajectory (correctly not interpolated, and the plan correctly avoids in-budget BN recal to protect the budget). Consequence: any evaluation whose epoch boundary lands immediately after a sync step (step % 5 == 0) evaluates slow-point weights against fast-trajectory BN stats — a param/stat mismatch that depresses that checkpoint. `best_test_acc` is the max over ≤19 evals, so this mostly adds downside noise. Given the acceptance margin is ~10 test images at one fixed seed with no reroll allowed (goal line 18), this is a concrete risk to a marginal pass. At minimum, add a diagnostic recording whether each eval epoch coincided with a post-sync step, so a near-miss can be attributed rather than mis-read as "Lookahead didn't help."

### 6. Sync placement/off-by-one must be pinned so counted cost and sync-count are exactly what the gate assumes
**Section:** Code Changes lines 36–37; Verification step 7 ("sync count consistent with `floor(num_steps/5)`").

`step` is incremented *after* `optimizer.step()` (line 251), so testing `(step+1) % K == 0` at the point right after `optimizer.step()` fires syncs at steps 5,10,15…, giving `floor(num_steps/5)` — consistent. But two placement details are load-bearing and only implied: (a) the sync block must sit *before* `torch.cuda.synchronize()` (line 248) for its cost to be counted (plan says so — verify it's not accidentally placed after `dt` is computed); and (b) the mid-epoch `break`s at the 80% switch and budget/max (lines 271–277) can leave the final segment ending on a non-multiple of 5 — harmless, but confirm the sync-count assertion tolerates that rather than expecting an exact multiple. Make the exact insertion point explicit in the plan to avoid a silent counted-cost or count-mismatch error.

---

**Scope note:** The plan stays within Hard Constraints — only `train.py` (tracked) plus experiment-local ignored preflight/timing scripts; no edits to `prepare.py`, the evaluator, or packaging. No out-of-scope code changes identified. The self-administered safety gates (95% concentration, 1.5× loss-EMA, 200-batch horizon) match the precedent that caught EXP-016/019/020/021 and are not a new reward-hacking surface.
