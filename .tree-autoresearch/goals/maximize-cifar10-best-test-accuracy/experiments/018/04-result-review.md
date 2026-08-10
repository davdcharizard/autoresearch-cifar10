All evidence gathered. Here is the audit.

# EXP-018 Independent Audit

## Scope & provenance

- **Scope — PASS.** `git diff --name-only a36dc09` shows only `train.py` changed (+278/−2 lines); `git status` shows only ` M train.py`. The only two lines removed from the parent are the config print (extended with lookahead keys) and the single `evaluator.evaluate` call (wrapped in the sync/swap branch). `prepare.py`, the `Eval` harness, seeds (`torch.manual_seed(42)`, `cutmix_seed=42`), CutMix, drop-path, LR schedule, and the `best_acc = max` selection rule are byte-identical to parent — no seed rerolling, no evaluation-protocol change.
- **Stale-output risk — NONE.** `run.log` (mtime 2026-08-06 14:59:45) postdates `train.py` (14:45:28) and matches the execute doc's run window (14:52–14:59 UTC). The log's config line contains `lookahead_k=5 lookahead_alpha=0.5 lookahead_audit_every_syncs=128`, which only EXP-018 code prints — this log cannot be a leftover from the parent or a prior experiment.

## Metric decision recomputation

- Parent EXP-002 metric re-confirmed from the tree store (`04-results.tsv`): **95.23**. Threshold = 95.23 + 0.10 = **95.33**.
- `run.log` final summary: `best_test_acc: 94.82%`. 94.82 < 95.33 → **no-improvement is correct**; delta vs parent = **−0.41 points**, matching `03-execute.md`. Context thresholds 95.53/95.61/95.71 correctly reported as unreached.

## Frozen budget & evaluation integrity

- `training_seconds: 300.0` ∈ preregistered [299.5, 301.0]; `total_seconds: 450.1` < 600; `startup_seconds: 1.0`. Lookahead sync + audit work sits before `torch.cuda.synchronize()` inside the timed region, so its cost is charged to the budget as planned.
- One evaluation per epoch: 146 epochs, 146 eval lines, `num_params: 2,748,890` unchanged. Evaluation always uses the frozen `Eval()` exactly once per boundary; the swapped path restores fast weights in `finally` with a bitwise-equality assertion and RNG-neutrality checks.

## Lookahead cadence/evaluation reconciliation (independently recomputed)

All internal numbers check out exactly:

| Claim | Recomputation | Match |
|---|---|---|
| syncs=5668, expected=5668 | ⌊28,341 / 5⌋ = 5,668 | ✓ |
| first=5, last=28,340, steps_since_sync=1 | 5×5,668 = 28,340; 28,341−28,340 = 1 | ✓ |
| Phase split 2,030+2,154+1,484 | = 5,668 total syncs | ✓ |
| 145 synchronized + 1 swapped evals | 145 full epochs × 195 steps = 28,275 (195 % 5 == 0 → synchronized); truncated epoch 146 ends at step 28,341 (not ÷5) → exactly one swap | ✓ |
| audit_samples=45 | sync 1 + syncs 128, 256, …, 5,632 (44 multiples) = 45 | ✓ |
| tail16 mean=94.7219, premium=0.0981 | Sum of 16 listed values = 1,515.55 → mean 94.721875; 94.82 − 94.7219 = 0.0981 | ✓ |

Restore failures = 0, nonfinite counts = 0, final normalized slow-fast gap 1.9e-5 (sensible one step after a sync).

## Is the no-improvement trustworthy?

Yes. This is a stable, well-measured negative, not an artifact: the final-16 plateau (94.61–94.82, mean 94.72) sits uniformly ~0.5 points below the parent, achieved step dose was *higher* than parent (28,341 vs 27,950; 146 vs 144 epochs — both above the 27,500/141 floors), CutMix exposure was on-target (0.4962 vs prob 0.5), and every mechanism assertion passed. The gap far exceeds the lineage's ~0.17-point selection-noise band. There is no incentive-gradient issue: the code faithfully implemented a preregistered intervention that simply hurt, and the agent reported it as a failure without retuning or rerunning — the opposite of reward hacking. Slow-weight evaluation was the preregistered semantics (analogous to accepted EMA evaluation in EXP-011); it uses the same frozen evaluator and, if anything, the run had no mechanism to inflate the metric.

## Minor observations (no correction required)

1. `cumulative_displacement_l2=18.36` sums only the 45 audited syncs, not all 5,668 — `03-execute.md` correctly labels it "cumulative **audited** displacement," so no misreporting, but readers should not treat it as total path length.
2. `run.log` is still present; the definition requires removal only "before beginning another experiment," and the loop is in the analyzing phase, so retention as raw evidence is compliant — it must be deleted before the next experiment starts.
3. The reported exit code 0 comes from the (unreproducible) exec session, but the complete final summary in the log independently satisfies the "complete final summary" necessary condition.

## Conclusion

Execution integrity, frozen budget, scope, cadence reconciliation, and the metric decision all verify independently. The recorded outcome — no-improvement at 94.82 vs the 95.33 threshold over parent 95.23 — is correct and trustworthy.

AUDIT_VERDICT: PASS
